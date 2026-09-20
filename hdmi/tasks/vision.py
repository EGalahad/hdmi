"""Camera-based observations for HDMI object tasks.

All terms read one scene-owned :class:`mjlab.sensor.CameraSensor` (see
``hdmi.sensors.head_camera``) that renders depth and per-pixel segmentation
once per control step for every environment (per-world object mesh variants
included).

* ``hdmi.head_depth`` - normalised depth image ``[N, 1, H, W]`` (uint8 by default).
* ``hdmi.object_pcd_camera`` - the object point cloud a real perception stack would
  produce: object mask from segmentation (a stand-in for a detector such as
  YOLO/SAM), randomised mask corruption, depth noise, unprojection with the pinhole
  intrinsics, transform into the robot's projected-yaw anchor frame (the same frame
  as ``hdmi.object_surface_points_local``) and a fixed number of points.
* ``hdmi.object_pcd_valid`` - ``[N, 1]`` flag, 1 when the object was visible.

Camera model (MuJoCo / mujoco_warp): pixel ``(u, v)`` with ``u`` increasing to the
right and ``v`` increasing downwards, camera looks along ``-Z`` with ``+Y`` up, so a
pixel at planar depth ``z`` unprojects to ``((u + 0.5 - cx) / fx * z,
-(v + 0.5 - cy) / fy * z, -z)``. Depth misses are written as ``0``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from mimic_lite.tasks.transforms import projected_yaw_quat

from active_adaptation.utils.math import quat_from_angle_axis, quat_rotate_inverse

from .observations import _single_object_observation

MJ_OBJ_GEOM = 5  # mujoco.mjtObj.mjOBJ_GEOM


def camera_intrinsics(width: int, height: int, fovy_deg: float) -> tuple[float, float, float, float]:
    """Return ``(fx, fy, cx, cy)`` in pixels for a MuJoCo camera (square pixels)."""
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return fy, fy, width / 2.0, height / 2.0


def pixel_directions(width: int, height: int, fovy_deg: float, device=None) -> torch.Tensor:
    """Per-pixel camera-frame directions scaled so that ``dir * planar_depth`` is the 3-D point.

    Returns ``[height, width, 3]`` with ``z = -1`` (MuJoCo camera looks along -Z).
    """
    fx, fy, cx, cy = camera_intrinsics(width, height, fovy_deg)
    u = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    v = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    dirs = torch.stack([(uu - cx) / fx, -(vv - cy) / fy, -torch.ones_like(uu)], dim=-1)
    return dirs


def world_points_to_anchor_frame(
    points_w: torch.Tensor, anchor_pos_w: torch.Tensor, anchor_quat_w: torch.Tensor
) -> torch.Tensor:
    """Express world points ``[N, P, 3]`` in the projected-yaw anchor frame.

    Same convention as Mimic-Lite's ``_body_pose_in_anchor_frame``: the anchor
    position with ``z`` zeroed and its yaw-only rotation.
    """
    anchor_pos_z0 = anchor_pos_w.clone()
    anchor_pos_z0[..., 2] = 0.0
    yaw_quat = projected_yaw_quat(anchor_quat_w)
    return quat_rotate_inverse(yaw_quat[:, None, :], points_w - anchor_pos_z0[:, None, :])


def select_fixed_points(
    points: torch.Tensor, mask: torch.Tensor, num_points: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick ``num_points`` of the masked points per row without any host sync.

    ``points`` is ``[N, M, 3]``, ``mask`` ``[N, M]`` (bool). Rows with more valid
    points than ``num_points`` get a uniformly random subset; rows with fewer repeat
    their valid points cyclically; rows with none return zeros. Also returns the
    valid count ``[N]``.
    """
    n, m = mask.shape
    if num_points > m:
        raise ValueError("num_points must not exceed the number of candidate points")
    count = mask.sum(dim=-1)  # [N]
    keys = torch.rand(n, m, device=points.device)
    keys = torch.where(mask, keys, torch.full_like(keys, 2.0))
    idx = keys.topk(num_points, dim=-1, largest=False).indices  # [N, P], valid first
    slot = torch.arange(num_points, device=points.device)[None].expand(n, -1)
    pos = torch.where(slot < count[:, None], slot, slot % count.clamp_min(1)[:, None])
    idx = idx.gather(1, pos)
    selected = points.gather(1, idx[..., None].expand(-1, -1, points.shape[-1]))
    selected = torch.where((count > 0)[:, None, None], selected, torch.zeros_like(selected))
    return selected, count


class _head_camera_observation(_single_object_observation):
    """Shared camera access, intrinsics and object geom ids."""

    def _initialize_camera(self, sensor_name: str) -> None:
        self.sensor_name = sensor_name
        self.sensor = self.env.scene.sensors[sensor_name]
        self.env.sensor_render_enabled = True
        cfg = self.sensor.cfg
        self.width = int(cfg.width)
        self.height = int(cfg.height)
        self.fovy = float(cfg.fovy)
        self.dirs = pixel_directions(self.width, self.height, self.fovy, device=self.device)
        obj = self.command_manager.object
        self.object_geom_ids = torch.as_tensor(
            obj.indexing.geom_ids, device=self.device, dtype=torch.int32
        )
        self._sim_data = obj.data.data

    @property
    def cam_idx(self) -> int:
        return int(self.sensor.camera_idx)

    def _depth(self) -> torch.Tensor:
        depth = self.sensor.data.depth
        if depth is None:
            raise RuntimeError(f"Sensor {self.sensor_name!r} does not render depth")
        return depth[..., 0]  # [N, H, W], planar depth in metres, 0 = miss

    def _object_mask(self, depth: torch.Tensor, cutoff: float) -> torch.Tensor:
        seg = self.sensor.data.segmentation
        if seg is None:
            raise RuntimeError(f"Sensor {self.sensor_name!r} does not render segmentation")
        is_geom = seg[..., 1] == MJ_OBJ_GEOM
        is_object = torch.isin(seg[..., 0], self.object_geom_ids)
        return is_geom & is_object & (depth > 0.0) & (depth < cutoff)

    def _camera_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World camera position ``[N, 3]`` and rotation ``[N, 3, 3]`` (columns = camera axes)."""
        pos = self._sim_data.cam_xpos[:, self.cam_idx]
        mat = self._sim_data.cam_xmat[:, self.cam_idx].reshape(-1, 3, 3)
        return pos, mat


class head_depth(_head_camera_observation, namespace="hdmi"):
    """Normalised head-camera depth image ``[N, 1, H, W]``.

    Misses (depth 0) are treated as the far plane. Output is ``uint8`` in
    ``[0, 255]`` by default (``depth / cutoff_m``), which keeps the rollout buffer
    small; the learner upcasts.
    """

    def _initialize_impl(
        self,
        sensor_name: str = "head_camera",
        cutoff_m: float = 3.0,
        min_depth_m: float = 0.15,
        noise_std_m: float = 0.0,
        noise_quad: float = 0.0,
        output_dtype: str = "uint8",
    ) -> None:
        super()._initialize_impl()
        self._initialize_camera(sensor_name)
        self.cutoff = float(cutoff_m)
        self.min_depth = float(min_depth_m)
        self.noise_std = float(noise_std_m)
        self.noise_quad = float(noise_quad)
        if output_dtype not in ("uint8", "float32"):
            raise ValueError(f"output_dtype must be 'uint8' or 'float32', got {output_dtype!r}")
        self.output_dtype = output_dtype

    def compute(self) -> torch.Tensor:
        z = self._depth()
        z = torch.where(z <= 0.0, torch.full_like(z, self.cutoff), z)
        if self.noise_std > 0.0 or self.noise_quad > 0.0:
            sigma = self.noise_std + self.noise_quad * z * z
            z = z + torch.randn_like(z) * sigma
        z = z.clamp(self.min_depth, self.cutoff) / self.cutoff
        if self.output_dtype == "uint8":
            return (z * 255.0).round().to(torch.uint8)[:, None]
        return z[:, None]


class object_pcd_camera(_head_camera_observation, namespace="hdmi"):
    """Object point cloud as a real depth-camera + detector pipeline would produce it.

    Output ``[N, num_points * 3]`` in the projected-yaw anchor frame (same layout and
    frame as ``hdmi.object_surface_points_local`` so encoders are interchangeable).
    """

    def _initialize_impl(
        self,
        sensor_name: str = "head_camera",
        num_points: int = 256,
        cutoff_m: float = 3.0,
        depth_noise_std_m: float = 0.002,
        depth_noise_quad: float = 0.0025,
        depth_quant_m: float = 0.004,
        mask_corrupt_prob: float = 0.5,
        mask_erode_dilate_max: int = 1,
        pixel_dropout: float = 0.1,
        false_negative_prob: float = 0.05,
        extrinsic_rot_jitter_deg: float = 2.0,
        extrinsic_pos_jitter_m: float = 0.01,
        randomize: bool = True,
    ) -> None:
        super()._initialize_impl()
        self._initialize_camera(sensor_name)
        self.num_points = int(num_points)
        if self.num_points > self.width * self.height:
            raise ValueError("num_points must not exceed the number of camera pixels")
        self.cutoff = float(cutoff_m)
        self.depth_noise_std = float(depth_noise_std_m)
        self.depth_noise_quad = float(depth_noise_quad)
        self.depth_quant = float(depth_quant_m)
        self.mask_corrupt_prob = float(mask_corrupt_prob)
        self.mask_kernel_max = int(mask_erode_dilate_max)
        self.pixel_dropout = float(pixel_dropout)
        self.false_negative_prob = float(false_negative_prob)
        self.rot_jitter = math.radians(float(extrinsic_rot_jitter_deg))
        self.pos_jitter = float(extrinsic_pos_jitter_m)
        self.randomize = bool(randomize)

        n = self.num_envs
        # per-episode corruption state
        self.mask_mode = torch.zeros(n, dtype=torch.long, device=self.device)  # 0 none, 1 dilate, 2 erode
        self.false_negative = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.jitter_rot = torch.eye(3, device=self.device).expand(n, 3, 3).clone()
        self.jitter_pos = torch.zeros(n, 3, device=self.device)
        # outputs shared with object_pcd_valid
        self.valid = torch.zeros(n, 1, device=self.device)
        self.num_visible = torch.zeros(n, dtype=torch.long, device=self.device)
        self.command_manager.head_camera_pcd = self
        self.reset(torch.arange(n, device=self.device), None)

    def reset(self, env_ids, tensordict=None) -> None:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        k = env_ids.numel()
        if k == 0:
            return
        if not self.randomize:
            self.mask_mode[env_ids] = 0
            self.false_negative[env_ids] = False
            self.jitter_rot[env_ids] = torch.eye(3, device=self.device)
            self.jitter_pos[env_ids] = 0.0
            return
        corrupt = torch.rand(k, device=self.device) < self.mask_corrupt_prob
        mode = torch.randint(1, 3, (k,), device=self.device)
        self.mask_mode[env_ids] = torch.where(corrupt, mode, torch.zeros_like(mode))
        self.false_negative[env_ids] = torch.rand(k, device=self.device) < self.false_negative_prob
        axis = torch.randn(k, 3, device=self.device)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        angle = (torch.rand(k, device=self.device) * 2.0 - 1.0) * self.rot_jitter
        quat = quat_from_angle_axis(angle, axis)
        self.jitter_rot[env_ids] = _matrix_from_quat(quat)
        self.jitter_pos[env_ids] = (torch.rand(k, 3, device=self.device) * 2.0 - 1.0) * self.pos_jitter

    def _corrupt_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if not self.randomize:
            return mask
        m = mask.float()[:, None]  # [N, 1, H, W]
        k = 2 * self.mask_kernel_max + 1
        if self.mask_kernel_max > 0:
            dilated = F.max_pool2d(m, k, stride=1, padding=k // 2)
            eroded = 1.0 - F.max_pool2d(1.0 - m, k, stride=1, padding=k // 2)
            mode = self.mask_mode[:, None, None, None]
            m = torch.where(mode == 1, dilated, torch.where(mode == 2, eroded, m))
        mask = m[:, 0] > 0.5
        if self.pixel_dropout > 0.0:
            mask = mask & (torch.rand_like(m[:, 0]) >= self.pixel_dropout)
        mask = mask & ~self.false_negative[:, None, None]
        return mask

    def _noisy_depth(self, z: torch.Tensor) -> torch.Tensor:
        if not self.randomize:
            return z
        if self.depth_noise_std > 0.0 or self.depth_noise_quad > 0.0:
            sigma = self.depth_noise_std + self.depth_noise_quad * z * z
            z = z + torch.randn_like(z) * sigma
        if self.depth_quant > 0.0:
            z = torch.round(z / self.depth_quant) * self.depth_quant
        return z

    def compute(self) -> torch.Tensor:
        n = self.num_envs
        z = self._depth()  # [N, H, W]
        mask = self._corrupt_mask(self._object_mask(z, self.cutoff))
        z = self._noisy_depth(z)

        # unproject every pixel (cheap: N x H x W x 3) and move to the world frame
        p_cam = self.dirs[None] * z[..., None]  # [N, H, W, 3]
        p_cam = p_cam.reshape(n, -1, 3)
        cam_pos, cam_rot = self._camera_pose_w()
        if self.randomize:
            cam_rot = cam_rot @ self.jitter_rot
            cam_pos = cam_pos + torch.einsum("nij,nj->ni", cam_rot, self.jitter_pos)
        p_w = torch.einsum("nij,npj->npi", cam_rot, p_cam) + cam_pos[:, None, :]
        command = self.command_manager
        p_anchor = world_points_to_anchor_frame(
            p_w, command.robot_anchor_pos_w, command.robot_anchor_quat_w
        )  # [N, HW, 3]

        points, count = select_fixed_points(p_anchor, mask.reshape(n, -1), self.num_points)
        self.num_visible = count
        self.valid = (count > 0).float()[:, None]
        return points.reshape(n, -1)


class object_pcd_valid(_single_object_observation, namespace="hdmi"):
    """``[N, 1]`` flag: 1 if the camera point cloud saw the object this step."""

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        source = getattr(self.command_manager, "head_camera_pcd", None)
        if source is None:
            raise ValueError(
                "object_pcd_valid requires hdmi.object_pcd_camera in an observation group "
                "declared before this one"
            )
        self.source = source

    def compute(self) -> torch.Tensor:
        return self.source.valid


def _matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    from active_adaptation.utils.math import matrix_from_quat

    return matrix_from_quat(quat)


__all__ = [
    "camera_intrinsics",
    "pixel_directions",
    "select_fixed_points",
    "world_points_to_anchor_frame",
    "head_depth",
    "object_pcd_camera",
    "object_pcd_valid",
]

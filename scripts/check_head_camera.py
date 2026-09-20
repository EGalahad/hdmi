"""Render one frame of the head camera and sanity-check orientation, mask and unprojection.

Usage (from the active-adaptation root, MjLab venv)::

    uv run --project venv/mjlab python projects/hdmi/scripts/check_head_camera.py \
        task=omomo-rigid5-pcd-camera task.num_envs=4 +out_dir=/tmp/head_camera

Writes ``depth_env0.png`` / ``mask_env0.png`` / ``pcd_env0.png`` into ``out_dir`` and
prints: camera pose, forward-axis alignment with the torso +X axis, fraction of
object pixels, and the distance between the unprojected object points and the
ground-truth surface points (``object_surface_points_local``) in the same frame.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import active_adaptation  # noqa: F401  # registers the Hydra search-path plugin / learning modules

FILE_PATH = Path(__file__).resolve()
MIMIC_LITE_CFG = FILE_PATH.parents[3] / "projects" / "mimic-lite" / "cfg"


def _save_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(image).save(path)


@hydra.main(version_base=None, config_path=str(MIMIC_LITE_CFG), config_name="train")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    active_adaptation.init(cfg, auto_rank=True)  # backend + project registries, as in train.py
    from active_adaptation.helpers import make_env_policy

    cfg.headless = True
    cfg.wandb.mode = "disabled"
    out_dir = Path(cfg.get("out_dir", "/tmp/head_camera"))
    out_dir.mkdir(parents=True, exist_ok=True)
    randomize_key = "task.observation.object_pcd.surface_points.randomize"
    OmegaConf.update(cfg, randomize_key, False, merge=True) if OmegaConf.select(
        cfg, "task.observation.object_pcd.surface_points"
    ) is not None else None

    env, policy = make_env_policy(cfg.task, cfg.algo, cfg.seed, cfg.headless, cfg.get("device", "cuda:0"))
    # observation groups not consumed by ``algo.in_keys`` are pruned, so make sure the
    # camera is rendered regardless of which algo the config selected
    env.sensor_render_enabled = True
    td = env.reset()
    for _ in range(5):
        td["action"] = torch.zeros(env.num_envs, env.action_manager.action_dim, device=env.device)
        td = env.step(td)["next"]

    sensor = env.scene.sensors["head_camera"]
    depth = sensor.data.depth[..., 0]  # [N, H, W]
    seg = sensor.data.segmentation
    command = env.command_manager
    if hasattr(command, "get_term"):
        command = command.get_term("command") if not hasattr(command, "object") else command
    obj = command.object
    geom_ids = torch.as_tensor(obj.indexing.geom_ids, device=depth.device, dtype=torch.int32)
    mask = (seg[..., 1] == 5) & torch.isin(seg[..., 0], geom_ids) & (depth > 0)

    sim_data = obj.data.data
    cam = sensor.camera_idx
    cam_pos = sim_data.cam_xpos[:, cam]
    cam_mat = sim_data.cam_xmat[:, cam].reshape(-1, 3, 3)
    forward_w = -cam_mat[:, :, 2]
    from active_adaptation.utils.math import matrix_from_quat

    torso_x_w = matrix_from_quat(command.robot_anchor_quat_w)[:, :, 0]
    align = (forward_w * torso_x_w).sum(-1)
    print(f"[camera] pos env0 = {cam_pos[0].tolist()}")
    print(f"[camera] forward.dot(torso_x) per env = {align.tolist()}  (expect ~1 for pitch 0)")
    print(f"[camera] unique seg ids env0 = {seg[0, ..., 0].unique().tolist()}  object geom ids = {geom_ids.tolist()}")
    frac = mask.float().mean(dim=(1, 2))
    print(f"[camera] object pixel fraction per env = {frac.tolist()}")
    valid = depth[0] > 0
    rows = torch.arange(depth.shape[1], device=depth.device)[:, None].expand_as(depth[0])
    lower = valid & (rows >= depth.shape[1] // 2)
    print(f"[camera] depth env0: valid px {int(valid.sum())}/{valid.numel()}, seg types {seg[0, ..., 1].unique().tolist()}")
    if valid.any():
        print(f"[camera] depth env0: min {depth[0][valid].min():.3f} max {depth[0][valid].max():.3f}; lower-half mean {depth[0][lower].mean():.3f}")
    else:
        print("[camera] WARNING: depth is all zero -> the camera was not rendered (sensor_render_enabled / sim.sense())")

    d = depth[0].clone()
    d[d <= 0] = 3.0
    _save_png(out_dir / "depth_env0.png", (d.clamp(0, 3) / 3 * 255).to(torch.uint8).cpu().numpy())
    _save_png(out_dir / "mask_env0.png", (mask[0].float() * 255).to(torch.uint8).cpu().numpy())

    # unprojection round trip against the GT surface cloud (same anchor frame)
    from hdmi.tasks.vision import pixel_directions, world_points_to_anchor_frame

    dirs = pixel_directions(sensor.cfg.width, sensor.cfg.height, float(sensor.cfg.fovy), device=depth.device)
    p_cam = (dirs[None] * depth[..., None]).reshape(env.num_envs, -1, 3)
    p_w = torch.einsum("nij,npj->npi", cam_mat, p_cam) + cam_pos[:, None, :]
    p_anchor = world_points_to_anchor_frame(p_w, command.robot_anchor_pos_w, command.robot_anchor_quat_w)
    if command.object_surface_points is not None:
        from hdmi.tasks.observations import _transform_object_points

        idx = command.tracking_body_names.index(command.object_tracking_body_names[0])
        gt = _transform_object_points(
            command.object_surface_points[command.object_variant_ids],
            command.robot_body_pos_local[:, idx],
            command.robot_body_quat_local[:, idx],
        )  # [N, 256, 3]
        for e in range(env.num_envs):
            pts = p_anchor[e][mask[e].reshape(-1)]
            if pts.numel() == 0:
                print(f"[roundtrip] env{e}: object not visible")
                continue
            dist = torch.cdist(pts, gt[e]).min(dim=1).values
            print(
                f"[roundtrip] env{e}: {pts.shape[0]} object px, nearest-GT-point distance "
                f"median {dist.median():.4f} m, p90 {dist.quantile(0.9):.4f} m"
            )
    np.save(out_dir / "pcd_env0.npy", p_anchor[0][mask[0].reshape(-1)].cpu().numpy())
    print(f"[done] wrote {out_dir}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # skip slow warp/torch teardown


if __name__ == "__main__":
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra.run.dir=.")
    main()

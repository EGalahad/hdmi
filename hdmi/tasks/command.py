from __future__ import annotations

from typing import List

import torch

from active_adaptation.envs.utils import find_bodies, find_joints
from active_adaptation.utils.string import resolve_matching_names
from mimic_lite.tasks.command import RobotTracking


def _resolve_unique(
    requested: List[str] | str,
    available: list[str],
    *,
    label: str,
) -> tuple[list[int], list[str]]:
    _, names = resolve_matching_names(requested, available)
    if not names:
        raise ValueError(f"No {label} matched {requested!r} in {available}")
    return [available.index(name) for name in names], list(names)


class RobotObjectTracking(RobotTracking, namespace="hdmi"):
    """Track one combined-qpos reference using separate robot/object entities."""

    def __init__(
        self,
        *,
        object_name: str,
        object_root_body_name: str,
        object_tracking_body_names: List[str],
        object_tracking_joint_names: List[str] | None = None,
        call_update: bool = True,
        **kwargs,
    ) -> None:
        self.object_name = object_name
        self.object_root_body_name = object_root_body_name
        self._object_tracking_body_names_cfg = list(object_tracking_body_names)
        self._object_tracking_joint_names_cfg = list(object_tracking_joint_names or ())
        self._hdmi_call_update = call_update
        extra_bodies = list(
            dict.fromkeys([*self._object_tracking_body_names_cfg, object_root_body_name])
        )
        super().__init__(
            extra_motion_body_names=extra_bodies,
            extra_motion_joint_names=self._object_tracking_joint_names_cfg,
            call_update=False,
            **kwargs,
        )

    def _initialize(self, env) -> None:
        self.object = env.scene[self.object_name]
        robot = env.scene.articulations["robot"]
        _, robot_body_names = find_bodies(robot, self._tracking_body_names_cfg)
        _, robot_joint_names = find_joints(robot, self._tracking_joint_names_cfg)
        object_body_ids, object_body_names = _resolve_unique(
            self._object_tracking_body_names_cfg,
            list(self.object.body_names),
            label="object body",
        )
        object_joint_ids: list[int] = []
        object_joint_names: list[str] = []
        if self._object_tracking_joint_names_cfg:
            object_joint_ids, object_joint_names = _resolve_unique(
                self._object_tracking_joint_names_cfg,
                list(self.object.joint_names),
                label="object joint",
            )

        if self.object_root_body_name not in self.object.body_names:
            raise ValueError(
                f"Object root body {self.object_root_body_name!r} is missing from "
                f"runtime object bodies {list(self.object.body_names)}"
            )

        duplicate_bodies = set(robot_body_names) & set(object_body_names)
        duplicate_joints = set(robot_joint_names) & set(object_joint_names)
        if duplicate_bodies or duplicate_joints:
            raise ValueError(
                "Robot/object logical tracking names must be disjoint; "
                f"duplicate bodies={sorted(duplicate_bodies)}, "
                f"duplicate joints={sorted(duplicate_joints)}"
            )

        self._extra_motion_body_names = list(
            dict.fromkeys([*object_body_names, self.object_root_body_name])
        )
        self._extra_motion_joint_names = list(object_joint_names)
        super()._initialize(env)

        if self.tracking_body_names != list(robot_body_names):
            raise RuntimeError("Robot tracking body order changed during initialization")
        if self.tracking_joint_names != list(robot_joint_names):
            raise RuntimeError("Robot tracking joint order changed during initialization")

        self.robot_tracking_body_names = list(self.tracking_body_names)
        self.robot_tracking_joint_names = list(self.tracking_joint_names)
        missing_reference_bodies = sorted(
            set([*object_body_names, self.object_root_body_name])
            - set(self.dataset.body_names)
        )
        missing_reference_joints = sorted(
            set(object_joint_names) - set(self.dataset.joint_names)
        )
        if missing_reference_bodies or missing_reference_joints:
            raise ValueError(
                "Combined reference is missing configured object names; "
                f"bodies={missing_reference_bodies}, joints={missing_reference_joints}"
            )

        self.object_tracking_body_names = object_body_names
        self.object_tracking_joint_names = object_joint_names
        self.object_tracking_body_indices_asset = object_body_ids
        self.object_tracking_joint_indices_asset = object_joint_ids
        self.object_tracking_body_indices_motion = [
            self.dataset.body_names.index(name) for name in object_body_names
        ]
        self.object_tracking_joint_indices_motion = [
            self.dataset.joint_names.index(name) for name in object_joint_names
        ]
        self.object_root_body_idx_motion = self.dataset.body_names.index(
            self.object_root_body_name
        )

        self.tracking_body_names.extend(object_body_names)
        self.tracking_joint_names.extend(object_joint_names)
        self.tracking_body_indices_motion.extend(
            self.object_tracking_body_indices_motion
        )
        self.tracking_joint_indices_motion.extend(
            self.object_tracking_joint_indices_motion
        )

        if self._hdmi_call_update:
            self._read_current_robot_state()
            self._refresh_future_buffers()
            self.update()

    def _read_current_robot_state(self) -> None:
        super()._read_current_robot_state()
        data = self.object.data
        ids = self.object_tracking_body_indices_asset
        self.robot_body_link_pos_w = torch.cat(
            [self.robot_body_link_pos_w, data.body_link_pos_w[:, ids]], dim=1
        )
        self.robot_body_lin_vel_w = torch.cat(
            [self.robot_body_lin_vel_w, data.body_com_lin_vel_w[:, ids]], dim=1
        )
        self.robot_body_link_quat_w = torch.cat(
            [self.robot_body_link_quat_w, data.body_link_quat_w[:, ids]], dim=1
        )
        self.robot_body_ang_vel_w = torch.cat(
            [self.robot_body_ang_vel_w, data.body_com_ang_vel_w[:, ids]], dim=1
        )
        if self.object_tracking_joint_indices_asset:
            joint_ids = self.object_tracking_joint_indices_asset
            object_joint_pos = data.joint_pos[:, joint_ids]
            object_joint_vel = data.joint_vel[:, joint_ids]
        else:
            object_joint_pos = self.robot_joint_pos.new_empty(self.num_envs, 0)
            object_joint_vel = self.robot_joint_vel.new_empty(self.num_envs, 0)
        self.robot_joint_pos = torch.cat(
            [self.robot_joint_pos, object_joint_pos], dim=1
        )
        self.robot_joint_vel = torch.cat(
            [self.robot_joint_vel, object_joint_vel], dim=1
        )

    def _write_object_reference(
        self,
        motion,
        env_ids: torch.Tensor,
        time_index: int = 0,
    ) -> None:
        pos = motion.body_pos_w[:, time_index, self.object_root_body_idx_motion]
        pos = pos + self.env.scene.env_origins[env_ids]
        quat = motion.body_quat_w[:, time_index, self.object_root_body_idx_motion]
        lin_vel = motion.body_lin_vel_w[:, time_index, self.object_root_body_idx_motion]
        ang_vel = motion.body_ang_vel_w[:, time_index, self.object_root_body_idx_motion]
        is_fixed = bool(getattr(self.object, "is_fixed_base", False))
        if not is_fixed:
            self.object.write_root_link_pose_to_sim(
                torch.cat([pos, quat], dim=-1), env_ids=env_ids
            )
            self.object.write_root_com_velocity_to_sim(
                torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids
            )
        if self.object_tracking_joint_indices_asset:
            motion_ids = self.object_tracking_joint_indices_motion
            self.object.write_joint_state_to_sim(
                motion.joint_pos[:, time_index, motion_ids],
                motion.joint_vel[:, time_index, motion_ids],
                joint_ids=self.object_tracking_joint_indices_asset,
                env_ids=env_ids,
            )

    def sample_init(self, env_ids: torch.Tensor, reset_td=None) -> None:
        super().sample_init(env_ids, reset_td)
        motion = self.dataset.get_slice(
            self.motion_ids[env_ids], self.t[env_ids], self.future_one_step
        ).to(self.device)
        self._write_object_reference(motion, env_ids)

    def update(self) -> None:
        if self.replay_motion:
            self._write_object_reference(
                self.future_ref_motion,
                self.all_env_ids,
                self.obs_current_step_index,
            )
        super().update()

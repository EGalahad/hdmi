from __future__ import annotations

import torch

from active_adaptation.utils.math import matrix_from_quat, quat_conjugate, quat_mul
from mimic_lite.tasks.deferred import DeferredObservation as BaseObservation
from mimic_lite.tasks.transforms import _body_pose_in_anchor_frame

from .command import RobotObjectTracking


def _spatial_motion_from_local_poses(
    position: torch.Tensor,
    rotation: torch.Tensor,
    current_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``T(h) @ inv(T(current))`` for poses in one anchor frame."""
    rotation_current = rotation[:, current_index]
    position_current = position[:, current_index]
    spatial_rotation = rotation @ rotation_current.transpose(-1, -2).unsqueeze(1)
    spatial_position = position - torch.einsum(
        "nhij,nj->nhi", spatial_rotation, position_current
    )
    return spatial_position, spatial_rotation


ObjectObservation = BaseObservation[RobotObjectTracking]


class _single_object_observation(ObjectObservation):
    def _initialize_impl(self) -> None:
        names = self.command_manager.object_tracking_body_names
        if len(names) != 1:
            raise ValueError(
                "Phase 1 object observations require exactly one tracked body, "
                f"got {names}"
            )
        self.object_index = self.command_manager.tracking_body_names.index(names[0])


class object_pose_local(_single_object_observation, namespace="hdmi"):
    """Actual object pose in the actual robot projected-yaw anchor."""

    def compute(self) -> torch.Tensor:
        position = self.command_manager.robot_body_pos_local[:, self.object_index]
        rotation = matrix_from_quat(
            self.command_manager.robot_body_quat_local[:, self.object_index]
        )
        return torch.cat([position, rotation[:, :2, :].reshape(self.num_envs, 6)], -1)


class object_spatial_motion_local(_single_object_observation, namespace="hdmi"):
    """Reference object spatial motion in the current reference robot anchor."""

    def compute(self) -> torch.Tensor:
        command = self.command_manager
        current_index = command.obs_current_step_index
        position, quaternion = _body_pose_in_anchor_frame(
            command.ref_anchor_pos_future_w[:, current_index, None],
            command.ref_anchor_quat_future_w[:, current_index, None],
            command.ref_body_pos_future_w[:, :, self.object_index],
            command.ref_body_quat_future_w[:, :, self.object_index],
        )
        spatial_position, spatial_rotation = _spatial_motion_from_local_poses(
            position,
            matrix_from_quat(quaternion),
            current_index,
        )
        rotation_6d = spatial_rotation[:, :, :2, :].reshape(self.num_envs, -1, 6)
        return torch.cat([spatial_position, rotation_6d], -1).reshape(
            self.num_envs, -1
        )


class object_pose_error_local(_single_object_observation, namespace="hdmi"):
    """Critic-only object pose error using Mimic-Lite body-local semantics."""

    def compute(self) -> torch.Tensor:
        command = self.command_manager
        index = self.object_index
        position = command.ref_body_pos_local[:, index] - command.robot_body_pos_local[
            :, index
        ]
        quaternion = quat_mul(
            quat_conjugate(command.ref_body_quat_local[:, index]),
            command.robot_body_quat_local[:, index],
        )
        rotation = matrix_from_quat(quaternion)
        return torch.cat([position, rotation[:, :2, :].reshape(self.num_envs, 6)], -1)

from __future__ import annotations

import torch
from mimic_lite.tasks.deferred import DeferredObservation as BaseObservation

from active_adaptation.utils.math import matrix_from_quat, quat_rotate

from .command import RobotObjectTracking

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


def _transform_object_points(
    points: torch.Tensor,
    position: torch.Tensor,
    quaternion: torch.Tensor,
) -> torch.Tensor:
    return quat_rotate(quaternion[:, None, :], points) + position[:, None, :]


class object_surface_points_local(_single_object_observation, namespace="hdmi"):
    """Fixed object-surface points in the actual projected-yaw robot frame."""

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        command = self.command_manager
        if command.object_surface_points is None or command.object_variant_ids is None:
            raise ValueError("object_surface_points_local requires object mesh variants")

    def compute(self) -> torch.Tensor:
        command = self.command_manager
        points = command.object_surface_points[command.object_variant_ids]
        position = command.robot_body_pos_local[:, self.object_index]
        quaternion = command.robot_body_quat_local[:, self.object_index]
        return _transform_object_points(points, position, quaternion).flatten(1)


class object_category(ObjectObservation, namespace="hdmi"):
    """Fixed per-environment object variant ID for category metrics."""

    def _initialize_impl(self) -> None:
        if self.command_manager.object_variant_ids is None:
            raise ValueError("object_category requires object mesh variants")

    def compute(self) -> torch.Tensor:
        return self.command_manager.object_variant_ids[:, None].float()


class object_motion_progress(ObjectObservation, namespace="hdmi"):
    """Reference-motion progress used only for per-category diagnostics."""

    def compute(self) -> torch.Tensor:
        command = self.command_manager
        return (command.t.float() / command.motion_len.clamp_min(1)).unsqueeze(-1)

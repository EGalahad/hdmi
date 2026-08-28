from __future__ import annotations

import torch

from active_adaptation.utils.math import matrix_from_quat
from mimic_lite.tasks.deferred import DeferredObservation as BaseObservation

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

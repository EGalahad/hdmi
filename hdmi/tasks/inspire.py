from __future__ import annotations

import torch
from mimic_lite.tasks.actions import JointPosition


class InspireJointPosition(JointPosition, namespace="hdmi"):
    """Keep MimicLite delay/smoothing and write the official Inspire mimic targets."""

    def __init__(self, *, mimic: dict[str, tuple[str, float, float]], **kwargs):
        super().__init__(**kwargs)
        self.mimic = mimic

    def _initialize(self, env) -> None:
        super()._initialize(env)
        targets = list(self.mimic)
        drivers = [value[0] for value in self.mimic.values()]
        if set(targets) & set(self.joint_names):
            raise ValueError("Inspire coupled targets must be excluded from independent actions")
        self.mimic_joint_ids = torch.tensor(
            [self.asset.joint_names.index(name) for name in targets], device=self.device
        )
        self.driver_joint_ids = torch.tensor(
            [self.asset.joint_names.index(name) for name in drivers], device=self.device
        )
        self.driver_action_ids = torch.tensor(
            [self.joint_names.index(name) for name in drivers], device=self.device
        )
        self.mimic_multipliers = torch.tensor(
            [value[1] for value in self.mimic.values()], device=self.device
        )
        self.mimic_offsets = torch.tensor(
            [value[2] for value in self.mimic.values()], device=self.device
        )

    def apply_action(self, substep: int):
        super().apply_action(substep)
        driver_targets = (
            self.default_joint_pos[:, self.driver_joint_ids]
            + self.offset[:, self.driver_joint_ids]
            + self.applied_action[:, self.driver_action_ids]
            * self.action_scaling[self.driver_action_ids]
        )
        self.asset.set_joint_position_target(
            driver_targets * self.mimic_multipliers + self.mimic_offsets,
            joint_ids=self.mimic_joint_ids,
        )

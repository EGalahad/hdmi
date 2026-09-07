"""MDP components for the standalone G1 + Sharpa control task."""

from __future__ import annotations

from active_adaptation.envs.mdp.actions.composite import ConcatenatedAction
from active_adaptation.envs.mdp.commands.base import Command
from tensordict import TensorDictBase

from hdmi.tasks.command import RobotObjectTracking

from hdmi.g1_sharpa.asset import (
    G1_JOINT_NAMES,
    LEFT_SHARPA_JOINT_NAMES,
    RIGHT_SHARPA_JOINT_NAMES,
)


class G1SharpaJointPositionAction(ConcatenatedAction):
    """Deterministic body/left-hand/right-hand joint-position action."""

    namespace = "hdmi"
    supported_backends = ("mjlab",)

    def __init__(
        self,
        body_action_scaling: float = 0.1,
        hand_action_scaling: float = 0.12,
    ) -> None:
        actions = (
            {
                "_target_": "JointPosition",
                "action_scaling": {
                    name: body_action_scaling for name in G1_JOINT_NAMES
                },
                "max_delay": 0,
                "alpha_range": (1.0, 1.0),
            },
            {
                "_target_": "JointPosition",
                "action_scaling": {
                    name: hand_action_scaling for name in LEFT_SHARPA_JOINT_NAMES
                },
                "max_delay": 0,
                "alpha_range": (1.0, 1.0),
            },
            {
                "_target_": "JointPosition",
                "action_scaling": {
                    name: hand_action_scaling for name in RIGHT_SHARPA_JOINT_NAMES
                },
                "max_delay": 0,
                "alpha_range": (1.0, 1.0),
            },
        )
        super().__init__(actions=list(actions))


class StaticPoseCommand(Command):
    """Initialize the robot at its default pose without external task data."""

    namespace = "hdmi"
    supported_backends = ("mjlab",)

    def update(self) -> None:
        pass


class RobotObjectTrackingFixedSharpa(RobotObjectTracking):
    """Track the 29-DoF G1 reference while holding both Sharpa hands fixed."""

    namespace = "hdmi"
    supported_backends = ("mjlab",)

    def __init__(self, hand_input_key: str = "hand_control", **kwargs) -> None:
        self.hand_input_key = hand_input_key
        super().__init__(motion_asset_joint_names=list(G1_JOINT_NAMES), **kwargs)

    def prescribe(self, tensordict: TensorDictBase) -> None:
        if tensordict.get(self.hand_input_key) is None:
            tensordict.set(
                self.hand_input_key,
                self.asset.data.joint_pos.new_zeros(
                    self.num_envs,
                    len(LEFT_SHARPA_JOINT_NAMES) + len(RIGHT_SHARPA_JOINT_NAMES),
                ),
            )


__all__ = [
    "G1SharpaJointPositionAction",
    "RobotObjectTrackingFixedSharpa",
    "StaticPoseCommand",
]

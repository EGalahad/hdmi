"""Active Adaptation asset registration for G1 with dual Sharpa Hands."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from active_adaptation.registry import Registry
from active_adaptation.assets.humanoids.g1 import JOINT_NAMES_SIMULATION
from mjlab.actuator import XmlActuatorCfg

from hdmi.g1_sharpa.spec import make_g1_sharpa_spec

if TYPE_CHECKING:
    from active_adaptation.assets.asset_cfg import AssetSpec

_SUPPORTED_MODES = (5, 11, 13, 15)

G1_JOINT_NAMES = tuple(JOINT_NAMES_SIMULATION)
LEFT_SHARPA_JOINT_NAMES = (
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
)
RIGHT_SHARPA_JOINT_NAMES = tuple(
    name.replace("left_", "right_", 1) for name in LEFT_SHARPA_JOINT_NAMES
)
G1_SHARPA_JOINT_NAMES = (
    G1_JOINT_NAMES + LEFT_SHARPA_JOINT_NAMES + RIGHT_SHARPA_JOINT_NAMES
)

BODY_ACTION_SLICE = slice(0, len(G1_JOINT_NAMES))
LEFT_HAND_ACTION_SLICE = slice(
    BODY_ACTION_SLICE.stop,
    BODY_ACTION_SLICE.stop + len(LEFT_SHARPA_JOINT_NAMES),
)
RIGHT_HAND_ACTION_SLICE = slice(
    LEFT_HAND_ACTION_SLICE.stop,
    LEFT_HAND_ACTION_SLICE.stop + len(RIGHT_SHARPA_JOINT_NAMES),
)
ACTION_GROUPS = {
    "body_action": G1_JOINT_NAMES,
    "left_hand_action": LEFT_SHARPA_JOINT_NAMES,
    "right_hand_action": RIGHT_SHARPA_JOINT_NAMES,
}


def _prefixed_names(spec, collection: str) -> list[str]:
    """Return attached hand names in spec order with MjSpec prefixes removed."""
    elements = getattr(spec, collection)
    return [
        element.name.split("/")[-1]
        for element in elements
        if element.name.startswith(("lh/", "rh/"))
        and element.name not in ("lh/world", "rh/world")
    ]


def make_g1_sharpa_asset(backend: str, *, mode: int = 15) -> "AssetSpec":
    """Build the MJLab AssetSpec for a G1 with two full-DoF Sharpa Hands."""
    if backend != "mjlab":
        raise NotImplementedError(
            "G1 with Sharpa Hands currently supports only the MjLab backend"
        )
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"Unsupported G1 mode {mode}; expected one of {_SUPPORTED_MODES}")

    from active_adaptation.assets.asset_cfg import AssetSpec

    base_factory = Registry.instance().get("asset", f"g1-mode_{mode}")
    base_asset = base_factory(backend=backend)
    base_cfg = base_asset.config
    if base_cfg.articulation is None:
        raise ValueError("The base G1 asset must define an articulation")
    if tuple(base_cfg.joint_names_simulation or ()) != G1_JOINT_NAMES:
        raise ValueError("Base G1 joint order differs from G1_JOINT_NAMES")

    combined_spec = make_g1_sharpa_spec(mode)
    hand_joint_names = _prefixed_names(combined_spec, "joints")
    hand_body_names = _prefixed_names(combined_spec, "bodies")
    actual_left_hand_joint_names = tuple(
        name for name in hand_joint_names if name.startswith("left_")
    )
    actual_right_hand_joint_names = tuple(
        name for name in hand_joint_names if name.startswith("right_")
    )
    if actual_left_hand_joint_names != LEFT_SHARPA_JOINT_NAMES:
        raise ValueError(
            "Left Sharpa joint order differs from LEFT_SHARPA_JOINT_NAMES: "
            f"{actual_left_hand_joint_names}"
        )
    if actual_right_hand_joint_names != RIGHT_SHARPA_JOINT_NAMES:
        raise ValueError(
            "Right Sharpa joint order differs from RIGHT_SHARPA_JOINT_NAMES: "
            f"{actual_right_hand_joint_names}"
        )

    articulation = replace(
        base_cfg.articulation,
        actuators=base_cfg.articulation.actuators
        + (
            XmlActuatorCfg(target_names_expr=LEFT_SHARPA_JOINT_NAMES),
            XmlActuatorCfg(target_names_expr=RIGHT_SHARPA_JOINT_NAMES),
        ),
    )
    cfg = replace(
        base_cfg,
        spec_fn=lambda: make_g1_sharpa_spec(mode, configure_physics=False),
        articulation=articulation,
        joint_names_simulation=list(G1_SHARPA_JOINT_NAMES),
        body_names_simulation=list(base_cfg.body_names_simulation or ())
        + hand_body_names,
    )
    return AssetSpec(config=cfg, sensors=base_asset.sensors)


_registry = Registry.instance()
for _mode in _SUPPORTED_MODES:
    _registry.register(
        "asset",
        f"g1-sharpa-mode_{_mode}",
        lambda backend, mode=_mode: make_g1_sharpa_asset(backend, mode=mode),
    )

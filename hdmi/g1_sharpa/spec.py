"""Compose the stock G1 MJCF with a full Sharpa Hand on each wrist."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import mujoco
import numpy as np

from mimic_lite.assets.g1 import G1_MJCF_REF_BY_MODE
from mjhub import resolve_asset_reference

Side = Literal["left", "right"]

_XML_DIR = Path(__file__).resolve().parent / "xmls"
_HAND_XML_BY_SIDE = {
    "left": _XML_DIR / "left_hand.xml",
    "right": _XML_DIR / "right_hand.xml",
}

LEFT_HAND_PREFIX = "lh/"
RIGHT_HAND_PREFIX = "rh/"
HAND_JOINT_COUNT = 22

# MuJoCo options required by the Menagerie Sharpa model.  MjSpec.attach() keeps
# the parent's global options, so set these on the composed G1 before attaching
# either hand.  Besides preserving the hand model's intended contact behavior,
# this avoids attach-time child/parent option conflict warnings.
SHARPA_INTEGRATOR = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
SHARPA_IMPRATIO = 10.0
SHARPA_CONE = mujoco.mjtCone.mjCONE_ELLIPTIC

# The upstream G1 MJCF contains commented-out hand mount bodies at this pose,
# relative to each wrist-yaw link. Keep these values explicit so later mesh
# alignment work can adjust them without changing the composition logic.
# The stock wrist mesh ends at x ~= 0.041 m. A 0.5 mm overlap prevents a
# rendering seam without burying the Sharpa wrist inside the G1 wrist.
LEFT_HAND_ATTACH_POS = (0.0405, 0.0, 0.0)
RIGHT_HAND_ATTACH_POS = (0.0405, 0.0, 0.0)
# Sharpa fingers extend along local +Z and its palm normal is local +X.
# Rotate +Z into the G1 wrist's forward +X direction, then mirror the palm
# normals so each palm faces toward the robot's centerline.
LEFT_HAND_ATTACH_QUAT = (-0.5, 0.5, -0.5, 0.5)
RIGHT_HAND_ATTACH_QUAT = (0.5, 0.5, 0.5, 0.5)


def _load_hand_spec(side: Side) -> mujoco.MjSpec:
    """Load one complete 22-DoF Sharpa Hand specification."""
    xml_path = _HAND_XML_BY_SIDE[side]
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

    spec = mujoco.MjSpec.from_file(str(xml_path))
    if len(spec.joints) != HAND_JOINT_COUNT:
        raise ValueError(
            f"Expected {HAND_JOINT_COUNT} {side} hand joints, got {len(spec.joints)}"
        )
    if len(spec.actuators) != HAND_JOINT_COUNT:
        raise ValueError(
            f"Expected {HAND_JOINT_COUNT} {side} hand actuators, "
            f"got {len(spec.actuators)}"
        )
    return spec


def _remove_stock_rubber_hands(spec: mujoco.MjSpec) -> None:
    """Remove the stock rubber-hand visuals and their capsule collisions."""
    rubber_meshes = {"left_rubber_hand", "right_rubber_hand"}
    collision_geoms = {"left_hand_collision", "right_hand_collision"}
    geoms = [
        geom
        for geom in spec.geoms
        if geom.meshname in rubber_meshes or geom.name in collision_geoms
    ]
    for geom in geoms:
        spec.delete(geom)


def _configure_sharpa_physics(spec: mujoco.MjSpec) -> None:
    """Apply the Menagerie solver settings to the combined model."""
    spec.option.integrator = SHARPA_INTEGRATOR
    spec.option.impratio = SHARPA_IMPRATIO
    spec.option.cone = SHARPA_CONE


def attach_sharpa_hand(
    g1_spec: mujoco.MjSpec,
    side: Side,
    *,
    pos: Sequence[float] | None = None,
    quat: Sequence[float] | None = None,
) -> None:
    """Attach one full-DoF Sharpa Hand to the matching G1 wrist-yaw link."""
    prefix = LEFT_HAND_PREFIX if side == "left" else RIGHT_HAND_PREFIX
    wrist_name = f"{side}_wrist_yaw_link"
    wrist = g1_spec.body(wrist_name)
    if wrist is None:
        raise ValueError(f"G1 spec has no body named {wrist_name!r}")

    if pos is None:
        pos = LEFT_HAND_ATTACH_POS if side == "left" else RIGHT_HAND_ATTACH_POS
    if quat is None:
        quat = (
            LEFT_HAND_ATTACH_QUAT if side == "left" else RIGHT_HAND_ATTACH_QUAT
        )
    frame = wrist.add_frame()
    frame.name = f"{side}_sharpa_mount"
    frame.pos = np.asarray(pos, dtype=float)
    frame.quat = np.asarray(quat, dtype=float)
    hand_spec = _load_hand_spec(side)
    # Global options belong to the final scene, not an attached entity. Match
    # the current parent here to avoid attach conflicts; the standalone path
    # already configures the parent for Sharpa, while MJLab uses MujocoCfg.
    hand_spec.option.integrator = g1_spec.option.integrator
    hand_spec.option.impratio = g1_spec.option.impratio
    hand_spec.option.cone = g1_spec.option.cone
    g1_spec.attach(child=hand_spec, prefix=prefix, frame=frame)

    # Exclude only the rigid wrist-to-palm mounting interface.  In particular,
    # do not suppress contacts between the G1 wrist and movable finger links.
    exclude = g1_spec.add_exclude()
    exclude.bodyname1 = wrist_name
    exclude.bodyname2 = f"{prefix}{side}_hand_C_MC"


def make_g1_sharpa_spec(
    mode: int = 15, *, configure_physics: bool = True
) -> mujoco.MjSpec:
    """Return a fresh G1 spec with full left and right Sharpa Hands attached.

    ``configure_physics=False`` is used when MJLab will attach the entity to a
    scene and apply equivalent global options through ``MujocoCfg``.
    """
    if mode not in G1_MJCF_REF_BY_MODE:
        supported = ", ".join(str(value) for value in G1_MJCF_REF_BY_MODE)
        raise ValueError(f"Unsupported G1 mode {mode}; expected one of: {supported}")

    g1_path = resolve_asset_reference(G1_MJCF_REF_BY_MODE[mode])
    spec = mujoco.MjSpec.from_file(str(g1_path))
    _remove_stock_rubber_hands(spec)
    if configure_physics:
        _configure_sharpa_physics(spec)
    attach_sharpa_hand(spec, "left")
    attach_sharpa_hand(spec, "right")
    return spec

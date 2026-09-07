"""Regression tests for the G1 + dual Sharpa Hand MJLab asset."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf

from hdmi.g1_sharpa.asset import (
    ACTION_GROUPS,
    BODY_ACTION_SLICE,
    G1_JOINT_NAMES,
    G1_SHARPA_JOINT_NAMES,
    LEFT_HAND_ACTION_SLICE,
    LEFT_SHARPA_JOINT_NAMES,
    RIGHT_HAND_ACTION_SLICE,
    RIGHT_SHARPA_JOINT_NAMES,
)

from hdmi.g1_sharpa.spec import (
    HAND_JOINT_COUNT,
    LEFT_HAND_ATTACH_POS,
    LEFT_HAND_ATTACH_QUAT,
    LEFT_HAND_PREFIX,
    RIGHT_HAND_ATTACH_POS,
    RIGHT_HAND_ATTACH_QUAT,
    RIGHT_HAND_PREFIX,
    SHARPA_CONE,
    SHARPA_IMPRATIO,
    SHARPA_INTEGRATOR,
    make_g1_sharpa_spec,
)
from mimic_lite.assets.g1 import G1_MJCF_REF_BY_MODE
from mjhub import resolve_asset_reference

_XML_DIR = Path(__file__).parents[1] / "hdmi" / "g1_sharpa" / "xmls"
_TASK_PATH = Path(__file__).parents[1] / "cfg" / "task" / "g1-sharpa-control.yaml"
_PHASE9_TASK_PATH = (
    Path(__file__).parents[1]
    / "cfg"
    / "task"
    / "omomo-suitcase-object-pose-sharpa.yaml"
)


def _model_names(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int
) -> list[str]:
    names = [mujoco.mj_id2name(model, object_type, index) for index in range(count)]
    assert all(name is not None for name in names)
    return names  # type: ignore[return-value]


def _assert_unique_model_names(model: mujoco.MjModel) -> None:
    collections = (
        ("body", mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        ("joint", mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        ("actuator", mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
        ("mesh", mujoco.mjtObj.mjOBJ_MESH, model.nmesh),
        ("material", mujoco.mjtObj.mjOBJ_MATERIAL, model.nmat),
    )
    for collection, object_type, count in collections:
        names = _model_names(model, object_type, count)
        duplicates = [name for name, occurrences in Counter(names).items() if occurrences > 1]
        assert not duplicates, f"Duplicate {collection} names: {duplicates}"


def test_standalone_hand_specs_compile() -> None:
    for side in ("left", "right"):
        spec = mujoco.MjSpec.from_file(str(_XML_DIR / f"{side}_hand.xml"))
        model = spec.compile()

        assert len(spec.joints) == HAND_JOINT_COUNT
        assert len(spec.actuators) == HAND_JOINT_COUNT
        assert model.njnt == HAND_JOINT_COUNT
        assert model.nu == HAND_JOINT_COUNT


def test_combined_spec_contains_both_full_dof_hands() -> None:
    spec = make_g1_sharpa_spec(mode=15)
    model = spec.compile()

    base_path = resolve_asset_reference(G1_MJCF_REF_BY_MODE[15])
    base_spec = mujoco.MjSpec.from_file(str(base_path))
    base_joint_names = {joint.name for joint in base_spec.joints}

    geom_names = {geom.name for geom in spec.geoms}
    mesh_names = {geom.meshname for geom in spec.geoms}

    left_joints = [joint.name for joint in spec.joints if joint.name.startswith("lh/")]
    right_joints = [
        joint.name for joint in spec.joints if joint.name.startswith("rh/")
    ]
    left_actuators = [
        actuator.name
        for actuator in spec.actuators
        if actuator.name.startswith("lh/")
    ]
    right_actuators = [
        actuator.name
        for actuator in spec.actuators
        if actuator.name.startswith("rh/")
    ]

    assert len(left_joints) == len(right_joints) == HAND_JOINT_COUNT
    assert len(left_actuators) == len(right_actuators) == HAND_JOINT_COUNT
    assert base_joint_names <= {joint.name for joint in spec.joints}
    assert spec.body("left_wrist_yaw_link") is not None
    assert spec.body("right_wrist_yaw_link") is not None
    assert spec.body("lh/left_hand_C_MC") is not None
    assert spec.body("rh/right_hand_C_MC") is not None
    assert "left_rubber_hand" not in mesh_names
    assert "right_rubber_hand" not in mesh_names
    assert "left_hand_collision" not in geom_names
    assert "right_hand_collision" not in geom_names
    assert model.nq == 80
    assert model.nv == 79
    assert model.nu == 44


def test_combined_spec_names_and_hand_actuator_mappings_are_valid() -> None:
    model = make_g1_sharpa_spec(mode=15).compile()
    _assert_unique_model_names(model)

    joint_names = _model_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    actuator_names = _model_names(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu
    )
    left_joints = {name for name in joint_names if name.startswith(LEFT_HAND_PREFIX)}
    right_joints = {name for name in joint_names if name.startswith(RIGHT_HAND_PREFIX)}
    left_actuators = {
        name for name in actuator_names if name.startswith(LEFT_HAND_PREFIX)
    }
    right_actuators = {
        name for name in actuator_names if name.startswith(RIGHT_HAND_PREFIX)
    }

    assert len(left_joints) == len(left_actuators) == HAND_JOINT_COUNT
    assert len(right_joints) == len(right_actuators) == HAND_JOINT_COUNT
    actuator_targets = {LEFT_HAND_PREFIX: set(), RIGHT_HAND_PREFIX: set()}
    for actuator_id, actuator_name in enumerate(actuator_names):
        assert model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        target_joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        assert target_joint_name is not None
        prefix = (
            LEFT_HAND_PREFIX
            if actuator_name.startswith(LEFT_HAND_PREFIX)
            else RIGHT_HAND_PREFIX
        )
        assert target_joint_name.startswith(prefix)
        actuator_targets[prefix].add(target_joint_name)

    assert actuator_targets[LEFT_HAND_PREFIX] == left_joints
    assert actuator_targets[RIGHT_HAND_PREFIX] == right_joints


def test_combined_spec_initial_state_and_ranges_are_valid() -> None:
    model = make_g1_sharpa_spec(mode=15).compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    assert np.isfinite(data.qpos).all()

    joint_limited = np.asarray(model.jnt_limited, dtype=bool)
    joint_ranges = model.jnt_range[joint_limited]
    assert joint_limited.sum() == 73
    assert np.isfinite(joint_ranges).all()
    assert np.all(joint_ranges[:, 0] <= joint_ranges[:, 1])

    control_limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    control_ranges = model.actuator_ctrlrange[control_limited]
    assert control_limited.sum() == 2 * HAND_JOINT_COUNT
    assert np.isfinite(control_ranges).all()
    assert np.all(control_ranges[:, 0] <= control_ranges[:, 1])


def test_g1_sharpa_mounting_transforms_match_calibrated_baseline() -> None:
    spec = make_g1_sharpa_spec(mode=15)

    left_mount = spec.frame("left_sharpa_mount")
    right_mount = spec.frame("right_sharpa_mount")
    assert left_mount is not None
    assert right_mount is not None
    np.testing.assert_allclose(left_mount.pos, LEFT_HAND_ATTACH_POS)
    np.testing.assert_allclose(left_mount.quat, LEFT_HAND_ATTACH_QUAT)
    np.testing.assert_allclose(right_mount.pos, RIGHT_HAND_ATTACH_POS)
    np.testing.assert_allclose(right_mount.quat, RIGHT_HAND_ATTACH_QUAT)


def test_combined_spec_uses_sharpa_physics_and_scoped_mount_exclusions() -> None:
    spec = make_g1_sharpa_spec(mode=15)

    assert spec.option.integrator == SHARPA_INTEGRATOR
    assert spec.option.impratio == SHARPA_IMPRATIO
    assert spec.option.cone == SHARPA_CONE

    mount_pairs = {
        frozenset((exclude.bodyname1, exclude.bodyname2))
        for exclude in spec.excludes
        if exclude.bodyname1 in {"left_wrist_yaw_link", "right_wrist_yaw_link"}
    }
    assert mount_pairs == {
        frozenset(("left_wrist_yaw_link", "lh/left_hand_C_MC")),
        frozenset(("right_wrist_yaw_link", "rh/right_hand_C_MC")),
    }


def test_phase7_action_groups_have_fixed_29_22_22_layout() -> None:
    assert tuple(ACTION_GROUPS) == (
        "body_action",
        "left_hand_action",
        "right_hand_action",
    )


def test_phase8_standalone_task_is_self_contained() -> None:
    cfg = OmegaConf.load(_TASK_PATH)

    assert cfg.robot.name == "g1-sharpa-mode_15"
    assert cfg.num_envs == 1
    assert cfg.input.action._target_ == "hdmi.G1SharpaJointPositionAction"
    assert cfg.command._target_ == "hdmi.StaticPoseCommand"
    assert cfg.sim.mujoco_integrator == "implicitfast"
    assert cfg.sim.mujoco_impratio == 10.0
    assert cfg.sim.mujoco_cone == "elliptic"
    assert "objects" not in cfg
    assert "motion" not in cfg
    assert "suitcase" not in str(OmegaConf.to_container(cfg)).lower()
    assert tuple(map(len, ACTION_GROUPS.values())) == (29, 22, 22)
    assert G1_SHARPA_JOINT_NAMES[BODY_ACTION_SLICE] == G1_JOINT_NAMES
    assert G1_SHARPA_JOINT_NAMES[LEFT_HAND_ACTION_SLICE] == LEFT_SHARPA_JOINT_NAMES
    assert G1_SHARPA_JOINT_NAMES[RIGHT_HAND_ACTION_SLICE] == RIGHT_SHARPA_JOINT_NAMES
    assert (
        G1_JOINT_NAMES + LEFT_SHARPA_JOINT_NAMES + RIGHT_SHARPA_JOINT_NAMES
        == G1_SHARPA_JOINT_NAMES
    )


def test_phase9_task_keeps_policy_body_only_and_hands_fixed() -> None:
    cfg = OmegaConf.load(_PHASE9_TASK_PATH)

    assert cfg.robot.name == "g1-sharpa-mode_15"
    assert cfg.command._target_ == "hdmi.RobotObjectTrackingFixedSharpa"
    assert cfg.command.hand_input_key == "hand_control"
    assert set(cfg.input) == {"hand_control"}
    assert cfg.input.hand_control._target_ == "JointPosition"
    assert set(cfg.input.hand_control.action_scaling) == {
        "left_(thumb|index|middle|ring|pinky)_.*",
        "right_(thumb|index|middle|ring|pinky)_.*",
    }
    assert all(
        value == 0.0 for value in cfg.input.hand_control.action_scaling.values()
    )
    assert cfg.sim.contact_sensor_maxmatch == 128


def test_registered_asset_builds_entity_and_accepts_controls() -> None:
    """Run backend initialization in a subprocess to avoid global test pollution."""
    script = r"""
import numpy as np
import mujoco

import active_adaptation as aa

aa.set_backend("mjlab")

import hdmi
from hdmi.g1_sharpa import (
    G1_SHARPA_JOINT_NAMES,
    LEFT_SHARPA_JOINT_NAMES,
    RIGHT_SHARPA_JOINT_NAMES,
)
from active_adaptation.registry import Registry

factory = Registry.instance().get("asset", "g1-sharpa-mode_15")
asset = factory(backend="mjlab")
entity = asset.config.build()
model = entity.spec.compile()

assert len(asset.config.joint_names_simulation) == 73
assert tuple(asset.config.joint_names_simulation) == G1_SHARPA_JOINT_NAMES
assert len(asset.config.body_names_simulation) == 78
assert len(entity.joint_names) == 73
assert len(entity.actuators) == 9
assert [len(group.target_names) for group in entity.actuators[-2:]] == [22, 22]
assert tuple(entity.actuators[-2].target_names) == LEFT_SHARPA_JOINT_NAMES
assert tuple(entity.actuators[-1].target_names) == RIGHT_SHARPA_JOINT_NAMES
assert len(entity.joint_names) - sum(len(group.target_names) for group in entity.actuators[-2:]) == 29
assert model.nu == 73

data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)
assert np.isfinite(data.qpos).all()
assert np.isfinite(data.qvel).all()
assert np.isfinite(data.qacc).all()
"""
    subprocess.run([sys.executable, "-c", script], check=True)

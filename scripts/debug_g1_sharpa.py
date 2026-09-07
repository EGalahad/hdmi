"""Standalone dynamics and viewer check for the 73-DoF G1 + Sharpa asset."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np

import active_adaptation as aa


@dataclass
class Diagnostics:
    max_actuator_force: float = 0.0
    max_left_wrist_contact_force: float = 0.0
    max_right_wrist_contact_force: float = 0.0
    max_joint_limit_violation: float = 0.0
    max_joint_limit_violation_name: str = ""
    max_hand_penetration: float = 0.0
    max_hand_penetration_pair: tuple[str, str] | None = None
    max_passive_mcp_aa_displacement: float = 0.0
    max_passive_mcp_aa_displacement_name: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "sequence",
            "joint-sweep",
            "hold",
            "left",
            "right",
            "both",
            "body",
        ),
        default="sequence",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--amplitude", type=float, default=0.08)
    parser.add_argument("--frequency", type=float, default=0.25)
    parser.add_argument(
        "--sweep-angle",
        type=float,
        default=0.12,
        help="Peak displacement in radians for each joint in joint-sweep mode.",
    )
    parser.add_argument(
        "--drive-mcp-aa",
        action="store_true",
        help="Actuate MCP-AA joints; by default their actuators are force-disabled.",
    )
    parser.add_argument("--max-hand-penetration", type=float, default=0.002)
    parser.add_argument("--viewer", choices=("none", "viser"), default="none")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--free-base",
        action="store_true",
        help="Do not pin the floating base (the open-loop G1 will fall).",
    )
    parser.add_argument("--report-interval", type=float, default=1.0)
    return parser.parse_args()


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    name = mujoco.mj_id2name(model, object_type, index)
    if name is None:
        raise ValueError(f"Unnamed {object_type} at index {index}")
    return name


def _body_subtree(model: mujoco.MjModel, root_name: str) -> set[int]:
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root_id < 0:
        raise ValueError(f"Missing body {root_name!r}")
    result = {root_id}
    for body_id in range(root_id + 1, model.nbody):
        parent_id = int(model.body_parentid[body_id])
        if parent_id in result:
            result.add(body_id)
    return result


def _contact_force_for_subtree(
    model: mujoco.MjModel, data: mujoco.MjData, body_ids: set[int]
) -> float:
    maximum = 0.0
    force = np.zeros(6)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if body1 not in body_ids and body2 not in body_ids:
            continue
        mujoco.mj_contactForce(model, data, contact_id, force)
        maximum = max(maximum, float(np.linalg.norm(force[:3])))
    return maximum


def _hand_penetration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_body_ids: set[int],
    right_body_ids: set[int],
) -> tuple[float, tuple[str, str] | None]:
    maximum = 0.0
    pair = None
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        same_hand = (
            body1 in left_body_ids and body2 in left_body_ids
        ) or (
            body1 in right_body_ids and body2 in right_body_ids
        )
        penetration = max(0.0, -float(contact.dist))
        if not same_hand or penetration <= maximum:
            continue
        maximum = penetration
        pair = (
            _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
            _name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
        )
    return maximum, pair


def _safe_target(
    default: float,
    lower: float,
    upper: float,
    phase: float,
    amplitude: float,
    one_sided: bool,
) -> float:
    if one_sided:
        destination = upper if upper - default >= default - lower else lower
        return float(default + phase * amplitude * (destination - default))
    return float(default + phase * amplitude * 0.5 * (upper - lower))


def _stage(mode: str, elapsed: float, duration: float) -> str:
    if mode != "sequence":
        return mode
    stages = ("hold", "left", "right", "both", "body")
    index = min(int(elapsed / duration * len(stages)), len(stages) - 1)
    return stages[index]


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.amplitude <= 0.25:
        raise ValueError("--amplitude must be in (0, 0.25]")
    if not 0.0 < args.sweep_angle <= 0.3:
        raise ValueError("--sweep-angle must be in (0, 0.3]")
    if args.duration <= 0.0 or args.frequency <= 0.0:
        raise ValueError("--duration and --frequency must be positive")

    aa.set_backend("mjlab")
    import hdmi  # noqa: F401  # register HDMI assets
    from active_adaptation.registry import Registry
    from hdmi.g1_sharpa import (
        BODY_ACTION_SLICE,
        G1_SHARPA_JOINT_NAMES,
        LEFT_HAND_ACTION_SLICE,
        RIGHT_HAND_ACTION_SLICE,
    )

    asset = Registry.instance().get("asset", "g1-sharpa-mode_15")(backend="mjlab")
    entity = asset.config.build()
    spec = entity.spec
    floor = spec.worldbody.add_geom()
    floor.name = "debug_ground"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = np.array((2.0, 2.0, 0.1))
    floor.rgba = np.array((0.65, 0.65, 0.65, 1.0))

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    joint_order = list(asset.config.joint_names_simulation)
    if tuple(joint_order) != G1_SHARPA_JOINT_NAMES:
        raise ValueError("Asset joint order differs from the Phase 7 action contract")
    body_joints = joint_order[BODY_ACTION_SLICE]
    left_joints = joint_order[LEFT_HAND_ACTION_SLICE]
    right_joints = joint_order[RIGHT_HAND_ACTION_SLICE]
    if (len(body_joints), len(left_joints), len(right_joints)) != (29, 22, 22):
        raise ValueError("Expected action groups 29 + 22 + 22")

    actuator_for_joint: dict[str, int] = {}
    model_joint_name: dict[str, str] = {}
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        namespaced_name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        canonical_name = namespaced_name.split("/")[-1]
        actuator_for_joint[canonical_name] = actuator_id
        model_joint_name[canonical_name] = namespaced_name
    missing_actuators = [name for name in joint_order if name not in actuator_for_joint]
    if missing_actuators:
        raise ValueError(f"Joints without actuators: {missing_actuators}")

    mcp_aa_joints = [name for name in joint_order if "_MCP_AA" in name]
    if not args.drive_mcp_aa:
        for joint_name in mcp_aa_joints:
            actuator_id = actuator_for_joint[joint_name]
            model.actuator_gainprm[actuator_id] = 0.0
            model.actuator_biasprm[actuator_id] = 0.0

    default_qpos = data.qpos.copy()
    default_ctrl = np.zeros(model.nu)
    joint_ids: dict[str, int] = {}
    for joint_name in joint_order:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, model_joint_name[joint_name]
        )
        joint_ids[joint_name] = joint_id
        default_ctrl[actuator_for_joint[joint_name]] = default_qpos[
            model.jnt_qposadr[joint_id]
        ]
    data.ctrl[:] = default_ctrl

    body_motion_joints = [
        name
        for name in body_joints
        if any(token in name for token in ("waist", "elbow", "shoulder_pitch"))
    ][:6]
    groups = {
        "hold": [],
        "left": [name for name in left_joints if args.drive_mcp_aa or name not in mcp_aa_joints],
        "right": [name for name in right_joints if args.drive_mcp_aa or name not in mcp_aa_joints],
        "both": [
            name
            for name in left_joints + right_joints
            if args.drive_mcp_aa or name not in mcp_aa_joints
        ],
        "body": body_motion_joints,
    }
    sweep_joints = [
        name
        for name in left_joints + right_joints
        if args.drive_mcp_aa or name not in mcp_aa_joints
    ]

    left_bodies = _body_subtree(model, "left_wrist_yaw_link")
    right_bodies = _body_subtree(model, "right_wrist_yaw_link")
    limited_joint_ids = np.flatnonzero(model.jnt_limited)

    viewer_scene = None
    if args.viewer == "viser":
        import viser
        from mjlab.viewer.viser import ViserMujocoScene

        server = viser.ViserServer(
            host=args.host, port=args.port, label="G1 + Sharpa dynamics check"
        )
        viewer_scene = ViserMujocoScene(server, model, num_envs=1)
        with server.gui.add_folder("Display"):
            viewer_scene.create_scene_gui(camera_distance=2.5)
        with server.gui.add_folder("Geometry groups"):
            viewer_scene.create_groups_gui()
        viewer_scene.update_from_mjdata(data)
        print(f"Viewer: http://{args.host}:{args.port}")

    print(f"Joint groups: body={len(body_joints)}, left={len(left_joints)}, right={len(right_joints)}")
    print(f"Action dimension: {model.nu}")
    print(
        "MCP-AA actuators: "
        f"{'driven' if args.drive_mcp_aa else 'present but force-disabled'} "
        f"({len(mcp_aa_joints)} joints)"
    )
    print("Resolved joint order:")
    for index, joint_name in enumerate(joint_order):
        print(f"  {index:02d}: {joint_name}")

    diagnostics = Diagnostics()
    start = time.monotonic()
    last_report = start
    previous_stage = ""
    previous_sweep_joint = ""
    timestep = float(model.opt.timestep)
    steps = max(1, math.ceil(args.duration / timestep))
    for step in range(steps):
        elapsed = step * timestep
        stage = _stage(args.mode, elapsed, args.duration)
        if stage != previous_stage:
            print(f"Stage: {stage}")
            previous_stage = stage

        data.ctrl[:] = default_ctrl
        angle = 2.0 * math.pi * args.frequency * elapsed
        phase = math.sin(angle)
        one_sided_phase = 0.5 - 0.5 * math.cos(angle)
        active_joints = groups.get(stage, [])
        if stage == "joint-sweep":
            sweep_position = min(elapsed / args.duration, 1.0 - np.finfo(float).eps)
            scaled_position = sweep_position * len(sweep_joints)
            sweep_index = min(int(scaled_position), len(sweep_joints) - 1)
            sweep_progress = scaled_position - sweep_index
            sweep_joint = sweep_joints[sweep_index]
            active_joints = [sweep_joint]
            one_sided_phase = 0.5 - 0.5 * math.cos(2.0 * math.pi * sweep_progress)
            if sweep_joint != previous_sweep_joint:
                print(
                    f"Joint sweep {sweep_index + 1:02d}/{len(sweep_joints)}: "
                    f"{sweep_joint}"
                )
                previous_sweep_joint = sweep_joint

        for joint_name in active_joints:
            joint_id = joint_ids[joint_name]
            actuator_id = actuator_for_joint[joint_name]
            lower, upper = model.jnt_range[joint_id]
            if stage == "joint-sweep":
                default = float(default_ctrl[actuator_id])
                destination = upper if upper - default >= default - lower else lower
                displacement = min(args.sweep_angle, abs(float(destination - default)))
                direction = 1.0 if destination >= default else -1.0
                target = default + direction * one_sided_phase * displacement
            else:
                target = _safe_target(
                    default=float(default_ctrl[actuator_id]),
                    lower=float(lower),
                    upper=float(upper),
                    phase=(
                        one_sided_phase
                        if joint_name in left_joints or joint_name in right_joints
                        else phase
                    ),
                    amplitude=args.amplitude,
                    one_sided=joint_name in left_joints or joint_name in right_joints,
                )
            data.ctrl[actuator_id] = np.clip(target, lower, upper)

        if not args.free_base:
            data.qpos[:7] = default_qpos[:7]
            data.qvel[:6] = 0.0
        mujoco.mj_step(model, data)
        if not args.free_base:
            data.qpos[:7] = default_qpos[:7]
            data.qvel[:6] = 0.0
            mujoco.mj_forward(model, data)
        arrays = (data.qpos, data.qvel, data.qacc, data.ctrl, data.actuator_force)
        if not all(np.isfinite(array).all() for array in arrays):
            raise FloatingPointError(f"NaN/Inf detected at step {step}, stage {stage}")

        violation = 0.0
        for joint_id in limited_joint_ids:
            qpos = data.qpos[model.jnt_qposadr[joint_id]]
            lower, upper = model.jnt_range[joint_id]
            joint_violation = max(float(lower - qpos), float(qpos - upper), 0.0)
            if joint_violation > violation:
                violation = joint_violation
                violation_name = _name(
                    model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)
                )
        if violation > diagnostics.max_joint_limit_violation:
            diagnostics.max_joint_limit_violation = violation
            diagnostics.max_joint_limit_violation_name = violation_name
        diagnostics.max_actuator_force = max(
            diagnostics.max_actuator_force,
            float(np.max(np.abs(data.actuator_force), initial=0.0)),
        )
        diagnostics.max_left_wrist_contact_force = max(
            diagnostics.max_left_wrist_contact_force,
            _contact_force_for_subtree(model, data, left_bodies),
        )
        diagnostics.max_right_wrist_contact_force = max(
            diagnostics.max_right_wrist_contact_force,
            _contact_force_for_subtree(model, data, right_bodies),
        )
        if not args.drive_mcp_aa:
            for joint_name in mcp_aa_joints:
                joint_id = joint_ids[joint_name]
                qpos_adr = model.jnt_qposadr[joint_id]
                displacement = abs(data.qpos[qpos_adr] - default_qpos[qpos_adr])
                if displacement > diagnostics.max_passive_mcp_aa_displacement:
                    diagnostics.max_passive_mcp_aa_displacement = float(displacement)
                    diagnostics.max_passive_mcp_aa_displacement_name = joint_name
        penetration, pair = _hand_penetration(
            model, data, left_bodies, right_bodies
        )
        if penetration > diagnostics.max_hand_penetration:
            diagnostics.max_hand_penetration = penetration
            diagnostics.max_hand_penetration_pair = pair

        if viewer_scene is not None:
            viewer_scene.update_from_mjdata(data)
        now = time.monotonic()
        if now - last_report >= args.report_interval:
            print(
                f"t={elapsed:.2f}s stage={stage} contacts={data.ncon} "
                f"max|qvel|={np.max(np.abs(data.qvel)):.3f}"
            )
            last_report = now
        if args.realtime or viewer_scene is not None:
            deadline = start + (step + 1) * timestep
            time.sleep(max(0.0, deadline - time.monotonic()))

    print("Summary:")
    print(f"  max actuator force: {diagnostics.max_actuator_force:.6g} N/Nm")
    print(
        "  max wrist contact force: "
        f"left={diagnostics.max_left_wrist_contact_force:.6g} N, "
        f"right={diagnostics.max_right_wrist_contact_force:.6g} N"
    )
    print(
        "  max joint-limit violation: "
        f"{diagnostics.max_joint_limit_violation:.6g} rad "
        f"({diagnostics.max_joint_limit_violation_name or 'none'})"
    )
    print(
        "  max hand self-penetration: "
        f"{diagnostics.max_hand_penetration:.6g} m "
        f"({diagnostics.max_hand_penetration_pair or 'none'})"
    )
    if not args.drive_mcp_aa:
        print(
            "  max passive MCP-AA displacement: "
            f"{diagnostics.max_passive_mcp_aa_displacement:.6g} rad "
            f"({diagnostics.max_passive_mcp_aa_displacement_name or 'none'})"
        )
    if diagnostics.max_joint_limit_violation > 1e-2:
        raise RuntimeError("Joint-limit violation exceeded 0.01 rad")
    if diagnostics.max_hand_penetration > args.max_hand_penetration:
        raise RuntimeError(
            "Hand self-penetration exceeded "
            f"{args.max_hand_penetration:.6g} m"
        )
    print("PASS: dynamics remained finite and within joint-limit tolerance")


if __name__ == "__main__":
    main()

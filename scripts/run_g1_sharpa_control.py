"""Run scripted controls through the standalone G1 + Sharpa MJLab task."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import active_adaptation as aa


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("sequence", "hold", "body", "left", "right", "both"),
        default="sequence",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    return parser.parse_args()


def _stage(mode: str, elapsed: float, duration: float) -> str:
    if mode != "sequence":
        return mode
    stages = ("hold", "left", "right", "both", "body")
    return stages[min(int(elapsed / duration * len(stages)), len(stages) - 1)]


def main() -> None:
    args = _parse_args()
    if args.duration <= 0.0 or args.num_envs <= 0:
        raise ValueError("--duration and --num-envs must be positive")

    aa.set_backend("mjlab")
    import active_adaptation.assets  # noqa: F401
    import active_adaptation.envs.mdp  # noqa: F401
    import active_adaptation.envs.sensors  # noqa: F401
    import hdmi  # noqa: F401
    from active_adaptation.envs.backends.mjlab import MjlabBackendEnv
    from hdmi.g1_sharpa import (
        BODY_ACTION_SLICE,
        G1_SHARPA_JOINT_NAMES,
        LEFT_HAND_ACTION_SLICE,
        RIGHT_HAND_ACTION_SLICE,
    )

    task_path = Path(__file__).parents[1] / "cfg" / "task" / "g1-sharpa-control.yaml"
    cfg = OmegaConf.load(task_path)
    cfg.num_envs = args.num_envs
    env = MjlabBackendEnv(cfg, device=args.device, headless=not args.viewer)
    env.reset()

    action_manager = env.input_managers["action"]
    if tuple(action_manager.names) != G1_SHARPA_JOINT_NAMES:
        raise RuntimeError("Runtime action order differs from the Phase 7 contract")
    if action_manager.action_dims != [29, 22, 22]:
        raise RuntimeError(f"Unexpected action groups: {action_manager.action_dims}")

    mcp_aa_ids = torch.tensor(
        [
            index
            for index, name in enumerate(G1_SHARPA_JOINT_NAMES)
            if "_MCP_AA" in name
        ],
        device=args.device,
    )
    step_dt = float(cfg.sim.step_dt)
    steps = max(1, math.ceil(args.duration / step_dt))
    start = time.monotonic()
    previous_stage = ""
    max_abs_joint_pos = 0.0

    for step in range(steps):
        elapsed = step * step_dt
        stage = _stage(args.mode, elapsed, args.duration)
        if stage != previous_stage:
            print(f"Stage: {stage}")
            previous_stage = stage

        action = torch.zeros(args.num_envs, 73, device=args.device)
        phase = 0.5 - 0.5 * math.cos(2.0 * math.pi * 0.25 * elapsed)
        if stage in ("left", "both"):
            action[:, LEFT_HAND_ACTION_SLICE] = phase
        if stage in ("right", "both"):
            action[:, RIGHT_HAND_ACTION_SLICE] = -phase
        if stage == "body":
            action[:, BODY_ACTION_SLICE] = 0.15 * math.sin(
                2.0 * math.pi * 0.25 * elapsed
            )
        action[:, mcp_aa_ids] = 0.0

        env.step(
            TensorDict(
                {"action": action},
                batch_size=[args.num_envs],
                device=args.device,
            )
        )
        joint_pos = env.scene.articulations["robot"].data.joint_pos
        if not torch.isfinite(joint_pos).all():
            raise FloatingPointError(f"Non-finite joint state at step {step}")
        max_abs_joint_pos = max(max_abs_joint_pos, float(joint_pos.abs().max()))

        if args.realtime or args.viewer:
            deadline = start + (step + 1) * step_dt
            time.sleep(max(0.0, deadline - time.monotonic()))

    print("Standalone task PASS")
    print(f"  environments: {args.num_envs}")
    print(f"  action groups: {action_manager.action_dims}")
    print(f"  action dimension: {action_manager.action_dim}")
    print(f"  max |joint position|: {max_abs_joint_pos:.6g} rad")
    env.close()


if __name__ == "__main__":
    main()

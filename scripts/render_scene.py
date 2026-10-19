"""Continuous checkpoint playback, with a render-only spatial layout and camera move."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import json, math, time
from pathlib import Path
import numpy as np
import torch
import mujoco
import imageio.v2 as imageio
from omegaconf import OmegaConf
from hydra.core.utils import setup_globals
from torchrl.envs.utils import ExplorationType, set_exploration_type
import active_adaptation as aa

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--config",
    type=Path,
    required=True,
    help="Resolved training config, e.g. cfg.yaml from the run",
)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", type=Path, required=True, help="New output directory")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seconds", type=float, default=20)
parser.add_argument("--center-objects", nargs="*", default=[])
parser.add_argument("--edge-objects", nargs="*", default=[])
parser.add_argument("--far-distance", type=float, default=42.03811978464831)
parser.add_argument("--pullback-start", type=float, default=3)
parser.add_argument("--pullback-end", type=float, default=15)
args = parser.parse_args()
if (
    args.num_envs < 1
    or args.seconds <= 0
    or args.far_distance <= 0
    or args.pullback_end <= args.pullback_start
):
    parser.error("Invalid environment count, duration, camera distance or timing")
OUT = args.output
OUT.mkdir(parents=True, exist_ok=False)
setup_globals()
cfg = OmegaConf.load(args.config)
OmegaConf.resolve(cfg)
OmegaConf.set_struct(cfg, False)
cfg.device = "cuda"
cfg.headless = True
cfg.app.headless = True
cfg.seed = 0
cfg.wandb.mode = "disabled"
cfg.task.num_envs = args.num_envs
cfg.task.randomization = {}
cfg.task.total_iters = cfg.total_iters
cfg.task.command.play_from_start = True
cfg.task.termination.root_pos_error.enabled = False
if "lost_expected_contact" in cfg.task.termination:
    cfg.task.termination.lost_expected_contact.enabled = False
aa.init(cfg, auto_rank=False)
from active_adaptation.helpers import make_env_policy
from active_adaptation.learning.modules.vecnorm import VecNorm
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

env, policy = make_env_policy(
    cfg.task,
    cfg.algo,
    seed=0,
    headless=True,
    device="cuda",
    checkpoint_path=args.checkpoint,
)
e = env.base_env
e.eval()
policy.eval()
carry = env.reset()
assert bool((e.command_manager.t == 0).all())
names = e.command_manager.object_names
assert len(names) == args.num_envs
preferred = list(dict.fromkeys(args.center_objects))
edge = list(dict.fromkeys(args.edge_objects))
if set(preferred) & set(edge):
    parser.error("center-objects and edge-objects must not overlap")
middle = sorted(set(names) - set(preferred) - set(edge))
# Interleave classes within each radial band; retain every initialized environment.
order = []
for group in [preferred, middle, edge]:
    pools = [[i for i, n in enumerate(names) if n == name] for name in group]
    for j in range(max(map(len, pools), default=0)):
        order.extend(pool[j] for pool in pools if j < len(pool))
assert len(order) == args.num_envs and len(set(order)) == args.num_envs
side = math.ceil(math.sqrt(args.num_envs))
slots = np.array(
    [
        [(i % side - (side - 1) / 2) * 4.0, (i // side - (side - 1) / 2) * 4.0, 0.0]
        for i in range(args.num_envs)
    ]
)
radial = sorted(
    range(args.num_envs),
    key=lambda i: (np.linalg.norm(slots[i, :2]), math.atan2(slots[i, 1], slots[i, 0])),
)
targets = np.zeros_like(slots)
for env_id, slot_id in zip(order, radial):
    targets[env_id] = slots[slot_id]
initial = e.sim.data.qpos.cpu().numpy().copy()
free = [
    int(e.sim.mj_model.jnt_qposadr[i])
    for i in range(e.sim.mj_model.njnt)
    if e.sim.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
]
assert len(free) == 2, free
# Center each robot/object pair in its display cell without changing the simulation.
paircenter = sum(initial[:, adr : adr + 3] for adr in free) / len(free)
shift = targets - paircenter
shift[:, 2] = 0


class SceneRenderer(OffscreenRenderer):
    def _sync_data_fields(self, data, env_idx):
        self._data.qpos[:] = self.qpos[env_idx]
        for addr in free:
            self._data.qpos[addr : addr + 3] += shift[env_idx]
        self._data.qvel[:] = self.qvel[env_idx]
        mujoco.mj_forward(self._model, self._data)

    def update_cached(self):
        self.qpos = e.sim.data.qpos.cpu().numpy()
        self.qvel = e.sim.data.qvel.cpu().numpy()
        super().update(e.sim.data)


viewer = ViewerConfig(
    width=1920,
    height=1088,
    max_extra_envs=args.num_envs - 1,
    lookat=(0, 0, 0.8),
    distance=4.8,
    elevation=-30,
    azimuth=90,
    fovy=45,
    enable_shadows=False,
    enable_reflections=False,
)
r = SceneRenderer(
    e.sim.mj_model,
    viewer,
    e.scene,
    sim_model=e.sim.model,
    expanded_fields=e.sim.expanded_fields,
)
r._model.vis.map.zfar = 1000
r._model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
if r._renderer is not None:
    r._renderer.close()
r._renderer = mujoco.Renderer(r._model, height=1088, width=1920, max_geom=200000)
r._env_ids = tuple(order)


def camera(t):
    u = np.clip(
        (t - args.pullback_start) / (args.pullback_end - args.pullback_start), 0, 1
    )
    s = u * u * (3 - 2 * u)
    # Log-distance interpolation keeps the retreat gentle near the subjects.
    final_distance = args.far_distance
    r._cam.distance = float(4.8 * (final_distance / 4.8) ** s)
    r._cam.elevation = float(-30 - 30 * s)
    r._cam.lookat[:] = [0.0, -2.0 * (1 - s), 0.8]
    return {
        "time": t,
        "distance": r._cam.distance,
        "elevation": r._cam.elevation,
        "lookat": r._cam.lookat.tolist(),
        "height_m": float(
            0.8 - r._cam.distance * math.sin(math.radians(r._cam.elevation))
        ),
    }


video = OUT
metadata = dict(
    checkpoint=args.checkpoint,
    num_envs=args.num_envs,
    seconds=args.seconds,
    fps=25,
    frames=round(args.seconds * 25),
    exploration="DETERMINISTIC",
    seed=0,
    training_motion_count=len(e.command_manager.dataset.motion_paths),
    source="all original training motion configurations",
    termination="base; root_pos_error and lost_expected_contact disabled",
    resets="all resets start at literal motion frame 0; rewind disabled; native automatic reset timing; not a success-rate evaluation",
    layout="one MuJoCo scene, render-only translation of complete robot/object pairs",
    spacing_m=4,
    center_classes=preferred,
    edge_classes=edge,
    environments=[
        dict(env_id=i, object=names[i], display_origin=targets[i].tolist())
        for i in range(args.num_envs)
    ],
    camera=[camera(t) for t in [0, 3, 9, 15, 20]],
)
(video / "manifest.json").write_text(json.dumps(metadata, indent=2))
OmegaConf.save(cfg, video / "resolved_config.yaml")
for t in [0, 9, 15]:
    camera(t)
    r.update_cached()
    imageio.imwrite(video / f"preview-{t:02}.png", r.render())
print("PREVIEW_READY", flush=True)
if os.environ.get("PREVIEW_ONLY") == "1":
    r.close()
    env.close()
    raise SystemExit(0)
rollout = policy.get_rollout_policy("eval")
steps = round(args.seconds * 25) * 2
assert math.isclose(env.step_dt, 0.02), "Recorder requires a 50 Hz policy"
done_counts = np.zeros(args.num_envs, dtype=int)
started = time.monotonic()
with (
    imageio.get_writer(
        video / "scene.mp4", fps=25, codec="libx264", quality=8, macro_block_size=16
    ) as writer,
    VecNorm.freeze(),
    torch.inference_mode(),
    set_exploration_type(ExplorationType.DETERMINISTIC),
):
    for i in range(steps):
        carry = rollout(carry)
        td, carry = env.step_and_maybe_reset(carry)
        done = td["next", "done"].squeeze(-1)
        done_counts += done.cpu().numpy().astype(int)
        assert bool((e.command_manager.t[done] == 0).all()), (
            "Reset did not start at frame 0"
        )
        if i % 2 == 0:
            camera(i * env.step_dt)
            r.update_cached()
            frame = r.render()
            writer.append_data(frame)
            if i in [0, 150, 450, 750, 998]:
                imageio.imwrite(video / f"frame-{i:04}.png", frame)
        if i % 50 == 0:
            print(
                "PROGRESS",
                i,
                steps,
                "elapsed",
                round(time.monotonic() - started, 1),
                flush=True,
            )
metadata["reset_audit"] = {"all_start_frame_zero": True}
metadata["resets_per_env"] = done_counts.tolist()
metadata["wall_seconds"] = time.monotonic() - started
(video / "manifest.json").write_text(json.dumps(metadata, indent=2))
r.close()
env.close()
(video / "COMPLETED").write_text(
    f"{args.seconds} seconds, {steps // 2} frames, 25fps\n"
)
print("COMPLETED", flush=True)

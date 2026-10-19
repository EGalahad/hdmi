# HDMI

HDMI extends [Mimic-Lite](https://github.com/EGalahad/mimic-lite) with paired
robot–object tracking in MjLab. Each environment has one robot and one rigid
`object` entity. Mesh, mass, inertia, motion and contact labels come from the
motion dataset; Python classes are not needed for each object category.

## Reproducible setup

Run these commands from a new checkout. The pinned code and environment below
reproduce the configuration used for the G1 + Inspire Weave experiment
`lclcuh3c` (8000 iterations), without applying the later performance experiments.
Full RL trajectories are not bitwise deterministic.

```bash
git clone -b dev/hdmi https://github.com/Agent-3154/active-adaptation.git
cd active-adaptation
git checkout b6a6b3f101344dfe8d5bdd6a06e921f4333127eb
git clone -b dev/hdmi https://github.com/EGalahad/mimic-lite projects/mimic-lite
git -C projects/mimic-lite checkout 61cc670
git clone https://github.com/EGalahad/hdmi projects/hdmi
mkdir -p venv/hdmi
cp projects/hdmi/pyproject-mjlab.toml venv/hdmi/pyproject.toml
cp projects/hdmi/uv-mjlab.lock venv/hdmi/uv.lock
uv sync --project venv/hdmi --locked
uv run --project venv/hdmi aa-discover-projects
uv run --project venv/hdmi aa-project enable mimic_lite
uv run --project venv/hdmi aa-project enable hdmi
```

The environment pins MjLab 1.6.0, MuJoCo 3.11.0, Warp 1.14.0, Torch 2.11.0
(CUDA 12.8), TorchRL 0.13.1 and TensorDict 0.13.0. It points `mujoco-warp` at
[the HDMI fork](https://github.com/EGalahad/mujoco_warp/tree/hdmi/3.11-repro),
commit `0e5b5b5657f2c68e0d6c20c2ff0a7d355112e532`. This fork adds absent-mesh
collision filtering and opt-in bounded CCD scratch reuse. Set
`HDMI_CCD_THREAD_SCRATCH=1` before importing it. AA needs no source change;
Mimic-Lite's `dev/hdmi` includes the expanded-count `clamp_min` fix.
The any4hdmi dependency is pinned to `d5dff76034a6d63fe3dfd11297b3e58ad1acc33b`, which builds
reference FK with kinematics/COM velocity only, avoiding unnecessary mesh
collision and constraint solves for reference poses.

## Data and defaults

The [converted Weave reference dataset](https://huggingface.co/datasets/elijahgalahad/any4hdmi-g1-inspire-weave-reference-1605)
contains 1605 motions across nine objects, with MJCF/meshes and contact labels.
`cfg/task/motion/g1/weave-reference.yaml` contains only pinned `hf://` references
to revision `c716c1e0e316b8688d3825ce9e60607b82ca90c6`.
The upstream split is named `reference/test`, but **these motions were used for
training**. Playback and the 576-motion progress evaluation are training-set
fit measurements, not held-out generalization or object-task success.

A new rigid object needs a converted any4hdmi directory (`manifest.json`, MJCF,
meshes and motions), plus a `motion_cfgs` entry with its path and sampling weight.
Store its real name in `manifest.json` as `object_name`. The general mesh-variant
loader reads that name; `env.command_manager.object_names` returns the names
for the current environments. Existing datasets without the attribute fall back
to the motion configuration key. Contact tasks additionally need compatible
contact labels and robot contact link names. Category-dependent policy inputs
change dimension when categories change: retrain such policies rather than
loading an old nine-category checkpoint unchanged.

HDMI defaults now use:

- entropy 0.008 → 0.001, linearly between 60% and 100% of the run;
- weighted advantage sum followed by one normalization;
- global object position/orientation exponential rewards (sigma 0.5);
- action-rate penalty weight 0.1;
- constraint capacity `njmax=1024`, contact capacity `nconmax=256`, matching
  the experiment (both native and MjLab configuration fields).

The Weave preset also supplies robot-specific Inspire coupling: follower
finger targets are affine functions of driver targets, e.g. thumb intermediate
and distal targets use 1.6× and 2.4× proximal pitch. This is not object-specific.
The object's temporary `initial_pos` parks it during scene construction; every
motion reset already sets the real position/orientation from the reference.

Populate the asset cache once with network access before an offline launch:

```bash
uv run --project venv/hdmi python -c 'from huggingface_hub import snapshot_download; snapshot_download("elijahgalahad/any4hdmi-g1-inspire-weave-reference-1605", repo_type="dataset", revision="c716c1e0e316b8688d3825ce9e60607b82ca90c6")'
uv run --project venv/hdmi python -c 'from mjhub import resolve_asset_reference; print(resolve_asset_reference("hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml"))'
```

Copy the working cache to training hosts before setting `HF_HUB_OFFLINE=1`.

## Train

The hardware scale is a command-line choice, not a YAML experiment preset.
For the recorded experiment: eight ranks, 8192 environments **per rank**, 8000
iterations; entropy decays over iterations 4800–8000. Authenticate W&B first.

```bash
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HDMI_CCD_THREAD_SCRATCH=1 WANDB_MODE=online
export PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
bash scripts/launch_ddp.sh 0,1,2,3,4,5,6,7 \
  projects/mimic-lite/scripts/train.py venv/hdmi \
  task=weave-contact-full +exp=hdmi/weave \
  task.num_envs=8192 total_iters=8000 wandb.mode=online
```

Training still samples random starting frames and retains its rewind behavior.
The 8k/16k legacy preset filenames remain compatibility aliases; pass the desired
`total_iters` explicitly. OMOMO presets remain available.

## Play and record

HDMI evaluation/play resets **every motion to literal frame 0**, including
subsequent termination resets, and disables rewind. This is implemented in
HDMI's command subclass; training is unaffected. Override
`+task.command.play_from_start=false` only to request the training sampler.

```bash
uv run --project venv/hdmi projects/mimic-lite/scripts/play.py \
  task=weave-contact-full algo=from_checkpoint checkpoint_path=/path/checkpoint_8000.pt \
  task.num_envs=16 task.termination.root_pos_error.enabled=false \
  task.termination.lost_expected_contact.enabled=false
```

Multi-environment playback already exists in Mimic-Lite. `render_scene.py` adds
render-only object ordering and the requested cinematic camera. It uses the
resolved `cfg.yaml` saved by training and a local checkpoint; it does not bundle
weights or alter the physical simulation to arrange the scene.

```bash
uv run --project venv/hdmi projects/hdmi/scripts/render_scene.py \
  --config /path/to/run/cfg.yaml --checkpoint /path/checkpoint_8000.pt \
  --output /path/to/new-video-directory --num-envs 256 --seconds 20 \
  --center-objects clothesstand floorlamp smalltable woodchair \
  --edge-objects largebox largetable smallbox \
  --pullback-start 3 --pullback-end 15 --far-distance 42.03811978464831
```

This records deterministic policy actions in one MuJoCo scene at 25 fps.
The near camera is 4.8 m away at −30°; 0–3 seconds holds near, 3–15 seconds
pulls back smoothly, and 15–20 seconds holds far. The far camera at −60° has
absolute height 37.206 m, half the earlier video's height. Preferred objects
occupy central cells; explicitly listed edge objects occupy the outermost cells. All source
motions remain available to the native sampler. The recorder uses base/aligned
termination with root-position and lost-contact termination disabled, checks
frame-zero resets, and saves `scene.mp4`, the config and a per-environment
manifest. It is a visual playback, not a success-rate evaluation.

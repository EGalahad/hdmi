# HDMI

HDMI extends [MimicLite](https://github.com/EGalahad/mimic-lite) from robot-only
motion tracking to paired robot-object interaction tracking. The current
release supports MjLab and a single rigid object per environment.

Robot tracking, palm/FK handling, and the standard ROA stages come from the
Mimic-Lite 1.1 branch. This repository owns the HDMI-specific object assets,
motions, observations, rewards, and learning modules; it does not fork the
robot tracking configuration.

## Setup

Clone the HDMI development branches of Active Adaptation and Mimic-Lite, then
clone HDMI's `main`:

```bash
git clone -b dev/hdmi https://github.com/Agent-3154/active-adaptation.git
cd active-adaptation
git clone -b dev/hdmi https://github.com/EGalahad/mimic-lite projects/mimic-lite
git clone https://github.com/EGalahad/hdmi projects/hdmi
```

Create the MjLab environment and install HDMI:

```bash
mkdir -p venv/mjlab
cp projects/mimic-lite/pyproject-mjlab.toml venv/mjlab/pyproject.toml
uv add --project venv/mjlab --editable projects/hdmi
uv sync --project venv/mjlab
```

Refresh project discovery:

```bash
uv run --project venv/mjlab aa-discover-projects
uv run --project venv/mjlab aa-project enable mimic_lite
uv run --project venv/mjlab aa-project enable hdmi
uv run --project venv/mjlab aa-list-tasks
```

Set credentials used for W&B logging and Hugging Face assets:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
export HF_TOKEN=<your_huggingface_token>
```

`any4hdmi` is installed directly from GitHub by HDMI's `pyproject.toml`; no
separate checkout is required. The task preset resolves the
[G1 OMOMO suitcase dataset](https://huggingface.co/datasets/elijahgalahad/any4hdmi-g1-omomo-suitcase)
through its `hf://` URI and reuses the standard Hugging Face cache.

## Train

Train the verified G1 suitcase policy for 4,000 iterations on one 8-GPU node:

```bash
bash scripts/launch_ddp.sh 0,1,2,3,4,5,6,7 \
  projects/mimic-lite/scripts/train.py venv/mjlab \
  task=omomo-suitcase-object-pose \
  +exp=hdmi/ppo
```

The preset selects one suitcase entity per environment, feeds its local pose to
the policy as `object_pose_local`, and tracks the accepted OMOMO suitcase
motions. Each GPU runs 8,192 environments; checkpoints are written every 1,000
of the 4,000 iterations. Change `task/object` and `task/motion` together when
training another paired robot-object dataset.

## Play

Play a W&B checkpoint:

```bash
uv run --project venv/mjlab projects/mimic-lite/scripts/play.py \
  task=omomo-suitcase-object-pose \
  algo=from_checkpoint \
  checkpoint_path=run:elijahgalahad/mimic_lite/nnds9gg2:4000
```

The Viser viewer displays translucent reference meshes for both the robot and
the object.

## Five-rigid PCD-only reproduction

The public five-category PCD-only task uses the five accepted OMOMO datasets,
256 object points, and the PCD encoder. Run the 8,000-iteration reproduction
on one eight-GPU node with:

```bash
export ANY4HDMI_CACHE_BUILD_LOADER_BATCH_SIZE=1
export ANY4HDMI_CACHE_BUILD_BATCH_SIZE=4096

bash scripts/launch_ddp.sh 0,1,2,3,4,5,6,7 \
  projects/mimic-lite/scripts/train.py venv/mjlab \
  task=omomo-rigid5-pcd-only \
  +exp=hdmi/pcd8k
```

The motion config uses public `hf://` dataset references, so a fresh checkout
does not depend on a sibling private dataset directory.

# HDMI

HDMI extends [MimicLite](https://github.com/EGalahad/mimic-lite) from robot-only
motion tracking to paired robot-object interaction tracking. The current
release supports MjLab and a single rigid object per environment.

## Setup

Clone the matching `dev/mimic-hdmi-v08` branches and keep `any4hdmi` next to
Active Adaptation:

```bash
git clone -b dev/mimic-hdmi-v08 https://github.com/Agent-3154/active-adaptation.git
cd active-adaptation
git clone -b dev/mimic-hdmi-v08 https://github.com/EGalahad/mimic-lite projects/mimic-lite
git clone -b dev/mimic-hdmi-v08 https://github.com/EGalahad/hdmi projects/hdmi

cd ..
git clone https://github.com/EGalahad/any4hdmi.git
cd active-adaptation
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

Motion conversion and validation tools live in
[`EGalahad/any4hdmi`](https://github.com/EGalahad/any4hdmi). The training
example below expects the paired dataset at:

```text
any4hdmi/output/g1/omomo_suitcase_gmr_first_root_v2_accepted/
```

## Train

Train the verified G1 suitcase policy for 4,000 iterations on one 8-GPU node:

```bash
bash scripts/launch_ddp.sh 0,1,2,3,4,5,6,7 \
  projects/mimic-lite/scripts/train.py venv/mjlab \
  task=hdmi-base \
  task/object=omomo-suitcase-v2 \
  task/motion=g1/omomo-suitcase-gmr-first-root-v2-accepted \
  task/observation/command=root_joint \
  +task/variant=root-aware-global \
  +algo/ppo/module=large \
  task.num_envs=8192 \
  seed=0 \
  total_iters=4000 \
  checkpoint_interval=1000 \
  upload_interval=1000 \
  wandb.mode=online
```

Each GPU runs 8,192 environments. Change `task/object` and `task/motion`
together when training another paired robot-object dataset.

## Play

Play a W&B checkpoint:

```bash
uv run --project venv/mjlab projects/mimic-lite/scripts/play.py \
  task=hdmi-base \
  task/object=omomo-suitcase-v2 \
  task/motion=g1/omomo-suitcase-gmr-first-root-v2-accepted \
  task/observation/command=root_joint \
  +task/variant=root-aware-global \
  algo=from_checkpoint \
  checkpoint_path=run:<entity>/<project>/<run_id>:<iteration>
```

The Viser viewer displays translucent reference meshes for both the robot and
the object.

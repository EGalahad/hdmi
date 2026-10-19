# HDMI

Robot–object motion tracking with [Mimic-Lite](https://github.com/EGalahad/mimic-lite) and MjLab.

## Setup

Clone the repositories:

```bash
git clone -b dev/hdmi https://github.com/Agent-3154/active-adaptation.git
cd active-adaptation
git clone -b dev/hdmi https://github.com/EGalahad/mimic-lite projects/mimic-lite
git clone https://github.com/EGalahad/hdmi projects/hdmi
```

Install dependencies:

```bash
mkdir -p venv/hdmi
cp projects/hdmi/pyproject-mjlab.toml venv/hdmi/pyproject.toml
cp projects/hdmi/uv-mjlab.lock venv/hdmi/uv.lock
uv sync --project venv/hdmi --locked
```

The environment uses the [HDMI MuJoCo-Warp fork](https://github.com/EGalahad/mujoco_warp/tree/hdmi/3.11-repro)
for mesh collision filtering and CCD scratch memory reuse. Enable the scratch patch:

```bash
export HDMI_CCD_THREAD_SCRATCH=1
```

Enable the projects:

```bash
uv run --project venv/hdmi aa-discover-projects
uv run --project venv/hdmi aa-project enable mimic_lite
uv run --project venv/hdmi aa-project enable hdmi
uv run --project venv/hdmi aa-list-tasks
```

Authenticate W&B:

```bash
uv run --project venv/hdmi wandb login
```

## Train

Run from the Active Adaptation repository. Task presets select the robot, objects
and motion data; referenced Hugging Face assets are downloaded automatically
before distributed training starts.

```bash
bash projects/hdmi/scripts/train.sh 0,1,2,3,4,5,6,7 \
  task=weave-contact-full +exp=hdmi/weave \
  task.num_envs=8192 total_iters=8000 wandb.mode=online
```

## Play and record

Play a checkpoint:

```bash
uv run --project venv/hdmi projects/mimic-lite/scripts/play.py \
  task=weave-contact-full algo=from_checkpoint checkpoint_path=/path/to/checkpoint.pt \
  task.num_envs=16 task.termination.root_pos_error.enabled=false \
  task.termination.lost_expected_contact.enabled=false
```

Record a MuJoCo scene using the saved training configuration:

```bash
uv run --project venv/hdmi projects/hdmi/scripts/render_scene.py \
  --config /path/to/run/cfg.yaml --checkpoint /path/to/checkpoint.pt \
  --output /path/to/video --num-envs 256 --seconds 20
```

See `render_scene.py --help` for camera and layout options.

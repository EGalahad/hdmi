#!/usr/bin/env bash
set -euo pipefail

AA_ROOT=/home/elijah/Documents/projects/simple-tracking/active-adaptation
RUN_ROOT=/data/elijah/hdmi-omomo-gmr-spatial-error-1k
mkdir -p "$RUN_ROOT"/{hydra,logs,tmp,wandb,torchinductor,warp,motion_cache}
exec > >(tee -a "$RUN_ROOT/logs/train.log") 2>&1

while pgrep -f 'uv run --project venv/mjlab python -c.*resolve_asset_reference' >/dev/null; do
  echo "Waiting for the existing venv asset smoke to finish"
  sleep 10
done
while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)" -ne 0 ]; do
  echo "Waiting for all eight GPUs to become idle"
  sleep 30
done

cd "$AA_ROOT"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR="$RUN_ROOT/tmp"
export WANDB_DIR="$RUN_ROOT/wandb"
export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/torchinductor"
export WARP_CACHE_PATH="$RUN_ROOT/warp"
export ANY4HDMI_QPOS_CACHE_ROOT="$RUN_ROOT/motion_cache"
export UV_BIN=/home/elijah/.local/bin/uv

bash scripts/launch_ddp.sh 0,1,2,3,4,5,6,7 \
  projects/mimic-lite/scripts/train.py venv/mjlab \
  task=hdmi-base \
  task/object=suitcase \
  task/motion=g1/omomo-suitcase-gmr-accepted \
  +algo/ppo/module=large \
  task.num_envs=8192 \
  seed=0 \
  total_iters=1000 \
  checkpoint_interval=250 \
  upload_interval=250 \
  wandb.mode=online \
  hydra.run.dir="$RUN_ROOT/hydra"

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash projects/hdmi/scripts/train.sh <gpu_ids> [Hydra overrides...]" >&2
  exit 2
fi
GPU_IDS=$1
shift
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$ROOT"
UV_PROJECT=${AA_UV_PROJECT:-venv/hdmi}
export HDMI_CCD_THREAD_SCRATCH=${HDMI_CCD_THREAD_SCRATCH:-1}
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}

# One process fills the shared cache; DDP workers then use it without concurrent
# Hub requests. An explicitly offline invocation validates the existing cache.
uv run --project "$UV_PROJECT" --no-sync "$SCRIPT_DIR/prepare_assets.py" "$@" \
  hydra.run.dir=. hydra.output_subdir=null hydra.job.chdir=false
export HF_HUB_OFFLINE=1
exec bash scripts/launch_ddp.sh "$GPU_IDS" \
  projects/mimic-lite/scripts/train.py "$UV_PROJECT" "$@"

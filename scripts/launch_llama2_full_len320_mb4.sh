#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [GPU_IDS] [STAGE]" >&2
  echo "  GPU_IDS defaults to 0,1; STAGE defaults to all." >&2
  exit 2
fi

GPU_IDS=${1:-0,1}
STAGE=${2:-all}
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_IDS must be comma-separated physical GPU indices, e.g. 0,1" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

CONFIG=configs/experiments/llama2_full_len320_mb4.yaml
RUN_DIR=$(python - "$CONFIG" <<'PY'
import sys
from fedproxy.config import load_config

print(load_config(sys.argv[1])["run"]["output_dir"])
PY
)
if [[ "$STAGE" == all ]] && [[ -e "$RUN_DIR/data_manifest.json" || -e "$RUN_DIR/compression" || -e "$RUN_DIR/checkpoints" ]]; then
  echo "Existing run artifacts found in $RUN_DIR; refusing to restart the full pipeline." >&2
  echo "Use a specific stage (for example, resume, fuse, or evaluate) to continue." >&2
  exit 2
fi

exec bash "$SCRIPT_DIR/launch_experiment.sh" "$CONFIG" "$GPU_IDS" "$STAGE"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 GPU_IDS [all|obqa|arc_easy|arc_challenge|commonsense_qa|sst2|mrpc|rte|mnli]" >&2
  exit 2
fi
GPU_IDS=$1
SELECTED=${2:-all}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python -m fedproxy.ma.homogeneous run --gpu-ids "$GPU_IDS" --task "$SELECTED"

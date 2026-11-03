#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 GPU_IDS [both|fedproxy|ma] [ROOT]" >&2
  exit 2
fi
GPU_IDS=$1
METHOD=${2:-both}
ROOT=${3:-runs/heterogeneous_ratio30}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export FEDPROXY_GPU_IDS="$GPU_IDS"
export FEDPROXY_REQUIRED_PARALLEL_CLIENTS=2
export TOKENIZERS_PARALLELISM=false

exec python -u -m fedproxy.ratio30_heterogeneous run --gpu-ids "$GPU_IDS" --method "$METHOD" --root "$ROOT"

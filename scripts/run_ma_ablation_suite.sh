#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 4 ]]; then
  echo "Usage: $0 [GPU_IDS] [all|A1|A2|A3|A4] [REFERENCE_MA] [OUTPUT_ROOT]" >&2
  exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
python -m fedproxy.ma.ablations run --gpu-ids "${1:-0,1}" --group "${2:-all}" \
  --reference "${3:-runs/heterogeneous_ratio30/MA}" \
  --root "${4:-runs/heterogeneous_ratio30/ablations}"

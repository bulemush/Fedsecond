#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 CONFIG GPU_IDS [STAGE]" >&2
  echo "  STAGE: all|prepare|compress|dry-run|train|resume|fuse|evaluate|evaluate-original|evaluate-proxy|evaluate-fused (default: all)" >&2
  exit 2
fi

CONFIG=$1
GPU_IDS=$2
STAGE=${3:-all}

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export FEDPROXY_GPU_IDS="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

readarray -t RESOLVED < <(
  python - "$CONFIG" <<'PY'
import sys
from fedproxy.config import load_config

cfg = load_config(sys.argv[1])
print(cfg["run"]["output_dir"])
print(cfg["federated"]["rounds"])
PY
)

RUN_DIR=${RESOLVED[0]}
ROUNDS=${RESOLVED[1]}

if [[ "$ROUNDS" == "None" || "$ROUNDS" == "null" ]]; then
  echo "The experiment config must define federated.rounds" >&2
  exit 2
fi

prepare() {
  python -m fedproxy.cli prepare-data --config "$CONFIG"
}

compress() {
  python -m fedproxy.cli compress --config "$CONFIG"
}

dry_run() {
  python -m fedproxy.cli train --config "$CONFIG" --dry-run
}

train() {
  python -m fedproxy.cli train --config "$CONFIG"
}

resume() {
  local latest
  if [[ ! -d "$RUN_DIR/checkpoints" ]]; then
    echo "No checkpoint directory found under $RUN_DIR; run the train stage first" >&2
    exit 2
  fi
  latest=$(find "$RUN_DIR/checkpoints" -maxdepth 1 -type d -name 'round_[0-9][0-9][0-9][0-9]' 2>/dev/null | sort | tail -n 1)
  if [[ -z "$latest" ]]; then
    echo "No completed checkpoint found under $RUN_DIR/checkpoints" >&2
    exit 2
  fi
  python -m fedproxy.cli train --config "$CONFIG" --resume "$latest"
}

fuse() {
  python -m fedproxy.cli fuse \
    --config "$CONFIG" \
    --checkpoint "$RUN_DIR/checkpoints/round_$(printf '%04d' "$ROUNDS")"
}

evaluate() {
  evaluate_original
  evaluate_proxy
  evaluate_fused
}

evaluate_original() {
  python -m fedproxy.cli evaluate --config "$CONFIG" --model-kind original
}

evaluate_proxy() {
  python -m fedproxy.cli evaluate --config "$CONFIG" --model-kind proxy
}

evaluate_fused() {
  python -m fedproxy.cli evaluate --config "$CONFIG" --model-kind fused
}

case "$STAGE" in
  all)
    prepare
    compress
    dry_run
    train
    fuse
    evaluate
    ;;
  prepare) prepare ;;
  compress) compress ;;
  dry-run) dry_run ;;
  train) train ;;
  resume) resume ;;
  fuse) fuse ;;
  evaluate) evaluate ;;
  evaluate-original) evaluate_original ;;
  evaluate-proxy) evaluate_proxy ;;
  evaluate-fused) evaluate_fused ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    exit 2
    ;;
esac

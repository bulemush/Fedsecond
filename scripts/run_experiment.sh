#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 CONFIG GPU_IDS [STAGE]" >&2
  echo "  STAGE: all|reuse-audit|reuse-all|audit|prepare|compress|dry-run|train|resume|fuse|evaluate|evaluate-original|evaluate-proxy|evaluate-fused (default: all)" >&2
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

audit() {
  python -m fedproxy.cli audit-truncation --config "$CONFIG"
}

reuse_inputs() {
  local source_dir source_abs name destination expected actual
  source_dir=$(python - "$CONFIG" <<'PY'
import sys
from fedproxy.config import load_config

print(load_config(sys.argv[1])["run"].get("reuse_artifacts_from", ""))
PY
)
  if [[ -z "$source_dir" ]]; then
    echo "run.reuse_artifacts_from is required for reuse-all" >&2
    exit 2
  fi
  if [[ ! -f "$source_dir/compression/compression_manifest.json" || ! -d "$source_dir/compression/proxy" || ! -f "$source_dir/data_manifest.json" ]]; then
    echo "Source run lacks compression or data manifest: $source_dir" >&2
    exit 2
  fi
  source_abs=$(realpath -- "$source_dir")
  mkdir -p "$RUN_DIR"
  if [[ "$(realpath -- "$RUN_DIR")" == "$source_abs" ]]; then
    echo "Refusing to reuse a run as its own output" >&2
    exit 2
  fi
  for name in compression data_manifest.json; do
    destination="$RUN_DIR/$name"
    expected="$source_abs/$name"
    if [[ -L "$destination" ]]; then
      actual=$(realpath -- "$destination")
      if [[ "$actual" != "$expected" ]]; then
        echo "Existing link points elsewhere: $destination" >&2
        exit 2
      fi
    elif [[ -e "$destination" ]]; then
      echo "Refusing to replace existing artifact: $destination" >&2
      exit 2
    else
      ln -s -- "$expected" "$destination"
    fi
  done
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
    audit
    compress
    dry_run
    train
    fuse
    evaluate
    ;;
  reuse-audit)
    reuse_inputs
    audit
    ;;
  reuse-all)
    reuse_inputs
    audit
    dry_run
    train
    fuse
    evaluate_fused
    ;;
  audit) audit ;;
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

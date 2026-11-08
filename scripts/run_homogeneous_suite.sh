#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 GPU_IDS [all|obqa|arc_easy|arc_challenge|commonsense_qa|sst2|mrpc|rte|mnli]" >&2
  exit 2
fi

GPU_IDS=$1
SELECTED=${2:-all}
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_IDS must be comma-separated physical GPU indices, e.g. 0,1" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

TASKS=(obqa arc_easy arc_challenge commonsense_qa sst2 mrpc rte mnli)
case "$SELECTED" in
  all|obqa|arc_easy|arc_challenge|commonsense_qa|sst2|mrpc|rte|mnli) ;;
  *) echo "Unknown task: $SELECTED" >&2; exit 2 ;;
esac

for task in "${TASKS[@]}"; do
  if [[ "$SELECTED" != all && "$SELECTED" != "$task" ]]; then
    continue
  fi
  config="configs/experiments/llama2_homogeneous_${task}_len320_mb4.yaml"
  run_dir="runs/llama2_homogeneous_${task}_len320_mb4_seed42"
  if [[ -f "$run_dir/evaluation/fused_summary.json" ]]; then
    if python - "$run_dir/evaluation/fused_summary.json" "$task" <<'PY'
import json
import sys
from pathlib import Path
from fedproxy.evaluation.protocol import validate_mnli_evaluation

try:
    summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if summary.get("model_kind") != "fused" or summary.get("synthetic") or set(summary.get("tasks", {})) != {sys.argv[2]}:
        raise ValueError("Incomplete or unexpected task evaluation")
    validate_mnli_evaluation(summary)
except (ValueError, TypeError) as error:
    print(f"[suite] Evaluation needs refreshing: {error}", flush=True)
    raise SystemExit(1)
PY
    then
      echo "[suite] Already complete: $task ($run_dir/evaluation/fused_summary.json)"
      continue
    fi
    echo "[suite] Re-evaluating existing fused model: $task"
    bash "$SCRIPT_DIR/run_experiment.sh" "$config" "$GPU_IDS" evaluate-fused
    continue
  fi
  if [[ -e "$run_dir/data_manifest.json" || -e "$run_dir/compression" || -e "$run_dir/checkpoints" || -e "$run_dir/exports" ]]; then
    echo "[suite] Partial artifacts in $run_dir; refusing to overwrite them." >&2
    echo "[suite] Resume or complete that task with scripts/run_experiment.sh, then rerun the suite." >&2
    exit 2
  fi
  stage=all
  if [[ "$task" != obqa ]]; then
    stage=reuse-compression-all
  fi
  echo "[suite] Starting task=$task stage=$stage GPUs=$GPU_IDS"
  bash "$SCRIPT_DIR/run_experiment.sh" "$config" "$GPU_IDS" "$stage"
  echo "[suite] Finished task=$task"
done

if [[ "$SELECTED" == all ]]; then
  python "$SCRIPT_DIR/summarize_homogeneous_suite.py"
fi

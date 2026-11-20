#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 4 ]]; then
  echo "Usage: $0 [GPU_IDS] [all|A1|A2|A3|A4] [REFERENCE_MA] [OUTPUT_ROOT]" >&2
  exit 2
fi
GPU_IDS=${1:-0,1}
if [[ ! "$GPU_IDS" =~ ^([0-9]+),([0-9]+)$ ]] || [[ "${BASH_REMATCH[1]}" == "${BASH_REMATCH[2]}" ]]; then
  echo "Two distinct GPU IDs are required, e.g. 0,1" >&2
  exit 2
fi
GROUP=${2:-all}
REFERENCE=${3:-runs/heterogeneous_ratio30/MA}
ROOT=${4:-runs/heterogeneous_ratio30/ablations}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m fedproxy.ma.ablations validate --reference "$REFERENCE" --root "$ROOT" --group "$GROUP"
mkdir -p "$ROOT/logs"
STAMP=$(date '+%Y%m%d_%H%M%S')
LOG_PATH="$ROOT/logs/${GROUP}_${STAMP}_$$.log"
nohup setsid bash "$SCRIPT_DIR/run_ma_ablation_suite.sh" "$GPU_IDS" "$GROUP" "$REFERENCE" "$ROOT" \
  >"$LOG_PATH" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$ROOT/logs/${GROUP}_$$.pid"
echo "Started PID $PID"
echo "Log: $LOG_PATH"
echo "Ctrl+C stops this log view; the background suite continues."
tail -n +1 --pid="$PID" -f "$LOG_PATH"
wait "$PID"

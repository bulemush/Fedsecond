#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [GPU_IDS] [all|TASK]" >&2
  exit 2
fi
GPU_IDS=${1:-0,1}
SELECTED=${2:-all}
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_IDS must be comma-separated physical GPU indices, e.g. 0,1" >&2
  exit 2
fi
case "$SELECTED" in
  all|obqa|arc_easy|arc_challenge|commonsense_qa|sst2|mrpc|rte|mnli) ;;
  *) echo "Unknown task: $SELECTED" >&2; exit 2 ;;
esac
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

LOG_DIR=runs/MA_homogeneous/logs
mkdir -p "$LOG_DIR"
STAMP=$(date '+%Y%m%d_%H%M%S')
LOG_PATH="$LOG_DIR/${SELECTED}_${STAMP}_$$.log"
PID_PATH="$LOG_DIR/${SELECTED}_$$.pid"

nohup setsid bash "$SCRIPT_DIR/run_ma_homogeneous_suite.sh" "$GPU_IDS" "$SELECTED" \
  >"$LOG_PATH" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_PATH"

echo "Started PID $PID"
echo "Log: $LOG_PATH"
echo "Follow: tail -f $LOG_PATH"
echo "Ctrl+C stops only this log view; the nohup job continues."
tail -n +1 --pid="$PID" -f "$LOG_PATH"
wait "$PID"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 3 ]]; then
  echo "Usage: $0 [GPU_IDS] [both|fedproxy|ma] [ROOT]" >&2
  exit 2
fi
GPU_IDS=${1:-0,1}
METHOD=${2:-both}
ROOT=${3:-runs/heterogeneous_ratio30}
if [[ ! "$GPU_IDS" =~ ^([0-9]+),([0-9]+)$ ]] || [[ "${BASH_REMATCH[1]}" == "${BASH_REMATCH[2]}" ]]; then
  echo "Two distinct GPU IDs are required, e.g. 2,3" >&2
  exit 2
fi
case "$METHOD" in both|fedproxy|ma) ;; *) echo "Unknown method: $METHOD" >&2; exit 2 ;; esac
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
STAMP=$(date '+%Y%m%d_%H%M%S')
LOG_PATH="$LOG_DIR/${METHOD}_${STAMP}_$$.log"
PID_PATH="$LOG_DIR/${METHOD}_$$.pid"

nohup setsid bash "$SCRIPT_DIR/run_ratio30_heterogeneous_suite.sh" "$GPU_IDS" "$METHOD" "$ROOT" \
  >"$LOG_PATH" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_PATH"
echo "Started PID $PID"
echo "Log: $LOG_PATH"
echo "Ctrl+C stops only this log view; the nohup job continues."
tail -n +1 --pid="$PID" -f "$LOG_PATH"
wait "$PID"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 CONFIG GPU_IDS [STAGE]" >&2
  echo "Starts run_experiment.sh with nohup; STAGE defaults to all." >&2
  exit 2
fi

CONFIG=$1
GPU_IDS=$2
STAGE=${3:-all}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

RUN_DIR=$(python - "$CONFIG" <<'PY'
import sys
from fedproxy.config import load_config

print(load_config(sys.argv[1])["run"]["output_dir"])
PY
)

mkdir -p "$RUN_DIR/logs"
STAMP=$(date '+%Y%m%d_%H%M%S')
LOG_PATH="$RUN_DIR/logs/${STAGE}_${STAMP}.log"
PID_PATH="$RUN_DIR/logs/${STAGE}.pid"

nohup bash "$SCRIPT_DIR/run_experiment.sh" "$CONFIG" "$GPU_IDS" "$STAGE" \
  >"$LOG_PATH" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_PATH"

echo "Started PID $PID"
echo "Log: $LOG_PATH"
echo "Follow: tail -f $LOG_PATH"

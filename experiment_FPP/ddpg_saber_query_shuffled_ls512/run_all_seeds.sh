#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${SEEDS:-1 2 3 4 5}"
STEPS="${STEPS:-1000000}"
DEVICE="${DEVICE:-cuda}"
SUPPORT_QUERY_MODE="${SUPPORT_QUERY_MODE:-shuffled_policy}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

mkdir -p "$LOG_ROOT"
pids=()

for seed in $SEEDS; do
  log_file="$LOG_ROOT/${SUPPORT_QUERY_MODE}_s${seed}_${STEPS}.log"
  if [[ -e "$log_file" ]]; then
    log_file="$LOG_ROOT/${SUPPORT_QUERY_MODE}_s${seed}_${STEPS}_pid$$.log"
  fi
  command=(
    python3 "$SCRIPT_DIR/run.py"
    --seed "$seed"
    --steps "$STEPS"
    --device "$DEVICE"
    --support-query-mode "$SUPPORT_QUERY_MODE"
  )
  if [[ -n "$OUTPUT_ROOT" ]]; then
    command+=(--output-root "$OUTPUT_ROOT")
  fi
  command+=("$@")
  echo "[launch] seed=$seed steps=$STEPS mode=$SUPPORT_QUERY_MODE log=$log_file"
  "${command[@]}" >"$log_file" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"

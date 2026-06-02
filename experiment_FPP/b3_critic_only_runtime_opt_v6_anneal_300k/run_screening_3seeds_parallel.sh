#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS="${STEPS:-1000000}"
DEVICE="${DEVICE:-cuda}"
SEED_LIST="${SEED_LIST:-1 2 3 4 5}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

pids=()
for seed in $SEED_LIST; do
  log_file="$LOG_DIR/v6_anneal_300k_s${seed}_${STEPS}.log"
  echo "[launch] method=b3_critic_only_runtime_opt_v6_anneal_300k seed=${seed} steps=${STEPS} device=${DEVICE} log=${log_file}"
  python3 "$SCRIPT_DIR/run.py" --seed "$seed" --steps "$STEPS" --device "$DEVICE" "$@" >"$log_file" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"

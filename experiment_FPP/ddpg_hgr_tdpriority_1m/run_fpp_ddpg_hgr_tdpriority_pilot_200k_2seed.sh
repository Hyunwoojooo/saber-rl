#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SEED_LIST="${SEED_LIST:-1 2}"
STEPS="${STEPS:-200000}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_FPP/runs/ddpg_hgr_tdpriority_original_pair_is_pilot200k}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/ddpg_hgr_tdpriority_original_pair_is_pilot200k}"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
pids=()
for seed in $SEED_LIST; do
  log_file="$LOG_DIR/seed${seed}.log"
  echo "[launch] method=hgr pilot seed=$seed steps=$STEPS device=$DEVICE output_root=$OUTPUT_ROOT log=$log_file"
  python3 "$SCRIPT_DIR/run.py" --seed "$seed" --steps "$STEPS" --device "$DEVICE" --output_root "$OUTPUT_ROOT" > "$log_file" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"

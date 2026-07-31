#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SEED_LIST="${SEED_LIST:-1 2 3 4 5}"
STEPS="${STEPS:-1000000}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_FPP/runs/td3_her_high_replay_ratio_x4}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/td3_her_high_replay_ratio_x4_1m}"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

pids=()
for seed in $SEED_LIST; do
  log_file="$LOG_DIR/seed${seed}.log"
  echo "[launch] method=td3_her_high_rr_x4 seed=$seed steps=$STEPS updates_per_step=4 device=$DEVICE output_root=$OUTPUT_ROOT log=$log_file"
  cmd=(
    python3 "$SCRIPT_DIR/run.py"
    --seed "$seed"
    --steps "$STEPS"
    --device "$DEVICE"
    --output_root "$OUTPUT_ROOT"
  )
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    continue
  fi
  "${cmd[@]}" > "$log_file" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SEED_LIST="${SEED_LIST:-1 2 3 4 5}"
STEPS="${STEPS:-1000000}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_FPP/runs/ddpg_hgr_tdpriority_original_pair_is_1m}"
RESUME_ROOT="${RESUME_ROOT:-$OUTPUT_ROOT}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/ddpg_hgr_tdpriority_original_pair_is_1m}"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
pids=()
for seed in $SEED_LIST; do
  log_file="$LOG_DIR/seed${seed}.log"
  run_name="fetch_pick_and_place_ddpg_hgr_tdpriority_s${seed}_${STEPS}"
  echo "[launch] method=hgr seed=$seed steps=$STEPS updates_per_step=1 device=$DEVICE output_root=$OUTPUT_ROOT log=$log_file"
  cmd=(python3 "$SCRIPT_DIR/run.py" --seed "$seed" --steps "$STEPS" --device "$DEVICE" --output_root "$OUTPUT_ROOT")
  if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
    cmd+=(--resume --resume_path "$RESUME_ROOT/$run_name/checkpoint.pt")
  fi
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
  if ! wait "$pid"; then status=1; fi
done
exit "$status"

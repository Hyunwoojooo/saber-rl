#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SEED_LIST="${SEED_LIST:-1 2 3 4 5}"
STEPS="${STEPS:-1000000}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_FPP/runs/td3_her_high_replay_ratio_x4}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/td3_her_high_replay_ratio_x4_resume_after_reboot}"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

resume_path_for_seed() {
  local seed="$1"
  local run_dir="$OUTPUT_ROOT/fetch_pick_and_place_td3_her_high_rr_x4_s${seed}_${STEPS}_ls512"
  if [[ "$seed" == "5" ]]; then
    printf '%s
' "$run_dir/best_checkpoint.pt"
  else
    printf '%s
' "$run_dir/checkpoint.pt"
  fi
}

pids=()
for seed in $SEED_LIST; do
  resume_path="$(resume_path_for_seed "$seed")"
  if [[ ! -s "$resume_path" ]]; then
    echo "[error] missing or empty resume checkpoint for seed=$seed: $resume_path" >&2
    exit 1
  fi
  log_file="$LOG_DIR/seed${seed}.log"
  echo "[resume-launch] method=td3_her_high_rr_x4 seed=$seed steps=$STEPS device=$DEVICE resume_path=$resume_path log=$log_file"
  cmd=(
    python3 "$SCRIPT_DIR/run.py"
    --seed "$seed"
    --steps "$STEPS"
    --device "$DEVICE"
    --output_root "$OUTPUT_ROOT"
    --resume
    --resume_path "$resume_path"
  )
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '
'
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

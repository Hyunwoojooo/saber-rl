#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_FPP/runs/fpp_ddpg_aer_noher_correctness20k}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/fpp_ddpg_aer_noher_correctness20k}"
DRY_RUN="${DRY_RUN:-0}"
SEED=101
STEPS=20000
ARMS=(ddpg_er ddpg_annealing_er ddpg_aer)

if [[ "$DRY_RUN" != "1" ]]; then
  for arm in "${ARMS[@]}"; do
    run_dir="$OUTPUT_ROOT/fetch_pick_and_place_${arm}_noher_s${SEED}_${STEPS}"
    log_file="$LOG_DIR/${arm}_seed${SEED}.log"
    if [[ -e "$run_dir" ]]; then
      echo "refusing to overwrite existing run: $run_dir" >&2
      exit 1
    fi
    if [[ -e "$log_file" ]]; then
      echo "refusing to overwrite existing log: $log_file" >&2
      exit 1
    fi
  done
  mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
fi

# Run the three arms sequentially on the same device. This is an exact 20k
# override; intentionally do not pass the trainer's legacy --smoke flag.
for arm in "${ARMS[@]}"; do
  log_file="$LOG_DIR/${arm}_seed${SEED}.log"
  cmd=(
    "$PYTHON" "$SCRIPT_DIR/run.py"
    --arm "$arm"
    --seed "$SEED"
    --steps "$STEPS"
    --device "$DEVICE"
    --output_root "$OUTPUT_ROOT"
  )
  echo "[launch] arm=$arm seed=$SEED steps=$STEPS device=$DEVICE log=$log_file"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    continue
  fi
  if ! "${cmd[@]}" >"$log_file" 2>&1; then
    echo "failed: arm=$arm seed=$SEED; inspect $log_file" >&2
    exit 1
  fi
done

if [[ "$DRY_RUN" != "1" ]]; then
  echo "[complete] 20k correctness triplet completed"
fi

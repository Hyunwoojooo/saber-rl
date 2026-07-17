#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_FPP/runs/fpp_ddpg_aer_noher_1m}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/fpp_ddpg_aer_noher_1m}"
DRY_RUN="${DRY_RUN:-0}"
SEED_LIST="${SEED_LIST:-1 2 3 4 5 6 7 8 9 10}"
STEPS=1000000
ARMS=(ddpg_er ddpg_annealing_er ddpg_aer)
read -r -a SEEDS <<< "$SEED_LIST"

for seed in "${SEEDS[@]}"; do
  case "$seed" in
    1|2|3|4|5|6|7|8|9|10) ;;
    *) echo "SEED_LIST must be a subset of: 1 2 3 4 5 6 7 8 9 10" >&2; exit 2 ;;
  esac
done

if [[ "$DRY_RUN" != "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    for arm in "${ARMS[@]}"; do
      run_dir="$OUTPUT_ROOT/fetch_pick_and_place_${arm}_noher_s${seed}_${STEPS}"
      log_file="$LOG_DIR/${arm}_seed${seed}.log"
      if [[ -e "$run_dir" ]]; then
        echo "refusing to overwrite existing run: $run_dir" >&2
        exit 1
      fi
      if [[ -e "$log_file" ]]; then
        echo "refusing to overwrite existing log: $log_file" >&2
        exit 1
      fi
    done
  done
  mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
fi

# Default: all paired triplets sequentially on one device. For a multi-GPU
# machine, invoke this script once per device with disjoint SEED_LIST shards.
for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    log_file="$LOG_DIR/${arm}_seed${seed}.log"
    cmd=(
      "$PYTHON" "$SCRIPT_DIR/run.py"
      --arm "$arm"
      --seed "$seed"
      --steps "$STEPS"
      --device "$DEVICE"
      --output_root "$OUTPUT_ROOT"
    )
    echo "[launch] arm=$arm seed=$seed steps=$STEPS device=$DEVICE log=$log_file"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '%q ' "${cmd[@]}"
      printf '\n'
      continue
    fi
    if ! "${cmd[@]}" >"$log_file" 2>&1; then
      echo "failed: arm=$arm seed=$seed; inspect $log_file" >&2
      exit 1
    fi
  done
done

if [[ "$DRY_RUN" != "1" ]]; then
  echo "[complete] 1M confirmatory triplets completed for seeds: ${SEEDS[*]}"
fi

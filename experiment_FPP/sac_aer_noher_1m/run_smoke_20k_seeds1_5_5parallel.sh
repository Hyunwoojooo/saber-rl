#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_FPP/runs/fpp_sac_aer_noher_smoke20k_s1_5}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/fpp_sac_aer_noher_smoke20k_s1_5}"
DRY_RUN="${DRY_RUN:-0}"
SEED_LIST="${SEED_LIST:-1 2 3 4 5}"
STEPS=20000
ARMS=(sac_er sac_annealing_er sac_aer)
ALLOWED_SEEDS=" 1 2 3 4 5 "

# Keep five independent trainers from each creating a full CPU thread pool.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

read -r -a SEEDS <<< "$SEED_LIST"
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "SEED_LIST must contain at least one seed" >&2
  exit 2
fi

declare -A SEEN_SEEDS=()
for seed in "${SEEDS[@]}"; do
  if [[ "$ALLOWED_SEEDS" != *" $seed "* ]]; then
    echo "invalid smoke seed: $seed (allowed: 1 2 3 4 5)" >&2
    exit 2
  fi
  if [[ -n "${SEEN_SEEDS[$seed]:-}" ]]; then
    echo "duplicate smoke seed: $seed" >&2
    exit 2
  fi
  SEEN_SEEDS["$seed"]=1
done

# Check every destination before starting any worker so a stale artifact cannot
# leave the three-arm/paired-seed block only partly launched.
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

build_command() {
  local arm="$1"
  local seed="$2"
  COMMAND=(
    "$PYTHON" "$SCRIPT_DIR/run.py"
    --arm "$arm"
    --seed "$seed"
    --steps "$STEPS"
    --device "$DEVICE"
    --output_root "$OUTPUT_ROOT"
  )
}

if [[ "$DRY_RUN" == "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    for arm in "${ARMS[@]}"; do
      build_command "$arm" "$seed"
      printf '[dry-run] arm=%s seed=%s: ' "$arm" "$seed"
      printf '%q ' "${COMMAND[@]}"
      printf '\n'
      "${COMMAND[@]}" --dry-run >/dev/null
    done
  done
  exit 0
fi

run_seed() {
  local seed="$1"
  local arm log_file
  for arm in "${ARMS[@]}"; do
    log_file="$LOG_DIR/${arm}_seed${seed}.log"
    build_command "$arm" "$seed"
    echo "[launch] arm=$arm seed=$seed steps=$STEPS device=$DEVICE log=$log_file"
    if ! "${COMMAND[@]}" >"$log_file" 2>&1; then
      echo "[failed] arm=$arm seed=$seed; inspect $log_file" >&2
      return 1
    fi
    echo "[complete] arm=$arm seed=$seed"
  done
}

echo "[block] five-wide seed workers; within each seed: ${ARMS[*]}"
pids=()
launched_seeds=()
for seed in "${SEEDS[@]}"; do
  run_seed "$seed" &
  pids+=("$!")
  launched_seeds+=("$seed")
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "[worker-failed] seed=${launched_seeds[$index]}" >&2
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "[block-failed] one or more seed workers failed" >&2
  exit "$status"
fi

echo "[block-complete] 20k smoke completed for seeds: ${SEEDS[*]}"

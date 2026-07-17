#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_FPP/runs/td3_her_per_style_loss_weighting_ls512}"
LOG_DIR="$OUTPUT_ROOT/launcher_logs"
STEPS="${STEPS:-1000000}"
DEVICE="${DEVICE:-cuda}"
SEEDS=(1 2 3 4 5)

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/launcher.log") 2>&1

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] preparing TD3+HER+PER-style seeds ${SEEDS[*]}"
echo "steps=$STEPS device=$DEVICE output_root=$OUTPUT_ROOT"

for seed in "${SEEDS[@]}"; do
  run_dir="$OUTPUT_ROOT/fetch_pick_and_place_td3_her_per_style_loss_weighting_ls512_s${seed}_${STEPS}_ls512"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite existing run directory: $run_dir" >&2
    exit 3
  fi
done

pids=()
terminate_children() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] stopping child processes"
  if ((${#pids[@]} > 0)); then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
  exit 130
}
trap terminate_children INT TERM

for seed in "${SEEDS[@]}"; do
  seed_log="$LOG_DIR/seed${seed}.log"
  "$PYTHON_BIN" "$SCRIPT_DIR/run.py" \
    --seed "$seed" \
    --steps "$STEPS" \
    --device "$DEVICE" \
    --output_root "$OUTPUT_ROOT" \
    >"$seed_log" 2>&1 &
  pids+=("$!")
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] launched seed=$seed pid=${pids[-1]} log=$seed_log"
done

status=0
for index in "${!pids[@]}"; do
  seed="${SEEDS[$index]}"
  pid="${pids[$index]}"
  if wait "$pid"; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] finished seed=$seed status=0"
  else
    code=$?
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] finished seed=$seed status=$code" >&2
    status=$code
  fi
done

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] all seeds exited aggregate_status=$status"
exit "$status"

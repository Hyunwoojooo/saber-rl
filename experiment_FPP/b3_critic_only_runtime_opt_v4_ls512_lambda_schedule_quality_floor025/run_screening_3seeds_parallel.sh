#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
STEPS="${STEPS:-1000000}"
DEVICE="${DEVICE:-cuda}"
SEED_LIST="${SEED_LIST:-1 2 3 4 5}"
QUALITY_MODE="${QUALITY_MODE:-fpp_phase_soft}"
QUALITY_FLOOR="${QUALITY_FLOOR:-0.25}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/SABER/FetchPickAndPlace/runs/b3_critic_only_runtime_opt_v4_ls512_lambda_schedule_quality_floor025}"
VERIFY_RUNTIME_OPTIMIZATION="${VERIFY_RUNTIME_OPTIMIZATION:-0}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

base_args=(
  --support_quality_mode "$QUALITY_MODE"
  --support_quality_apply_to_weight
  --support_quality_floor "$QUALITY_FLOOR"
  --output_root "$OUTPUT_ROOT"
)

if [[ "$VERIFY_RUNTIME_OPTIMIZATION" == "1" ]]; then
  base_args+=(--verify_runtime_optimization)
fi

pids=()
for seed in $SEED_LIST; do
  log_file="$LOG_DIR/ls512_lambda_schedule_quality_qf${QUALITY_FLOOR}_s${seed}_${STEPS}.log"
  echo "[launch] seed=${seed} steps=${STEPS} device=${DEVICE} mode=${QUALITY_MODE} floor=${QUALITY_FLOOR} output_root=${OUTPUT_ROOT} log=${log_file}"
  python3 "$SCRIPT_DIR/run.py" \
    --seed "$seed" \
    --steps "$STEPS" \
    --device "$DEVICE" \
    "${base_args[@]}" \
    "$@" >"$log_file" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"

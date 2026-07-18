#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$ROOT/experiment_FPP/td3_hgr_tdpriority_1m"
PY="$ROOT/.venv/bin/python"
DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiment_FPP/runs/td3_hgr_tdpriority_smoke20k_s1_5}"
LOG_ROOT="${LOG_ROOT:-$ROOT/experiment_FPP/logs/td3_hgr_tdpriority_smoke20k_s1_5}"
STEPS=20000
SEEDS=(1 2 3 4 5)

mkdir -p "$LOG_ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

pids=()
for seed in "${SEEDS[@]}"; do
  log="$LOG_ROOT/seed${seed}.log"
  "$PY" "$EXP/run.py" \
    --seed "$seed" \
    --steps "$STEPS" \
    --device "$DEVICE" \
    --output_root "$OUT_ROOT" \
    >"$log" 2>&1 &
  pids+=("$!")
  echo "[launch] seed=$seed pid=${pids[-1]} log=$log"
done

failed=0
for index in "${!pids[@]}"; do
  seed="${SEEDS[$index]}"
  if wait "${pids[$index]}"; then
    echo "[done] seed=$seed"
  else
    echo "[failed] seed=$seed" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 1
fi

"$PY" "$EXP/validate_results.py" \
  --output_root "$OUT_ROOT" \
  --steps "$STEPS" \
  --seeds "${SEEDS[@]}"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$ROOT/experiment_FPP/sac_her_per_style_loss_weighting_ls512"
PY="$ROOT/.venv/bin/python"
DEVICE="${DEVICE:-cpu}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiment_FPP/runs/sac_her_per_style_loss_weighting_ls512_correctness5k}"
LOG_ROOT="${LOG_ROOT:-$ROOT/experiment_FPP/logs/sac_her_per_style_loss_weighting_ls512_correctness5k}"

mkdir -p "$LOG_ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PY" "$EXP/run.py" \
  --seed 1 \
  --steps 5000 \
  --device "$DEVICE" \
  --output_root "$OUT_ROOT" \
  2>&1 | tee "$LOG_ROOT/seed1.log"

"$PY" "$EXP/validate_results.py" \
  --output_root "$OUT_ROOT" \
  --steps 5000 \
  --seeds 1

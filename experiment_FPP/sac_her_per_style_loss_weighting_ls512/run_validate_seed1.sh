#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$ROOT/experiment_FPP/sac_her_per_style_loss_weighting_ls512"
PY="$ROOT/.venv/bin/python"

export PYTHONPATH="$EXP/code/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PY" "$EXP/code/src/test_sac_per_style_integration.py"
"$PY" "$EXP/code/src/test_sac_per_style_runner.py"
"$PY" "$EXP/run.py" --seed 1 --steps 5000 --device cpu --validate_only

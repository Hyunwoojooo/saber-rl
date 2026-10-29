#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
RUNNER="$EXP_DIR/run.py"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

MODE="${MODE:-main2m}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
SEEDS="${SEEDS:-}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
VALIDATE_ONLY="${VALIDATE_ONLY:-0}"

args=(
  --mode "$MODE"
  --python "$PYTHON"
  --cuda-device "$CUDA_DEVICE"
  --max-parallel "$MAX_PARALLEL"
)
if [[ -n "$SEEDS" ]]; then
  read -r -a seed_values <<<"$SEEDS"
  args+=(--seeds "${seed_values[@]}")
fi
if [[ "$RESUME" == "1" ]]; then
  args+=(--resume)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi
if [[ "$VALIDATE_ONLY" == "1" ]]; then
  args+=(--validate-only)
fi

exec "$PYTHON" "$RUNNER" "${args[@]}" "$@"

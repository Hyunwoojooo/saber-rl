#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <td3_er|td3_annealing_er|td3_aer>" >&2
  exit 2
fi

METHOD="$1"
case "$METHOD" in
  td3_er|td3_annealing_er|td3_aer) ;;
  *) echo "unsupported method: $METHOD" >&2; exit 2 ;;
esac

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

pids=()
for seed in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
    "$PYTHON_BIN" "$EXPERIMENT_DIR/run.py" \
      --method "$METHOD" \
      --seed "$seed" \
      --steps 2000000 \
      --device cuda \
      --output-root metaworld_Drawer_Close_TD3_AER/results_2m &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"

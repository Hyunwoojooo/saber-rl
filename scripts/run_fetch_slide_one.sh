#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

METHOD="${METHOD:-b3_critic_only}"
SEED="${SEED:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
DEVICE="${DEVICE:-cuda}"
REWARD_TYPE="${REWARD_TYPE:-sparse}"
OUTPUT_ROOT="${OUTPUT_ROOT:-SABER/experiment/FetchSlide}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python3 SABER/src/saber_fetch_slide/experiment.py \
  --method "${METHOD}" \
  --seed "${SEED}" \
  --total_steps "${TOTAL_STEPS}" \
  --device "${DEVICE}" \
  --reward_type "${REWARD_TYPE}" \
  --output_root "${OUTPUT_ROOT}" \
  ${EXTRA_ARGS}

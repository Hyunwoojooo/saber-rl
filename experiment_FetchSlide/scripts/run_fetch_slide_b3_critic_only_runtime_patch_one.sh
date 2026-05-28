#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${WORKSPACE_ROOT}"

METHOD="${METHOD:-b3_critic_only}"
SEED="${SEED:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
DEVICE="${DEVICE:-cuda}"
REWARD_TYPE="${REWARD_TYPE:-sparse}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_ROOT}/experiment/FetchSlideRuntimePatch}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python3 "${EXPERIMENT_ROOT}/src/saber_fetch_slide/experiment_b3_critic_only_runtime_patch.py" \
  --method "${METHOD}" \
  --seed "${SEED}" \
  --total_steps "${TOTAL_STEPS}" \
  --device "${DEVICE}" \
  --reward_type "${REWARD_TYPE}" \
  --output_root "${OUTPUT_ROOT}" \
  ${EXTRA_ARGS}

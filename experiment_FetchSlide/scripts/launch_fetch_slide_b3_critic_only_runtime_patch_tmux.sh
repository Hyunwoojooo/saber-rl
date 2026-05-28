#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${WORKSPACE_ROOT}"

METHOD="${METHOD:-b3_critic_only}"
SEEDS="${SEEDS:-1 2 3 4 5}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
DEVICE="${DEVICE:-cuda}"
REWARD_TYPE="${REWARD_TYPE:-sparse}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_ROOT}/experiment/FetchSlideRuntimePatch}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

for seed in ${SEEDS}; do
  session="saber_fs_b3_patch_${METHOD}_s${seed}_${TOTAL_STEPS}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "skip existing session: ${session}"
    continue
  fi
  tmux new-session -d -s "${session}" -- bash -lc "
    cd '${WORKSPACE_ROOT}' &&
    METHOD='${METHOD}' \
    SEED='${seed}' \
    TOTAL_STEPS='${TOTAL_STEPS}' \
    DEVICE='${DEVICE}' \
    REWARD_TYPE='${REWARD_TYPE}' \
    OUTPUT_ROOT='${OUTPUT_ROOT}' \
    EXTRA_ARGS='${EXTRA_ARGS}' \
    bash '${EXPERIMENT_ROOT}/scripts/run_fetch_slide_b3_critic_only_runtime_patch_one.sh'
  "
  echo "started ${session}"
done

tmux ls

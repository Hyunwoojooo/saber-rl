#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${WORKSPACE_ROOT}"

METHODS="${METHODS:-b0 b3_original b3_critic_only b3_critic_only_random b3_critic_only_shuffled b3_critic_only_runtime_optimization b3_critic_only_fast_512 b3_critic_only_fast_2048}"
SEEDS="${SEEDS:-1 2 3 4 5}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
DEVICE="${DEVICE:-cuda}"
REWARD_TYPE="${REWARD_TYPE:-sparse}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_ROOT}/experiment/FetchSlide}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

for method in ${METHODS}; do
  for seed in ${SEEDS}; do
    session="saber_fs_${method}_s${seed}_${TOTAL_STEPS}"
    if tmux has-session -t "${session}" 2>/dev/null; then
      echo "skip existing session: ${session}"
      continue
    fi
    tmux new-session -d -s "${session}" -- bash -lc "
      cd '${WORKSPACE_ROOT}' &&
      METHOD='${method}' \
      SEED='${seed}' \
      TOTAL_STEPS='${TOTAL_STEPS}' \
      DEVICE='${DEVICE}' \
      REWARD_TYPE='${REWARD_TYPE}' \
      OUTPUT_ROOT='${OUTPUT_ROOT}' \
      EXTRA_ARGS='${EXTRA_ARGS}' \
      bash '${EXPERIMENT_ROOT}/scripts/run_fetch_slide_one.sh'
    "
    echo "started ${session}"
  done
done

tmux ls

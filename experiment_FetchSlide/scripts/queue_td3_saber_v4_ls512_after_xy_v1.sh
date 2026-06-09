#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

LOG_FILE="${LOG_FILE:-${EXPERIMENT_ROOT}/td3_saber_v4_ls512_after_xy_v1_queue.log}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-60}"
DEPENDENCY_SESSIONS="${DEPENDENCY_SESSIONS:-saber_fs_b3_critic_only_runtime_opt_v5_xy_v1_s1_1000000 saber_fs_b3_critic_only_runtime_opt_v5_xy_v1_s2_1000000 saber_fs_b3_critic_only_runtime_opt_v5_xy_v1_s3_1000000 saber_fs_b3_critic_only_runtime_opt_v5_xy_v1_s4_1000000 saber_fs_b3_critic_only_runtime_opt_v5_xy_v1_s5_1000000}"
METHODS_TO_LAUNCH="${METHODS_TO_LAUNCH:-td3_saber_critic_only_v4_ls512}"
SEEDS_TO_LAUNCH="${SEEDS_TO_LAUNCH:-1 2 3 4 5}"
TOTAL_STEPS_TO_LAUNCH="${TOTAL_STEPS_TO_LAUNCH:-1000000}"
DEVICE_TO_LAUNCH="${DEVICE_TO_LAUNCH:-cuda}"
REWARD_TYPE_TO_LAUNCH="${REWARD_TYPE_TO_LAUNCH:-sparse}"
EXTRA_ARGS_TO_LAUNCH="${EXTRA_ARGS_TO_LAUNCH:---local_search_max 512 --support_quality_mode none}"
PYTHON_BIN_TO_LAUNCH="${PYTHON_BIN_TO_LAUNCH:-${WORKSPACE_ROOT}/.venv/bin/python}"
SABER_MACHINE_LABEL_TO_LAUNCH="${SABER_MACHINE_LABEL_TO_LAUNCH:-edgexpert-d530}"

log() {
  printf "%s %s\n" "$(date "+%Y-%m-%d %H:%M:%S %Z")" "$*" >> "${LOG_FILE}"
}

log "queue started; waiting for dependency sessions: ${DEPENDENCY_SESSIONS}"
while true; do
  active=0
  for session in ${DEPENDENCY_SESSIONS}; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      active=1
      log "waiting; still active: ${session}"
      break
    fi
  done
  if [[ "${active}" -eq 0 ]]; then
    break
  fi
  sleep "${CHECK_INTERVAL_SEC}"
done

log "launching FetchSlide TD3+SABER main: methods=${METHODS_TO_LAUNCH} seeds=${SEEDS_TO_LAUNCH} steps=${TOTAL_STEPS_TO_LAUNCH} device=${DEVICE_TO_LAUNCH}"
cd "${WORKSPACE_ROOT}"
METHODS="${METHODS_TO_LAUNCH}" \
SEEDS="${SEEDS_TO_LAUNCH}" \
TOTAL_STEPS="${TOTAL_STEPS_TO_LAUNCH}" \
DEVICE="${DEVICE_TO_LAUNCH}" \
REWARD_TYPE="${REWARD_TYPE_TO_LAUNCH}" \
EXTRA_ARGS="${EXTRA_ARGS_TO_LAUNCH}" \
PYTHON_BIN="${PYTHON_BIN_TO_LAUNCH}" \
SABER_MACHINE_LABEL="${SABER_MACHINE_LABEL_TO_LAUNCH}" \
bash "${EXPERIMENT_ROOT}/scripts/launch_fetch_slide_matrix_tmux.sh" >> "${LOG_FILE}" 2>&1
log "queue finished"

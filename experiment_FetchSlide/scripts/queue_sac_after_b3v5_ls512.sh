#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${WORKSPACE_ROOT}"

LOG_FILE="${LOG_FILE:-${EXPERIMENT_ROOT}/sac_after_b3v5_ls512_queue.log}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-60}"
DEPENDENCY_METHOD="${DEPENDENCY_METHOD:-b3_critic_only_runtime_opt_v5}"
DEPENDENCY_SESSIONS="${DEPENDENCY_SESSIONS:-saber_fs_b3_critic_only_runtime_opt_v5_s1_1000000 saber_fs_b3_critic_only_runtime_opt_v5_s2_1000000 saber_fs_b3_critic_only_runtime_opt_v5_s3_1000000 saber_fs_b3_critic_only_runtime_opt_v5_s4_1000000 saber_fs_b3_critic_only_runtime_opt_v5_s5_1000000}"
DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-${EXPERIMENT_ROOT}/experiment/FetchSlide/${DEPENDENCY_METHOD}}"
DEPENDENCY_RUN_PREFIX="${DEPENDENCY_RUN_PREFIX:-fetch_slide_b3_critic_only_runtime_opt_v5}"
DEPENDENCY_STEPS="${DEPENDENCY_STEPS:-1000000}"
DEPENDENCY_LS="${DEPENDENCY_LS:-ls512}"
SEEDS_TO_CHECK="${SEEDS_TO_CHECK:-1 2 3 4 5}"
METHODS_TO_LAUNCH="${METHODS_TO_LAUNCH:-sac}"
SEEDS_TO_LAUNCH="${SEEDS_TO_LAUNCH:-1 2 3 4 5}"
TOTAL_STEPS_TO_LAUNCH="${TOTAL_STEPS_TO_LAUNCH:-1000000}"
DEVICE_TO_LAUNCH="${DEVICE_TO_LAUNCH:-cuda}"
PYTHON_BIN_TO_LAUNCH="${PYTHON_BIN_TO_LAUNCH:-/home/mimiclab/Hyunwoo/metacog-rl-v0-lite/.venv/bin/python}"
SABER_MACHINE_LABEL_TO_LAUNCH="${SABER_MACHINE_LABEL_TO_LAUNCH:-edgexpert-d530}"

log() {
  printf "%s %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" >> "${LOG_FILE}"
}

exact_summary_count() {
  local count=0
  local seed
  for seed in ${SEEDS_TO_CHECK}; do
    local summary
    for summary in "${DEPENDENCY_ROOT}/${DEPENDENCY_RUN_PREFIX}_s${seed}_${DEPENDENCY_STEPS}_${DEPENDENCY_LS}"*/summary.json; do
      if [[ -f "${summary}" ]]; then
        count=$((count + 1))
        break
      fi
    done
  done
  printf "%s\n" "${count}"
}

any_dependency_session_active() {
  local session
  for session in ${DEPENDENCY_SESSIONS}; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

log "queue started: wait for ${DEPENDENCY_METHOD} ${DEPENDENCY_LS}, then launch ${METHODS_TO_LAUNCH}"
while true; do
  active=0
  if any_dependency_session_active; then
    active=1
  fi
  count="$(exact_summary_count)"
  if [[ "${active}" -eq 0 && "${count}" -ge 5 ]]; then
    break
  fi
  log "waiting active=${active} exact_summaries=${count}/5"
  sleep "${CHECK_INTERVAL_SEC}"
done

log "launching ${METHODS_TO_LAUNCH}"
PYTHON_BIN="${PYTHON_BIN_TO_LAUNCH}" \
SABER_MACHINE_LABEL="${SABER_MACHINE_LABEL_TO_LAUNCH}" \
METHODS="${METHODS_TO_LAUNCH}" \
SEEDS="${SEEDS_TO_LAUNCH}" \
TOTAL_STEPS="${TOTAL_STEPS_TO_LAUNCH}" \
DEVICE="${DEVICE_TO_LAUNCH}" \
bash "${EXPERIMENT_ROOT}/scripts/launch_fetch_slide_matrix_tmux.sh" >> "${LOG_FILE}" 2>&1
log "queue finished"

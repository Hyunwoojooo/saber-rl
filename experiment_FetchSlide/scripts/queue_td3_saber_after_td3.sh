#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${WORKSPACE_ROOT}"

LOG_FILE="${LOG_FILE:-${EXPERIMENT_ROOT}/td3_saber_after_td3_queue.log}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-60}"
DEPENDENCY_SESSIONS="${DEPENDENCY_SESSIONS:-saber_fs_td3_s1_1000000 saber_fs_td3_s2_1000000 saber_fs_td3_s3_1000000 saber_fs_td3_s4_1000000 saber_fs_td3_s5_1000000}"
TD3_SABER_METHODS="${TD3_SABER_METHODS:-td3_saber}"

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" >> "${LOG_FILE}"
}

log "queue started"
while true; do
  active=0
  for session in ${DEPENDENCY_SESSIONS}; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      active=1
      break
    fi
  done
  if [[ "${active}" -eq 0 ]]; then
    break
  fi
  log "waiting for td3 baseline sessions"
  sleep "${CHECK_INTERVAL_SEC}"
done

log "launching td3_saber"
METHODS="${TD3_SABER_METHODS}" \
SEEDS="${SEEDS:-1 2 3 4 5}" \
TOTAL_STEPS="${TOTAL_STEPS:-1000000}" \
DEVICE="${DEVICE:-cuda}" \
bash "${EXPERIMENT_ROOT}/scripts/launch_fetch_slide_matrix_tmux.sh" >> "${LOG_FILE}" 2>&1
log "queue finished"

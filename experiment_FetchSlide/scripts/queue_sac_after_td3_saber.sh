#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${WORKSPACE_ROOT}"

LOG_FILE="${LOG_FILE:-${EXPERIMENT_ROOT}/sac_after_td3_saber_queue.log}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-60}"
TD3_SABER_SESSIONS="${TD3_SABER_SESSIONS:-saber_fs_td3_saber_s1_1000000 saber_fs_td3_saber_s2_1000000 saber_fs_td3_saber_s3_1000000 saber_fs_td3_saber_s4_1000000 saber_fs_td3_saber_s5_1000000}"
SAC_SESSIONS="${SAC_SESSIONS:-saber_fs_sac_s1_1000000 saber_fs_sac_s2_1000000 saber_fs_sac_s3_1000000 saber_fs_sac_s4_1000000 saber_fs_sac_s5_1000000}"
TD3_SABER_SUMMARY_GLOB="${TD3_SABER_SUMMARY_GLOB:-${EXPERIMENT_ROOT}/experiment/FetchSlide/td3_saber/fetch_slide_td3_saber_s*_1000000_ls512/summary.json}"
SAC_SUMMARY_GLOB="${SAC_SUMMARY_GLOB:-${EXPERIMENT_ROOT}/experiment/FetchSlide/sac/fetch_slide_sac_s*_1000000_lsnone/summary.json}"
SAC_METHODS="${SAC_METHODS:-sac}"
SAC_SABER_METHODS="${SAC_SABER_METHODS:-sac_saber}"

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" >> "${LOG_FILE}"
}

any_session_exists() {
  local session
  for session in "$@"; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

summary_count() {
  local pattern="$1"
  find $(dirname "${pattern}") -maxdepth 2 -type f -name summary.json 2>/dev/null | wc -l
}

wait_for_sessions_to_start_or_complete() {
  local summary_glob="$1"
  shift
  while true; do
    if any_session_exists "$@"; then
      return 0
    fi
    # shellcheck disable=SC2086
    if [[ "$(summary_count "${summary_glob}")" -ge 5 ]]; then
      return 0
    fi
    log "waiting for $* to start"
    sleep "${CHECK_INTERVAL_SEC}"
  done
}

wait_for_sessions_to_finish() {
  local summary_glob="$1"
  shift
  while true; do
    if any_session_exists "$@"; then
      log "waiting for $* to finish"
      sleep "${CHECK_INTERVAL_SEC}"
      continue
    fi
    # shellcheck disable=SC2086
    if [[ "$(summary_count "${summary_glob}")" -ge 5 ]]; then
      return 0
    fi
    log "sessions ended but waiting for summaries for $*"
    sleep "${CHECK_INTERVAL_SEC}"
  done
}

log "queue started"
wait_for_sessions_to_start_or_complete "${TD3_SABER_SUMMARY_GLOB}" ${TD3_SABER_SESSIONS}
wait_for_sessions_to_finish "${TD3_SABER_SUMMARY_GLOB}" ${TD3_SABER_SESSIONS}

log "launching sac"
METHODS="${SAC_METHODS}" \
SEEDS="${SEEDS:-1 2 3 4 5}" \
TOTAL_STEPS="${TOTAL_STEPS:-1000000}" \
DEVICE="${DEVICE:-cuda}" \
bash "${EXPERIMENT_ROOT}/scripts/launch_fetch_slide_matrix_tmux.sh" >> "${LOG_FILE}" 2>&1

wait_for_sessions_to_start_or_complete "${SAC_SUMMARY_GLOB}" ${SAC_SESSIONS}
wait_for_sessions_to_finish "${SAC_SUMMARY_GLOB}" ${SAC_SESSIONS}

log "launching sac_saber"
METHODS="${SAC_SABER_METHODS}" \
SEEDS="${SEEDS:-1 2 3 4 5}" \
TOTAL_STEPS="${TOTAL_STEPS:-1000000}" \
DEVICE="${DEVICE:-cuda}" \
bash "${EXPERIMENT_ROOT}/scripts/launch_fetch_slide_matrix_tmux.sh" >> "${LOG_FILE}" 2>&1
log "queue finished"

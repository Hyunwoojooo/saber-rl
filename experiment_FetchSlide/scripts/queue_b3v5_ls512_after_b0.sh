#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${WORKSPACE_ROOT}"

LOG_FILE="${LOG_FILE:-${EXPERIMENT_ROOT}/b3v5_ls512_after_b0_queue.log}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-60}"
B0_SESSIONS="${B0_SESSIONS:-saber_fs_b0_s1_1000000 saber_fs_b0_s2_1000000 saber_fs_b0_s3_1000000 saber_fs_b0_s4_1000000 saber_fs_b0_s5_1000000}"
B0_SUMMARY_ROOT="${B0_SUMMARY_ROOT:-${EXPERIMENT_ROOT}/experiment/FetchSlide/b0}"
METHODS_TO_LAUNCH="${METHODS_TO_LAUNCH:-b3_critic_only_runtime_opt_v5}"
SEEDS_TO_LAUNCH="${SEEDS_TO_LAUNCH:-1 2 3 4 5}"
TOTAL_STEPS_TO_LAUNCH="${TOTAL_STEPS_TO_LAUNCH:-1000000}"
DEVICE_TO_LAUNCH="${DEVICE_TO_LAUNCH:-cuda}"
PYTHON_BIN_TO_LAUNCH="${PYTHON_BIN_TO_LAUNCH:-/home/mimiclab/Hyunwoo/metacog-rl-v0-lite/.venv/bin/python}"
SABER_MACHINE_LABEL_TO_LAUNCH="${SABER_MACHINE_LABEL_TO_LAUNCH:-edgexpert-d530}"
EXTRA_ARGS_TO_LAUNCH="${EXTRA_ARGS_TO_LAUNCH:---local_search_max 512 --support_quality_mode none}"

log() {
  printf "%s %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" >> "${LOG_FILE}"
}

summary_count() {
  find "${B0_SUMMARY_ROOT}" -maxdepth 2 -type f -name summary.json 2>/dev/null | wc -l
}

any_b0_session_active() {
  local session
  for session in ${B0_SESSIONS}; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

log "queue started: wait for B0 sessions and 5 summaries"
while true; do
  active=0
  if any_b0_session_active; then
    active=1
  fi
  count="$(summary_count)"
  if [[ "${active}" -eq 0 && "${count}" -ge 5 ]]; then
    break
  fi
  log "waiting active=${active} summaries=${count}/5"
  sleep "${CHECK_INTERVAL_SEC}"
done

log "launching ${METHODS_TO_LAUNCH} ls512 no-quality"
PYTHON_BIN="${PYTHON_BIN_TO_LAUNCH}" \
SABER_MACHINE_LABEL="${SABER_MACHINE_LABEL_TO_LAUNCH}" \
METHODS="${METHODS_TO_LAUNCH}" \
SEEDS="${SEEDS_TO_LAUNCH}" \
TOTAL_STEPS="${TOTAL_STEPS_TO_LAUNCH}" \
DEVICE="${DEVICE_TO_LAUNCH}" \
EXTRA_ARGS="${EXTRA_ARGS_TO_LAUNCH}" \
bash "${EXPERIMENT_ROOT}/scripts/launch_fetch_slide_matrix_tmux.sh" >> "${LOG_FILE}" 2>&1
log "queue finished"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
WAIT_SESSION="${WAIT_SESSION:-fpp-td3-aer-main1m-s1-5}"
POLL_SECONDS="${POLL_SECONDS:-60}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_FPP/runs/fpp_td3_aer_noher_1m}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/fpp_td3_aer_noher_1m}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/fpp_td3_er_main_after_aer_queue}"
QUEUE_LOG="$QUEUE_LOG_DIR/queue.log"
SOURCE_MANIFEST="$QUEUE_LOG_DIR/frozen_source.sha256"
SEED_LIST="1 2 3 4 5"

if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  exit 2
fi

mkdir -p "$QUEUE_LOG_DIR"
if [[ -e "$SOURCE_MANIFEST" || -e "$QUEUE_LOG" ]]; then
  echo "refusing to overwrite existing queue state in $QUEUE_LOG_DIR" >&2
  exit 1
fi

log() {
  printf '%s | %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$QUEUE_LOG"
}

run_logged() {
  "$@" 2>&1 | tee -a "$QUEUE_LOG"
}

write_source_manifest() {
  (
    cd "$SCRIPT_DIR"
    find config.json run.py code/src \
      run_main_1m_td3_er_seeds1_5_5parallel.sh \
      queue_td3_er_main_after_td3_aer_s1_5.sh \
      validate_main_arm_gate.py \
      -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
      | sort -z \
      | xargs -0 sha256sum
  ) >"$SOURCE_MANIFEST"
}

verify_source_manifest() {
  (
    cd "$SCRIPT_DIR"
    sha256sum --check "$SOURCE_MANIFEST"
  )
}

write_source_manifest
log "queue armed; source manifest frozen at $SOURCE_MANIFEST"
log "waiting for TD3-AER tmux session '$WAIT_SESSION'"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

log "TD3-AER session ended; validating all five 1M AER runs"
run_logged "$PYTHON" "$SCRIPT_DIR/validate_main_arm_gate.py" \
  --arm td3_aer \
  --seeds "$SEED_LIST" \
  --output-root "$OUTPUT_ROOT" \
  --log-dir "$LOG_DIR"

log "AER completion gate passed; checking frozen source hashes"
run_logged verify_source_manifest

log "source is unchanged; validating frozen TD3-ER 1M command"
run_logged "$PYTHON" "$SCRIPT_DIR/run.py" \
  --arm td3_er \
  --seed 1 \
  --steps 1000000 \
  --device cpu \
  --validate_only

log "all gates passed; starting five parallel TD3-ER 1M runs for seeds 1-5"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PYTHON="$PYTHON" \
DEVICE="$DEVICE" \
SEED_LIST="$SEED_LIST" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
LOG_DIR="$LOG_DIR" \
  bash "$SCRIPT_DIR/run_main_1m_td3_er_seeds1_5_5parallel.sh" 2>&1 | tee -a "$QUEUE_LOG"

log "TD3-ER 1M main completed successfully for seeds 1-5"

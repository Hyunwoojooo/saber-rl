#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
WAIT_SESSION="${WAIT_SESSION:-fpp-td3-hgr-main1m-after-annealing-er-s1-5}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PREDECESSOR_DIR="$REPO_ROOT/experiment_FPP/td3_hgr_tdpriority_1m"
PREDECESSOR_OUTPUT_ROOT="${PREDECESSOR_OUTPUT_ROOT:-$REPO_ROOT/experiment_FPP/runs/td3_hgr_tdpriority_1m}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_FPP/runs/sac_hgr_tdpriority_1m}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/experiment_FPP/logs/sac_hgr_tdpriority_1m}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/sac_hgr_main_after_td3_hgr_queue}"
QUEUE_LOG="$QUEUE_LOG_DIR/queue.log"
SOURCE_MANIFEST="$QUEUE_LOG_DIR/frozen_source.sha256"
SEEDS=(1 2 3 4 5)

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
    find config.json run.py validate_results.py code/src \
      run_validate_seed1.sh \
      run_main_1m_seeds1_5_5parallel.sh \
      queue_main_1m_after_td3_hgr_s1_5.sh \
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

assert_targets_absent() {
  local seed target
  for seed in "${SEEDS[@]}"; do
    target="$OUTPUT_ROOT/fetch_pick_and_place_sac_hgr_tdpriority_s${seed}_1000000"
    if [[ -e "$target" ]]; then
      echo "refusing to overwrite existing SAC-HGR run: $target" >&2
      return 1
    fi
  done
}

write_source_manifest
log "queue armed; source manifest frozen at $SOURCE_MANIFEST"
log "waiting for TD3+HER+HGR tmux session '$WAIT_SESSION'"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

log "TD3-HGR session ended; running strict validation on all five predecessor runs"
run_logged "$PYTHON" "$PREDECESSOR_DIR/validate_results.py" \
  --output_root "$PREDECESSOR_OUTPUT_ROOT" \
  --steps 1000000 \
  --seeds "${SEEDS[@]}"

log "predecessor gate passed; checking frozen SAC-HGR source hashes"
run_logged verify_source_manifest
run_logged assert_targets_absent

log "source is unchanged and targets are clear; validating frozen SAC-HGR stack"
run_logged bash "$SCRIPT_DIR/run_validate_seed1.sh"

log "all gates passed; starting five parallel SAC+HER+HGR 1M runs for seeds 1-5"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

DEVICE="$DEVICE" \
OUT_ROOT="$OUTPUT_ROOT" \
LOG_ROOT="$LOG_ROOT" \
  bash "$SCRIPT_DIR/run_main_1m_seeds1_5_5parallel.sh" 2>&1 | tee -a "$QUEUE_LOG"

log "SAC+HER+HGR 1M main completed and strict result validation passed for seeds 1-5"

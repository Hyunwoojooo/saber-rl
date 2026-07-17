#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
WAIT_SESSION="${WAIT_SESSION:-fpp-ddpg-aer-noher-smoke20k-s1-5}"
POLL_SECONDS="${POLL_SECONDS:-30}"
SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-$REPO_ROOT/experiment_FPP/runs/fpp_ddpg_aer_noher_smoke20k_s1_5}"
SMOKE_LOG_DIR="${SMOKE_LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/fpp_ddpg_aer_noher_smoke20k_s1_5}"
MAIN_OUTPUT_ROOT="${MAIN_OUTPUT_ROOT:-$REPO_ROOT/experiment_FPP/runs/fpp_ddpg_aer_noher_1m}"
MAIN_LOG_DIR="${MAIN_LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/fpp_ddpg_aer_noher_1m}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/fpp_ddpg_aer_noher_main_after_smoke_queue}"
QUEUE_LOG="$QUEUE_LOG_DIR/queue.log"
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

log() {
  printf '%s | %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$QUEUE_LOG"
}

run_logged() {
  "$@" 2>&1 | tee -a "$QUEUE_LOG"
}

log "queue armed; waiting for smoke tmux session '$WAIT_SESSION'"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

log "smoke session ended; running strict 15-run completion gate"
run_logged "$PYTHON" "$SCRIPT_DIR/validate_smoke_gate.py" \
  --output-root "$SMOKE_OUTPUT_ROOT" \
  --log-dir "$SMOKE_LOG_DIR"

log "artifact gate passed; rerunning AER sampler regression test"
run_logged "$PYTHON" -B "$SCRIPT_DIR/code/src/test_aer_sampling.py"

log "regression test passed; validating the frozen DDPG-AER 1M command"
run_logged "$PYTHON" "$SCRIPT_DIR/run.py" \
  --arm ddpg_aer \
  --seed 1 \
  --steps 1000000 \
  --device cpu \
  --validate_only

log "all smoke gates passed; starting five parallel DDPG-AER 1M runs for seeds 1-5"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PYTHON="$PYTHON" \
DEVICE="$DEVICE" \
SEED_LIST="$SEED_LIST" \
OUTPUT_ROOT="$MAIN_OUTPUT_ROOT" \
LOG_DIR="$MAIN_LOG_DIR" \
  bash "$SCRIPT_DIR/run_main_1m_ddpg_aer_seeds1_5_5parallel.sh" 2>&1 | tee -a "$QUEUE_LOG"

log "DDPG-AER 1M main completed successfully for seeds 1-5"

#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mimiclab_asus/SABER"
WAIT_PID="${1:-73497}"
RUN_SCRIPT="$ROOT/FetchPush_SAC/run_all_sac_fetchpush_5parallel.sh"
LOG_DIR="$ROOT/FetchPush_SAC/orchestrator_logs"
mkdir -p "$LOG_DIR"
QUEUE_LOG="$LOG_DIR/queue_all_sac_after_current_td3_fppstyle_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$QUEUE_LOG"
}

log "QUEUE armed: waiting for current TD3 FPP-style orchestrator pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
log "Current TD3 FPP-style orchestrator pid=$WAIT_PID ended; launching SAC sequence"
exec bash "$RUN_SCRIPT"

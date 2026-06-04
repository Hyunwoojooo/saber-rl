#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mimiclab_asus/SABER"
WAIT_PID="${1:-74627}"
RUN_SCRIPT="$ROOT/FetchPush_DDPG/run_all_ddpg_fetchpush_5parallel.sh"
LOG_DIR="$ROOT/FetchPush_DDPG/orchestrator_logs"
mkdir -p "$LOG_DIR"
QUEUE_LOG="$LOG_DIR/queue_all_ddpg_after_sac_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$QUEUE_LOG"
}

log "QUEUE armed: waiting for SAC orchestrator/queue pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
log "SAC orchestrator/queue pid=$WAIT_PID ended; launching DDPG sequence"
exec bash "$RUN_SCRIPT"

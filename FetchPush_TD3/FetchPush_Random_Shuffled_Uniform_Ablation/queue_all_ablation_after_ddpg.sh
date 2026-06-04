#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mimiclab_asus/SABER"
WAIT_PID="${1:-}"
RUNNER="$ROOT/FetchPush_TD3/FetchPush_TD3/FetchPush_Random_Shuffled_Uniform_Ablation/run_all_td3_negative_controls_5parallel.sh"
LOG_DIR="$ROOT/FetchPush_TD3/FetchPush_Random_Shuffled_Uniform_Ablation/orchestrator_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/queue_all_ablation_after_ddpg_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$LOG"
}

if [[ -z "$WAIT_PID" ]]; then
  log "ERROR: missing DDPG orchestrator/queue pid"
  exit 2
fi

log "QUEUE armed: waiting for DDPG orchestrator/queue pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
log "DDPG orchestrator/queue pid=$WAIT_PID finished; starting TD3 negative-control ablations"
exec bash "$RUNNER"

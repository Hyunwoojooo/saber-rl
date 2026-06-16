#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mimiclab_asus/SABER"
LOG_DIR="$ROOT/FetchPush_TD3/orchestrator_logs"
mkdir -p "$LOG_DIR"
ORCH_LOG="$LOG_DIR/run_b3_actor_scope_td3_5parallel_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$ORCH_LOG"
}

log "BEGIN FetchPush TD3+HER B3 actor-scope sequence: Actor-Only -> Actor+Critic"
log "Actor-Only starts immediately; Actor+Critic starts only after Actor-Only completes successfully."

bash "$ROOT/FetchPush_TD3/run_b3_actor_only_td3_5parallel.sh" 2>&1 | tee -a "$ORCH_LOG"
log "END Actor-Only; launching Actor+Critic"

bash "$ROOT/FetchPush_TD3/run_b3_actor_critic_td3_5parallel.sh" 2>&1 | tee -a "$ORCH_LOG"
log "ALL FetchPush TD3+HER B3 actor-scope variants completed successfully"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH_LOG_DIR="$ROOT/orchestrator_logs"
ORCH_LOG="$ORCH_LOG_DIR/run_ddpg_drawer_open_actor_ablation_2m_5parallel_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ORCH_LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$ORCH_LOG"
}

log "BEGIN MetaWorld DrawerOpen DDPG actor-scope ablation sequence: Actor-Only -> Actor+Critic"
log "Running DDPG-HER-B3-Actor-Only-2M"
bash "$ROOT/DDPG-HER-B3-Actor-Only-2M.sh"
log "Running DDPG-HER-B3-Actor-Critic-2M"
bash "$ROOT/DDPG-HER-B3-Actor-Critic-2M.sh"
log "ALL MetaWorld DrawerOpen DDPG actor-scope ablation variants completed successfully"

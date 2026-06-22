#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH_LOG_DIR="$ROOT/orchestrator_logs"
ORCH_LOG="$ORCH_LOG_DIR/run_ddpg_drawer_open_ablation_2m_5parallel_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ORCH_LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$ORCH_LOG"
}

log "BEGIN MetaWorld DrawerOpen DDPG ablation sequence: Random-Fixed-Count -> Shuffled-Weight"
log "Running DDPG-HER-B3-Random-Fixed-Count-2M"
bash "$ROOT/DDPG-HER-B3-Random-Fixed-Count-2M.sh"
log "Running DDPG-HER-B3-Shuffled-Weight-2M"
bash "$ROOT/DDPG-HER-B3-Shuffled-Weight-2M.sh"
log "ALL MetaWorld DrawerOpen DDPG ablation variants completed successfully"

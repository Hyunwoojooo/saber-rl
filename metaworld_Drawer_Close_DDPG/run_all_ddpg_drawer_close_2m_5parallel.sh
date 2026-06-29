#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/orchestrator_logs"
mkdir -p "$LOG_DIR"
ORCH_LOG="$LOG_DIR/run_all_ddpg_drawer_close_2m_5parallel_$(date +%Y%m%d_%H%M%S).log"

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$ORCH_LOG"
}

scripts=(
  "DDPG-HER-B0-2M.sh"
  "DDPG-HER-B3-Critic-Only-2M.sh"
  "DDPG-HER-B3-Random-Fixed-Count-2M.sh"
  "DDPG-HER-B3-Random-Support-2M.sh"
  "DDPG-HER-B3-Shuffled-Weight-2M.sh"
  "DDPG-HER-PER-Style-2M.sh"
)

log "BEGIN DrawerClose DDPG full 2M sequence"
for script in "${scripts[@]}"; do
  log "BEGIN $script"
  CUDA_DEVICE="${CUDA_DEVICE:-0}" bash "$ROOT/$script"
  log "END $script"
done
log "DrawerClose DDPG full 2M sequence completed"

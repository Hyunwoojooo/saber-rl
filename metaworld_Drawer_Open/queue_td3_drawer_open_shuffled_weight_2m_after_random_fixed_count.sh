#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
CURRENT_SESSION="${CURRENT_SESSION:-drawer_random_fixed_count_2m}"
RUNNER="$ROOT/run_td3_drawer_open_shuffled_weight_2m_5parallel.sh"
ORCH_LOG_DIR="$ROOT/orchestrator_logs"
QUEUE_LOG="$ORCH_LOG_DIR/queue_td3_drawer_open_shuffled_weight_2m_after_random_fixed_count_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ORCH_LOG_DIR"

log() {
  printf "[%s] %s\n" "$(date +%Y-%m-%d_%H:%M:%S_%Z)" "$*" | tee -a "$QUEUE_LOG"
}

log "QUEUE BEGIN: wait for tmux session $CURRENT_SESSION, then run DrawerOpen TD3 Shuffled-Weight 2M."
while tmux has-session -t "=$CURRENT_SESSION" 2>/dev/null; do
  log "WAIT: tmux session $CURRENT_SESSION is still active."
  sleep 60
done

log "START: current DrawerOpen session finished; launching Shuffled-Weight 2M sequence."
cd "$REPO_ROOT"
bash "$RUNNER"
log "QUEUE END: DrawerOpen TD3 Shuffled-Weight 2M sequence finished."

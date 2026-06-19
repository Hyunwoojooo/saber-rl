#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
CURRENT_SESSION="${CURRENT_SESSION:-drawer_sac_b0_2m}"
TD3_RATECAP_RUNNER="$ROOT/TD3-HER-B3-Critic-Only-SupportRateCap-04/run_td3_drawer_open_b3_support_rate_cap_04_2m_5parallel.sh"
SAC_REMAINING_RUNNER="$REPO_ROOT/metaworld_Drawer_Open_SAC/run_sac_drawer_open_remaining_after_b0_2m_5parallel.sh"
ORCH_LOG_DIR="$ROOT/orchestrator_logs"
QUEUE_LOG="$ORCH_LOG_DIR/queue_td3_ratecap_then_sac_remaining_after_sac_b0_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ORCH_LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$QUEUE_LOG"
}

log "QUEUE BEGIN: wait for tmux session '$CURRENT_SESSION', then run TD3 SupportRateCap-0.4 and SAC remaining variants."
while tmux has-session -t "=$CURRENT_SESSION" 2>/dev/null; do
  log "WAIT: tmux session '$CURRENT_SESSION' is still active."
  sleep 60
done

log "START: '$CURRENT_SESSION' finished; launching TD3 SupportRateCap-0.4 2M."
cd "$REPO_ROOT"
bash "$TD3_RATECAP_RUNNER"
log "END: TD3 SupportRateCap-0.4 2M finished."

log "START: launching SAC remaining 2M sequence: B3-Critic-Only -> B3-Actor-Only -> B3-Actor+Critic -> PER-Full."
bash "$SAC_REMAINING_RUNNER"
log "QUEUE END: TD3 SupportRateCap-0.4 and SAC remaining sequence finished."

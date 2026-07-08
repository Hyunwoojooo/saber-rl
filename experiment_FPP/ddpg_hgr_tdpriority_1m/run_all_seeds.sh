#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[compat] forwarding to run_fpp_ddpg_hgr_tdpriority_1m_5parallel.sh" >&2
exec "$SCRIPT_DIR/run_fpp_ddpg_hgr_tdpriority_1m_5parallel.sh" "$@"

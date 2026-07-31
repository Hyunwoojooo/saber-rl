#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[compat] This folder is TD3 HER high-RR x4; forwarding to run_fpp_td3_her_high_rr_x4_1m_5parallel.sh" >&2
exec "$SCRIPT_DIR/run_fpp_td3_her_high_rr_x4_1m_5parallel.sh" "$@"

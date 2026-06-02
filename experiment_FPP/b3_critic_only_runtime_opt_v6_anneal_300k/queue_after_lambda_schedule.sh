#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_SESSION="${CURRENT_SESSION:-fpp_ls512_lambda_schedule_1m}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
SEED_LIST="${SEED_LIST:-1 2 3 4 5}"
STEPS="${STEPS:-1000000}"
DEVICE="${DEVICE:-cuda}"

echo "[queue] waiting for tmux session ${CURRENT_SESSION} to finish"
while tmux ls 2>/dev/null | grep -q "^${CURRENT_SESSION}:"; do
  date "+[queue] %F %T %Z still waiting for ${CURRENT_SESSION}"
  sleep "$SLEEP_SECONDS"
done

echo "[queue] previous session finished; launching b3_critic_only_runtime_opt_v6_anneal_300k"
SEED_LIST="$SEED_LIST" STEPS="$STEPS" DEVICE="$DEVICE" bash "$SCRIPT_DIR/run_screening_3seeds_parallel.sh"

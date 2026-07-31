#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SEED="${SEED:-1}"
STEPS="${STEPS:-50000}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_FPP/runs/td3_her_high_replay_ratio_x4_smoke}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/td3_her_high_replay_ratio_x4_smoke}"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
log_file="$LOG_DIR/seed${SEED}.log"
echo "[launch] method=td3_her_high_rr_x4 smoke seed=$SEED steps=$STEPS updates_per_step=4 device=$DEVICE output_root=$OUTPUT_ROOT log=$log_file"
python3 "$SCRIPT_DIR/run.py" \
  --seed "$SEED" \
  --steps "$STEPS" \
  --device "$DEVICE" \
  --output_root "$OUTPUT_ROOT" \
  > "$log_file" 2>&1

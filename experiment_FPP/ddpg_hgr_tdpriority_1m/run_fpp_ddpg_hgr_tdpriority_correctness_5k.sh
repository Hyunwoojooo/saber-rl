#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SEED="${SEED:-1}"
STEPS="${STEPS:-5000}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_FPP/runs/ddpg_hgr_tdpriority_original_pair_is_correctness5k}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/ddpg_hgr_tdpriority_original_pair_is_correctness5k}"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
log_file="$LOG_DIR/seed${SEED}.log"
echo "[launch] method=hgr correctness seed=$SEED steps=$STEPS device=$DEVICE output_root=$OUTPUT_ROOT log=$log_file"
python3 "$SCRIPT_DIR/run.py" --seed "$SEED" --steps "$STEPS" --device "$DEVICE" --output_root "$OUTPUT_ROOT" > "$log_file" 2>&1

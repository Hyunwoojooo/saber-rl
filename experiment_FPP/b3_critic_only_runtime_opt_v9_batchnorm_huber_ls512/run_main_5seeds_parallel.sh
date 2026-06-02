#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
SEED_LIST="${SEED_LIST:-1 2 3 4 5}"
STEPS="${STEPS:-1000000}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/SABER/FetchPickAndPlace/runs/b3_critic_only_runtime_opt_v9_batchnorm_huber_ls512}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/SABER/FetchPickAndPlace/logs/v9_batchnorm_huber_ls512_1m}"

mkdir -p "$LOG_DIR"

for seed in $SEED_LIST; do
  log_file="$LOG_DIR/seed${seed}.log"
  echo "[launch] seed=$seed steps=$STEPS device=$DEVICE output_root=$OUTPUT_ROOT log=$log_file"
  python3 "$SCRIPT_DIR/run.py" \
    --seed "$seed" \
    --steps "$STEPS" \
    --device "$DEVICE" \
    --support_quality_mode none \
    --critic_td_loss huber \
    --saber_loss_mode batch_norm \
    --support_bonus_beta 1.0 \
    --output_root "$OUTPUT_ROOT" \
    > "$log_file" 2>&1 &
done

wait

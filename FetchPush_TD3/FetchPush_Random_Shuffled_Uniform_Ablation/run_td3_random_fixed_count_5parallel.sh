#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mimiclab_asus/SABER"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
ABL_ROOT="$ROOT/FetchPush_TD3/FetchPush_Random_Shuffled_Uniform_Ablation"
EXP_DIR="$ABL_ROOT/TD3-HER-B3-Critic-Only-Random-Fixed-Count"
TRAIN="$EXP_DIR/code/src/train_fetch_push_lite_v0.py"
ORCH_LOG_DIR="$ABL_ROOT/orchestrator_logs"
mkdir -p "$ORCH_LOG_DIR"
ORCH_LOG="$ORCH_LOG_DIR/run_td3_random_fixed_count_5parallel_$(date +%Y%m%d_%H%M%S).log"

STEPS="${STEPS:-1000000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RANDOM_FIXED_COUNT_K="${RANDOM_FIXED_COUNT_K:-60}"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$ORCH_LOG"
}

common_args() {
  local seed="$1"
  printf "%s\0" \
    --reward_type sparse \
    --total_steps "$STEPS" \
    --start_steps 10000 \
    --warmup_steps 5000 \
    --updates_per_step 1 \
    --batch_size 256 \
    --buffer_capacity 300000 \
    --her_ratio 0.8 \
    --gamma 0.98 \
    --tau 0.005 \
    --actor_lr 0.0003 \
    --critic_lr 0.0003 \
    --hidden_dim 256 \
    --noise_std 0.2 \
    --policy_delay 2 \
    --target_policy_noise 0.2 \
    --target_noise_clip 0.5 \
    --eval_freq 10000 \
    --eval_episodes 20 \
    --checkpoint_every_steps 50000 \
    --log_every_steps 1000 \
    --device cuda \
    --env_seed "$seed" \
    --torch_seed "$seed" \
    --numpy_seed "$seed" \
    --action_noise_seed "$seed" \
    --weight_seed "$((20000 + seed))" \
    --eval_seed "$((10000 + seed))" \
    --no_save_step_checkpoints \
    --lambda_support 1.0 \
    --random_fixed_count_k "$RANDOM_FIXED_COUNT_K" \
    --weighting_mode batch_norm \
    --weight_scope critic_only
}

run_variant() {
  local label="TD3+HER+B3-Fully-Random-Fixed-Count"
  local run_prefix="td3_b3_fully_random_fixed_count"
  local output_root="$EXP_DIR/results_${run_prefix}"
  local log_dir="$EXP_DIR/logs_${run_prefix}"
  mkdir -p "$output_root" "$log_dir"

  log "BEGIN $label: method=b3_fully_random_fixed_count seeds=1..5 k=$RANDOM_FIXED_COUNT_K output_root=$output_root"
  log "Fully random fixed-count control: no SABER support lookup, no support-positive count matching"

  local pids=()
  for seed in 1 2 3 4 5; do
    local run_name="${run_prefix}_k${RANDOM_FIXED_COUNT_K}_s${seed}_1m_cuda"
    local args=()
    while IFS= read -r -d "" arg; do
      args+=("$arg")
    done < <(common_args "$seed")
    log "START $label seed=$seed run=$run_name"
    (
      cd "$EXP_DIR"
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" "$TRAIN" \
        --run_name "$run_name" \
        --method b3_fully_random_fixed_count \
        --output_root "$output_root" \
        "${args[@]}" \
        > "$log_dir/$run_name.out.log" \
        2> "$log_dir/$run_name.err.log"
    ) &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if wait "$pid"; then
      log "DONE $label pid=$pid"
    else
      local code=$?
      log "FAILED $label pid=$pid exit=$code"
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    log "STOP after failure in $label"
    exit 1
  fi
  log "ALL $label seeds completed successfully"
}

log "BEGIN FetchPush TD3+HER fully random fixed-count ablation"
log "Python=$PYTHON CUDA_VISIBLE_DEVICES=$CUDA_DEVICE k=$RANDOM_FIXED_COUNT_K"
run_variant
log "ALL FetchPush TD3+HER fully random fixed-count ablation completed successfully"

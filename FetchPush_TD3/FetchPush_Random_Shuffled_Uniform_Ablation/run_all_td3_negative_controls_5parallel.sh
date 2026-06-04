#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mimiclab_asus/SABER"
PYTHON="$ROOT/.venv/bin/python"
ABL_ROOT="$ROOT/FetchPush_TD3/FetchPush_Random_Shuffled_Uniform_Ablation"
ORCH_LOG_DIR="$ABL_ROOT/orchestrator_logs"
mkdir -p "$ORCH_LOG_DIR"
ORCH_LOG="$ORCH_LOG_DIR/run_all_td3_negative_controls_5parallel_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$ORCH_LOG"
}

common_args() {
  local seed="$1"
  printf "%s\0" \
    --reward_type sparse \
    --total_steps 1000000 \
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
    --M_B 16 \
    --M_Q_min 4 \
    --N_local_min 4 \
    --N_gap_min 128 \
    --lambda_support 1.0 \
    --weighting_mode batch_norm \
    --weight_scope critic_only \
    --local_search_max 1024 \
    --policy_support_mode bin \
    --support_lookup_optimization \
    --support_scan_mode sticky_random_v1 \
    --support_scan_seed 512 \
    --support_cache_max_entries 500000 \
    --support_local_scan_cache_max_entries 200000 \
    --support_numba_scan_enabled \
    --no_support_batch_query_coalescing
}

run_variant() {
  local label="$1"
  local folder="$2"
  local run_prefix="$3"
  local method="$4"

  local exp_dir="$ABL_ROOT/$folder"
  local train="$exp_dir/code/src/train_fetch_push_lite_v0.py"
  local output_root="$exp_dir/results_${run_prefix}"
  local log_dir="$exp_dir/logs_${run_prefix}"
  mkdir -p "$output_root" "$log_dir"

  log "BEGIN $label: method=$method, seeds=1..5, output_root=$output_root"
  log "TD3+HER negative control; FPP B3 params: M_B=16 M_Q_min=4 N_local_min=4 N_gap_min=128 local_search_max=1024 weight_scope=critic_only"

  local pids=()
  for seed in 1 2 3 4 5; do
    local run_name="${run_prefix}_s${seed}_1m_cuda"
    local args=()
    while IFS= read -r -d "" arg; do
      args+=("$arg")
    done < <(common_args "$seed")
    log "START $label seed=$seed run=$run_name"
    (
      cd "$exp_dir"
      CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$train" \
        --run_name "$run_name" \
        --method "$method" \
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

log "BEGIN FetchPush TD3+HER Random/Shuffled/Uniform negative-control ablations"
log "Python=$PYTHON"
log "CUDA_VISIBLE_DEVICES=0, sequential variants, 5 parallel seeds per variant"

run_variant \
  "TD3+HER+B3-Critic-Only-Random-Support" \
  "TD3-HER-B3-Critic-Only-Random-Support" \
  "td3_b3_critic_only_random_support" \
  "b3_support_random"

run_variant \
  "TD3+HER+B3-Critic-Only-Shuffled-Weight" \
  "TD3-HER-B3-Critic-Only-Shuffled-Weight" \
  "td3_b3_critic_only_shuffled_weight" \
  "b3_support_shuffled"

run_variant \
  "TD3+HER+B3-Critic-Only-Uniform-Support" \
  "TD3-HER-B3-Critic-Only-Uniform-Support" \
  "td3_b3_critic_only_uniform_support" \
  "b3_support_uniform"

log "ALL FetchPush TD3+HER negative-control ablations completed successfully"

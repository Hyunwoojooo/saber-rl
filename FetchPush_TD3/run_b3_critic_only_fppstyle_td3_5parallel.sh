#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mimiclab_asus/SABER"
PYTHON="$ROOT/.venv/bin/python"
EXP_DIR="$ROOT/FetchPush_TD3/TD3-HER-B3-Critic-Only"
TRAIN="$EXP_DIR/code/src/train_fetch_push_lite_v0.py"
OUTPUT_ROOT="$EXP_DIR/results_fppstyle_td3"
LOG_DIR="$EXP_DIR/logs_fppstyle_td3"
ORCH_LOG_DIR="$ROOT/FetchPush_TD3/orchestrator_logs"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$ORCH_LOG_DIR"
ORCH_LOG="$ORCH_LOG_DIR/run_b3_critic_only_fppstyle_td3_5parallel_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$ORCH_LOG"
}

common_args() {
  local seed="$1"
  printf "%s\0"     --reward_type sparse     --total_steps 1000000     --start_steps 10000     --warmup_steps 5000     --updates_per_step 1     --batch_size 256     --buffer_capacity 300000     --her_ratio 0.8     --gamma 0.98     --tau 0.005     --actor_lr 0.0003     --critic_lr 0.0003     --hidden_dim 256     --noise_std 0.2     --policy_delay 2     --target_policy_noise 0.2     --target_noise_clip 0.5     --eval_freq 10000     --eval_episodes 20     --checkpoint_every_steps 50000     --log_every_steps 1000     --device cuda     --env_seed "$seed"     --torch_seed "$seed"     --numpy_seed "$seed"     --action_noise_seed "$seed"     --weight_seed "$((20000 + seed))"     --eval_seed "$((10000 + seed))"     --no_save_step_checkpoints     --M_B 16     --M_Q_min 4     --N_local_min 4     --N_gap_min 128     --lambda_support 1.0     --weighting_mode batch_norm     --weight_scope critic_only     --local_search_max 1024     --policy_support_mode bin     --support_lookup_optimization     --support_scan_mode sticky_random_v1     --support_scan_seed 512     --support_cache_max_entries 500000     --support_local_scan_cache_max_entries 200000     --support_numba_scan_enabled     --no_support_batch_query_coalescing
}

log "BEGIN FetchPush FPP-style TD3+HER B3-Critic-Only 5-parallel"
log "Python=$PYTHON"
log "CUDA_VISIBLE_DEVICES=0, seeds=1..5, output_root=$OUTPUT_ROOT"
log "TD3 learner retained; FPP B3 support params: M_B=16 M_Q_min=4 N_local_min=4 N_gap_min=128 local_search_max=1024"

pids=()
for seed in 1 2 3 4 5; do
  run_name="fppstyle_td3_b3_critic_only_s${seed}_1m_cuda"
  args=()
  while IFS= read -r -d "" arg; do
    args+=("$arg")
  done < <(common_args "$seed")
  log "START seed=$seed run=$run_name"
  (
    cd "$EXP_DIR"
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$TRAIN"       --run_name "$run_name"       --method b3_support_only       --output_root "$OUTPUT_ROOT"       "${args[@]}"       > "$LOG_DIR/$run_name.out.log"       2> "$LOG_DIR/$run_name.err.log"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then
    log "DONE pid=$pid"
  else
    code=$?
    log "FAILED pid=$pid exit=$code"
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  log "STOP after failure in FPP-style TD3 B3-Critic-Only"
  exit 1
fi

log "ALL FetchPush FPP-style TD3+HER B3-Critic-Only seeds completed successfully"

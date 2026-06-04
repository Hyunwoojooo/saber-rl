#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/mimiclab_asus/SABER"
PYTHON="$ROOT/.venv/bin/python"
EXP_ROOT="$ROOT/FetchPush_SAC"
SESSION_LOG_DIR="$EXP_ROOT/orchestrator_logs"
mkdir -p "$SESSION_LOG_DIR"
ORCH_LOG="$SESSION_LOG_DIR/run_all_sac_fetchpush_5parallel_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$ORCH_LOG"
}

run_seed() {
  local variant_dir="$1"
  local run_name="$2"
  shift 2
  local exp_dir="$EXP_ROOT/$variant_dir"
  local train="$exp_dir/code/src/train_fetch_push_lite_v0.py"
  local log_dir="$exp_dir/logs"
  local output_root="$exp_dir/results"
  mkdir -p "$log_dir" "$output_root"
  (
    cd "$exp_dir"
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$train" "$@" --output_root "$output_root" > "$log_dir/$run_name.out.log" 2> "$log_dir/$run_name.err.log"
  ) &
  RUN_PID=$!
}

wait_variant() {
  local variant_label="$1"
  shift
  local pids=("$@")
  local failed=0
  for pid in "${pids[@]}"; do
    if wait "$pid"; then
      log "DONE $variant_label pid=$pid"
    else
      local code=$?
      log "FAILED $variant_label pid=$pid exit=$code"
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    log "STOP after failure in $variant_label"
    exit 1
  fi
}

common_args() {
  local seed="$1"
  printf "%s\0"     --reward_type sparse     --total_steps 1000000     --start_steps 10000     --warmup_steps 5000     --updates_per_step 1     --batch_size 256     --buffer_capacity 300000     --her_ratio 0.8     --gamma 0.98     --tau 0.005     --actor_lr 0.0003     --critic_lr 0.0003     --hidden_dim 256     --noise_std 0.2     --policy_delay 2     --target_policy_noise 0.2     --target_noise_clip 0.5     --eval_freq 10000     --eval_episodes 20     --checkpoint_every_steps 50000     --log_every_steps 1000     --device cuda     --env_seed "$seed"     --torch_seed "$seed"     --numpy_seed "$seed"     --action_noise_seed "$seed"     --weight_seed "$((20000 + seed))"     --eval_seed "$((10000 + seed))"     --no_save_step_checkpoints
}

launch_variant() {
  local variant_dir="$1"
  local label="$2"
  local prefix="$3"
  local method="$4"
  shift 4
  local extra=("$@")
  local pids=()
  for seed in 1 2 3 4 5; do
    local run_name="${prefix}_s${seed}_1m_cuda"
    local common=()
    while IFS= read -r -d "" arg; do
      common+=("$arg")
    done < <(common_args "$seed")
    log "START $label seed=$seed run=$run_name"
    run_seed "$variant_dir" "$run_name" --run_name "$run_name" --method "$method" "${common[@]}" "${extra[@]}"
    pids+=("$RUN_PID")
  done
  wait_variant "$label" "${pids[@]}"
}

log "BEGIN FetchPush SAC sequence: B0 -> B3-Critic-Only(FPP-style) -> PER-Style -> PER-Full"
log "Python=$PYTHON"
log "CUDA_VISIBLE_DEVICES=0, seeds=1..5, parallel=5 per variant"
log "SAC implementation: twin critics + tanh Gaussian actor + fixed entropy_alpha=0.2"

launch_variant SAC-HER-B0 SAC-HER-B0 sac_her_b0 b0_ddpg_her

launch_variant SAC-HER-B3-Critic-Only-FPP-Style SAC-HER-B3-Critic-Only-FPP-Style sac_her_b3_critic_only_fppstyle b3_support_only   --M_B 16   --M_Q_min 4   --N_local_min 4   --N_gap_min 128   --lambda_support 1.0   --weighting_mode batch_norm   --weight_scope critic_only   --local_search_max 1024   --policy_support_mode bin

launch_variant SAC-HER-PER-Style SAC-HER-PER-Style sac_her_per_style per_style_baseline   --weighting_mode batch_norm   --weight_scope critic_only   --per_alpha 0.6   --eps_PER 1e-6   --U_PER_max 2.0

launch_variant SAC-HER-PER-Full SAC-HER-PER-Full sac_her_per_full per_full_baseline   --weighting_mode batch_norm   --weight_scope critic_only   --per_alpha 0.6   --per_beta0 0.4   --per_beta1 1.0   --eps_PER 1e-6   --U_PER_max 2.0

log "ALL FetchPush SAC experiments completed successfully"

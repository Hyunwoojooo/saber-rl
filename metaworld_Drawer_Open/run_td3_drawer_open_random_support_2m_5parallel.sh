#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
EXP_DIR="$ROOT/B3_v4_512_s1_5"
TRAIN="$EXP_DIR/code/src/train_drawer_open_lite_v0.py"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

STEPS="${STEPS:-2000000}"
STEP_TAG="${STEP_TAG:-2m}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results_2m}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs_2m}"
ORCH_LOG_DIR="$ROOT/orchestrator_logs"
ORCH_LOG="$ORCH_LOG_DIR/run_td3_drawer_open_random_support_${STEP_TAG}_5parallel_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$ORCH_LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$ORCH_LOG"
}

common_args() {
  local seed="$1"
  printf "%s\0" \
    --task_name drawer_open \
    --reward_type sparse \
    --metaworld_env_name drawer-open-v3 \
    --metaworld_reward_function_version v2 \
    --max_episode_steps 200 \
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
    --no_save_step_checkpoints
}

b3_args() {
  printf "%s\0" \
    --M_B 16 \
    --M_Q_min 4 \
    --N_local_min 4 \
    --N_gap_min 128 \
    --local_search_max 512 \
    --lambda_support 1.0 \
    --drawer_success_threshold 0.05 \
    --drawer_d_reach 0.08 \
    --drawer_d_goal_near 0.05 \
    --drawer_d_opening_mid 0.12 \
    --weighting_mode batch_norm \
    --policy_support_mode bin \
    --support_scan_mode deterministic_tail_v1 \
    --support_cache_enabled \
    --support_cache_max_entries 200000 \
    --local_scan_cache_enabled \
    --local_scan_cache_max_entries 50000 \
    --support_short_circuit \
    --numba_support_scan \
    --support_predicate_early_exit \
    --support_diagnostic_audit_interval 100
}

run_variant() {
  local label="$1"
  local slug="$2"
  local method="$3"
  local weight_scope="$4"
  local output_dir="$OUTPUT_ROOT/$slug"
  local log_dir="$LOG_ROOT/$slug"
  mkdir -p "$output_dir" "$log_dir"

  log "BEGIN $label: method=$method weight_scope=$weight_scope seeds=1..5 steps=$STEPS output=$output_dir"
  local pids=()
  for seed in 1 2 3 4 5; do
    local run_name="drawer_open_${slug}_s${seed}_${STEP_TAG}_cuda"
    local log_file="$log_dir/${run_name}.log"
    local args=()
    while IFS= read -r -d "" arg; do
      args+=("$arg")
    done < <(common_args "$seed")
    while IFS= read -r -d "" arg; do
      args+=("$arg")
    done < <(b3_args)

    log "START $label seed=$seed run=$run_name log=$log_file"
    (
      cd "$EXP_DIR"
      PYTHONPATH="$EXP_DIR/code/src:${PYTHONPATH:-}" \
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
      "$PYTHON" "$TRAIN" \
        "${args[@]}" \
        --method "$method" \
        --weight_scope "$weight_scope" \
        --run_name "$run_name" \
        --output_root "$output_dir"
    ) >"$log_file" 2>&1 &
    pids+=("$!")
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    log "STOP after failure in $label"
    return "$status"
  fi
  log "END $label"
}

log "BEGIN MetaWorld DrawerOpen TD3 Random-Support 2M sequence"
log "Python=$PYTHON CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
run_variant "TD3+HER+B3-Random-Support 2M" "TD3-HER-B3-Random-Support-2M" "b3_support_random_runtime_opt" "critic_only"
log "ALL MetaWorld DrawerOpen TD3 Random-Support 2M variants completed successfully"

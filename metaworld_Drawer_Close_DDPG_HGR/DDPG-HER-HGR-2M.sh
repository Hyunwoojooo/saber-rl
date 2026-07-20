#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
EXP_DIR="$ROOT"
TRAIN="$ROOT/code/src/train_drawer_close_lite_v0.py"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

STEPS="${STEPS:-2000000}"
STEP_TAG="${STEP_TAG:-2m}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SEED_LIST="${SEED_LIST:-1 2 3 4 5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runs/ddpg_hgr_tdpriority_original_pair_is_${STEP_TAG}}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/ddpg_hgr_tdpriority_original_pair_is_${STEP_TAG}}"
DRY_RUN="${DRY_RUN:-0}"
ORCH_LOG_DIR="$ROOT/orchestrator_logs"
ORCH_LOG="$ORCH_LOG_DIR/DDPG-HER-HGR-${STEP_TAG}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$ORCH_LOG_DIR"
read -r -a SEEDS <<< "$SEED_LIST"
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  printf 'SEED_LIST must contain at least one seed\n' >&2
  exit 2
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$ORCH_LOG"
}

common_args() {
  local seed="$1"
  printf "%s\0" \
    --task_name drawer_close \
    --reward_type sparse \
    --metaworld_env_name drawer-close-v3 \
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
    --hgr_priority_eps 0.000001 \
    --hgr_priority_init 1.0 \
    --hgr_priority_clip_max 0.0 \
    --hgr_alpha_episode 0.6 \
    --hgr_alpha_goal 0.6 \
    --hgr_beta_episode_initial 0.4 \
    --hgr_beta_goal_initial 0.4 \
    --hgr_beta_episode_final 1.0 \
    --hgr_beta_goal_final 1.0 \
    --hgr_beta_anneal_updates 0 \
    --hgr_uniform_mix 0.0 \
    --env_seed "$seed" \
    --torch_seed "$seed" \
    --numpy_seed "$seed" \
    --action_noise_seed "$seed" \
    --weight_seed "$((20000 + seed))" \
    --eval_seed "$((10000 + seed))" \
    --no_save_step_checkpoints
}

run_variant() {
  local label="$1"
  local slug="$2"
  local output_dir="$OUTPUT_ROOT/$slug"
  local log_dir="$LOG_ROOT/$slug"
  mkdir -p "$output_dir" "$log_dir"

  log "BEGIN $label: method=hgr seeds=${SEEDS[*]} steps=$STEPS output=$output_dir"
  local pids=()
  for seed in "${SEEDS[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
      log "Invalid seed in SEED_LIST: $seed"
      return 2
    fi
    local run_name="drawer_close_ddpg_hgr_original_pair_is_s${seed}_${STEP_TAG}_cuda"
    local log_file="$log_dir/${run_name}.log"
    local args=()
    while IFS= read -r -d "" arg; do
      args+=("$arg")
    done < <(common_args "$seed")

    if [[ "$DRY_RUN" == "1" ]]; then
      log "DRY_RUN seed=$seed run=$run_name"
      printf 'cd %q && PYTHONPATH=%q CUDA_VISIBLE_DEVICES=%q %q %q' \
        "$EXP_DIR" "$EXP_DIR/code/src:${PYTHONPATH:-}" "$CUDA_DEVICE" "$PYTHON" "$TRAIN"
      printf ' %q' "${args[@]}" --method hgr --weight_scope none --run_name "$run_name" --output_root "$output_dir"
      printf '\n'
      continue
    fi

    log "START $label seed=$seed run=$run_name log=$log_file"
    (
      cd "$EXP_DIR"
      PYTHONPATH="$EXP_DIR/code/src:${PYTHONPATH:-}" \
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
      "$PYTHON" "$TRAIN" \
        "${args[@]}" \
        --method hgr \
        --weight_scope none \
        --run_name "$run_name" \
        --output_root "$output_dir"
    ) >"$log_file" 2>&1 &
    pids+=("$!")
  done

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN completed"
    return 0
  fi

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

log "BEGIN MetaWorld DrawerClose DDPG-HER-HGR-original-pair-is-${STEP_TAG}"
log "Python=$PYTHON CUDA_VISIBLE_DEVICES=$CUDA_DEVICE DRY_RUN=$DRY_RUN"
run_variant "DDPG+HER+HGR original-pair-IS ${STEP_TAG}" "DDPG-HER-HGR-original-pair-is-${STEP_TAG}"
log "DDPG-HER-HGR-original-pair-is-${STEP_TAG} completed successfully"

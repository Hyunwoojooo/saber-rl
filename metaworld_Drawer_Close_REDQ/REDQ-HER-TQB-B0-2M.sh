#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
EXP_DIR="$ROOT"
TRAIN="$EXP_DIR/code/src/train_drawer_close_redq_lite_v0.py"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

STEPS="${STEPS:-2000000}"
STEP_TAG="${STEP_TAG:-2m}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-4}"
GAMMA="${GAMMA:-0.98}"
REDQ_ENSEMBLE_SIZE="${REDQ_ENSEMBLE_SIZE:-10}"
TARGET_SUBSET_SIZE="${TARGET_SUBSET_SIZE:-2}"
ENTROPY_ALPHA="${ENTROPY_ALPHA:-0.2}"
SAC_AUTO_ALPHA="${SAC_AUTO_ALPHA:-true}"
TARGET_ENTROPY="${TARGET_ENTROPY:--4.0}"
ALPHA_LR="${ALPHA_LR:-0.0003}"
TARGET_Q_MIN="${TARGET_Q_MIN:--50.0}"
TARGET_Q_MAX="${TARGET_Q_MAX:-0.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results_2m}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs_2m}"
ORCH_LOG_DIR="$ROOT/orchestrator_logs"
ORCH_LOG="$ORCH_LOG_DIR/REDQ-HER-TQB-B0-2M_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$ORCH_LOG_DIR"

log() {
  printf '[%s] %s
' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$ORCH_LOG"
}

build_common_args() {
  local seed="$1"
  COMMON_ARGS=(
    --task_name drawer_close
    --reward_type sparse
    --metaworld_env_name drawer-close-v3
    --metaworld_reward_function_version v2
    --max_episode_steps 200
    --total_steps "$STEPS"
    --start_steps 10000
    --warmup_steps 5000
    --updates_per_step "$UPDATES_PER_STEP"
    --batch_size 256
    --buffer_capacity 300000
    --her_ratio 0.8
    --gamma "$GAMMA"
    --tau 0.005
    --actor_lr 0.0003
    --critic_lr 0.0003
    --hidden_dim 256
    --redq_ensemble_size "$REDQ_ENSEMBLE_SIZE"
    --target_subset_size "$TARGET_SUBSET_SIZE"
    --entropy_alpha "$ENTROPY_ALPHA"
    --target_entropy "$TARGET_ENTROPY"
    --alpha_lr "$ALPHA_LR"
    --noise_std 0.2
    --policy_delay 2
    --target_policy_noise 0.2
    --target_noise_clip 0.5
    --target_q_bounding
    --target_q_min "$TARGET_Q_MIN"
    --target_q_max "$TARGET_Q_MAX"
    --eval_freq 10000
    --eval_episodes 20
    --checkpoint_every_steps 50000
    --log_every_steps 1000
    --device cuda
    --env_seed "$seed"
    --torch_seed "$seed"
    --numpy_seed "$seed"
    --action_noise_seed "$seed"
    --weight_seed "$((20000 + seed))"
    --eval_seed "$((10000 + seed))"
    --no_save_step_checkpoints
  )
  if [[ "$SAC_AUTO_ALPHA" == "true" || "$SAC_AUTO_ALPHA" == "1" ]]; then
    COMMON_ARGS+=(--sac_auto_alpha)
  else
    COMMON_ARGS+=(--no-sac_auto_alpha)
  fi
}

run_variant() {
  local label="$1"
  local slug="$2"
  local method="$3"
  local weight_scope="$4"
  local output_dir="$OUTPUT_ROOT/$slug"
  local log_dir="$LOG_ROOT/$slug"
  mkdir -p "$output_dir" "$log_dir"

  log "BEGIN $label: method=$method weight_scope=$weight_scope seeds=1..5 steps=$STEPS updates_per_step=$UPDATES_PER_STEP ensemble=$REDQ_ENSEMBLE_SIZE target_subset=$TARGET_SUBSET_SIZE sac_auto_alpha=$SAC_AUTO_ALPHA entropy_alpha=$ENTROPY_ALPHA target_entropy=$TARGET_ENTROPY target_q=[$TARGET_Q_MIN,$TARGET_Q_MAX] output=$output_dir"
  local pids=()
  for seed in 1 2 3 4 5; do
    local run_name="drawer_close_${slug}_s${seed}_${STEP_TAG}_cuda"
    local log_file="$log_dir/${run_name}.log"
    build_common_args "$seed"

    log "START $label seed=$seed run=$run_name log=$log_file"
    (
      cd "$EXP_DIR"
      PYTHONPATH="$EXP_DIR/code/src:${PYTHONPATH:-}"       CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"       "$PYTHON" "$TRAIN"         "${COMMON_ARGS[@]}"         --method "$method"         --weight_scope "$weight_scope"         --run_name "$run_name"         --output_root "$output_dir"
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

log "BEGIN MetaWorld DrawerClose REDQ-HER-TQB-B0-2M"
log "Python=$PYTHON CUDA_VISIBLE_DEVICES=$CUDA_DEVICE updates_per_step=$UPDATES_PER_STEP gamma=$GAMMA ensemble=$REDQ_ENSEMBLE_SIZE target_subset=$TARGET_SUBSET_SIZE sac_auto_alpha=$SAC_AUTO_ALPHA entropy_alpha=$ENTROPY_ALPHA target_entropy=$TARGET_ENTROPY target_q=[$TARGET_Q_MIN,$TARGET_Q_MAX]"
run_variant "REDQ+HER+Target-Q-Bounding B0 2M" "REDQ-HER-TQB-B0-2M" "b0_ddpg_her" "none"
log "REDQ-HER-TQB-B0-2M completed successfully"

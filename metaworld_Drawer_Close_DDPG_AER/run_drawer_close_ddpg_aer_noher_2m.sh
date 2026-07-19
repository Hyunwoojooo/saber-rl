#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
TRAIN="$ROOT/code/src/train_drawer_close_lite_v0.py"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

STEPS="${STEPS:-2000000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
ARMS="${ARMS:-aer er annealing_er}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runs}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs}"
ORCH_LOG_DIR="$ROOT/orchestrator_logs"
ORCH_LOG="$ORCH_LOG_DIR/DDPG-AER-NOHER-2M_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$ORCH_LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$ORCH_LOG"
}

replay_method_for() {
  case "$1" in
    er) printf '%s' "ddpg_er" ;;
    annealing_er) printf '%s' "ddpg_annealing_er" ;;
    aer) printf '%s' "ddpg_aer" ;;
    *) return 2 ;;
  esac
}

run_arm() {
  local arm="$1"
  local replay_method
  replay_method="$(replay_method_for "$arm")"
  local output_dir="$OUTPUT_ROOT/$replay_method"
  local log_dir="$LOG_ROOT/$replay_method"
  mkdir -p "$output_dir" "$log_dir"
  log "BEGIN arm=$arm replay_method=$replay_method seeds=1..5 steps=$STEPS"

  local pids=()
  for seed in 1 2 3 4 5; do
    local run_name="drawer_close_${replay_method}_s${seed}_${STEPS}"
    local log_file="$log_dir/${run_name}.log"
    log "START arm=$arm seed=$seed run=$run_name"
    (
      cd "$ROOT"
      PYTHONPATH="$ROOT/code/src:${PYTHONPATH:-}" \
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
      "$PYTHON" "$TRAIN" \
        --task_name drawer_close \
        --run_name "$run_name" \
        --output_root "$output_dir" \
        --method b0_ddpg_er \
        --replay_method "$replay_method" \
        --weight_scope none \
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
        --her_ratio 0.0 \
        --gamma 0.98 \
        --tau 0.005 \
        --actor_lr 0.0003 \
        --critic_lr 0.0003 \
        --hidden_dim 256 \
        --noise_std 0.2 \
        --policy_delay 1 \
        --target_policy_noise 0.0 \
        --target_noise_clip 0.0 \
        --aer_lambda0 2.5 \
        --aer_alpha 0.6666666666666666 \
        --aer_cosine_eps 1e-8 \
        --eval_freq 10000 \
        --eval_episodes 20 \
        --final_eval_episodes 100 \
        --checkpoint_every_steps 50000 \
        --log_every_steps 1000 \
        --device cuda \
        --env_seed "$seed" \
        --torch_seed "$seed" \
        --numpy_seed "$seed" \
        --action_noise_seed "$seed" \
        --weight_seed "$((20000 + seed))" \
        --eval_seed "$((10000 + seed))" \
        --final_eval_seed "$((40000 + seed))" \
        --no_save_step_checkpoints
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
    log "STOP after failure arm=$arm"
    return "$status"
  fi
  log "END arm=$arm"
}

log "BEGIN DrawerClose DDPG no-HER AER three-arm block"
log "Python=$PYTHON CUDA_VISIBLE_DEVICES=$CUDA_DEVICE arms=$ARMS"
for arm in $ARMS; do
  run_arm "$arm"
done
log "ALL ARMS completed successfully"

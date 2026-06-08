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

CUDA_DEVICE="${CUDA_DEVICE:-0}"
STEPS="${STEPS:-1000}"
OUT_ROOT="$ROOT/results_smoke"
LOG_DIR="$ROOT/logs_smoke"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

base_args=(
  --task_name drawer_open
  --reward_type sparse
  --metaworld_env_name drawer-open-v3
  --metaworld_reward_function_version v2
  --max_episode_steps 200
  --total_steps "$STEPS"
  --start_steps 200
  --warmup_steps 200
  --updates_per_step 1
  --batch_size 64
  --buffer_capacity 20000
  --her_ratio 0.8
  --gamma 0.98
  --tau 0.005
  --actor_lr 0.0003
  --critic_lr 0.0003
  --hidden_dim 256
  --noise_std 0.2
  --policy_delay 2
  --target_policy_noise 0.2
  --target_noise_clip 0.5
  --eval_freq 500
  --eval_episodes 5
  --checkpoint_every_steps 500
  --log_every_steps 100
  --M_B 16
  --M_Q_min 4
  --N_local_min 4
  --N_gap_min 32
  --local_search_max 256
  --lambda_support 1.0
  --drawer_success_threshold 0.05
  --drawer_d_reach 0.08
  --drawer_d_goal_near 0.05
  --drawer_d_opening_mid 0.12
  --weighting_mode batch_norm
  --policy_support_mode bin
  --support_scan_mode deterministic_tail_v1
  --support_cache_enabled
  --support_cache_max_entries 20000
  --local_scan_cache_enabled
  --local_scan_cache_max_entries 10000
  --support_short_circuit
  --numba_support_scan
  --support_predicate_early_exit
  --support_diagnostic_audit_interval 20
  --device cuda
  --env_seed 1
  --torch_seed 1
  --numpy_seed 1
  --action_noise_seed 1
  --weight_seed 20001
  --eval_seed 10001
  --no_save_step_checkpoints
)

cd "$EXP_DIR"
export PYTHONPATH="$EXP_DIR/code/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

echo "[validate] DrawerOpen TD3/B3 setup"
"$PYTHON" "$TRAIN" "${base_args[@]}" \
  --method b3_support_runtime_opt \
  --weight_scope critic_only \
  --run_name drawer_open_td3_validate \
  --output_root "$OUT_ROOT/validate" \
  --validate_only

run_one() {
  local label="$1"
  local slug="$2"
  local method="$3"
  local weight_scope="$4"
  local log_file="$LOG_DIR/${slug}_${STEPS}.log"
  echo "[smoke] $label method=$method log=$log_file"
  "$PYTHON" "$TRAIN" "${base_args[@]}" \
    --method "$method" \
    --weight_scope "$weight_scope" \
    --run_name "drawer_open_${slug}_${STEPS}_smoke" \
    --output_root "$OUT_ROOT/$slug" >"$log_file" 2>&1
}

run_one "TD3+HER" "TD3-HER-B0" "b0_ddpg_her" "none"
run_one "TD3+HER+B3-Critic-Only" "TD3-HER-B3-Critic-Only" "b3_support_runtime_opt" "critic_only"
run_one "TD3+HER+PER-Style" "TD3-HER-PER-Style" "per_style_loss_weighting" "critic_only"
echo "[done] DrawerOpen TD3 smoke completed"

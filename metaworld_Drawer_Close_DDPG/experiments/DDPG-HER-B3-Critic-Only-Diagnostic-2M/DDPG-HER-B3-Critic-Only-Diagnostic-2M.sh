#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../../.." && pwd)"
TRAIN="$ROOT/code/src/train_drawer_close_lite_v0.py"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

STEPS="${STEPS:-2000000}"
STEP_TAG="${STEP_TAG:-2m}"
SEEDS="${SEEDS:-1 2 3 4 5}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results_2m_diag/DDPG-HER-B3-Critic-Only-2M-diagnostic}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs_2m_diag/DDPG-HER-B3-Critic-Only-2M-diagnostic}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/diagnostic_artifact}"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

pids=()
for seed in $SEEDS; do
  run_name="drawer_close_DDPG-HER-B3-Critic-Only-2M-diagnostic_s${seed}_${STEP_TAG}_cuda"
  log_file="$LOG_ROOT/${run_name}.log"
  (
    cd "$ROOT"
    PYTHONPATH="$ROOT/code/src:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
    "$PYTHON" "$TRAIN" \
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
      --env_seed "$seed" \
      --torch_seed "$seed" \
      --numpy_seed "$seed" \
      --action_noise_seed "$seed" \
      --weight_seed "$((20000 + seed))" \
      --eval_seed "$((10000 + seed))" \
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
      --support_diagnostic_audit_interval 100 \
      --method b3_support_runtime_opt \
      --weight_scope critic_only \
      --no_save_step_checkpoints \
      --run_name "$run_name" \
      --output_root "$OUTPUT_ROOT"
  ) >"$log_file" 2>&1 &
  pids+=("$!")
  printf 'started seed=%s pid=%s log=%s\n' "$seed" "${pids[-1]}" "$log_file"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  printf 'one or more diagnostic seeds failed; inspect %s\n' "$LOG_ROOT" >&2
  exit "$status"
fi

printf 'all diagnostic seeds completed: %s\n' "$OUTPUT_ROOT"
"$PYTHON" "$ROOT/analysis/build_diagnostic_artifact.py" \
  --input-root "$OUTPUT_ROOT" \
  --output-root "$ARTIFACT_ROOT" \
  --seeds $SEEDS \
  --expected-final-step "$STEPS"
printf 'diagnostic artifact completed: %s\n' "$ARTIFACT_ROOT"

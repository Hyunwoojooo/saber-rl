#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/context_count_only_smoke/results_5k/DDPG-HER-B3-Context-Count-Only-2M}"
LOG_ROOT="${LOG_ROOT:-$ROOT/context_count_only_smoke/logs_5k}"
RUN_NAME="drawer_open_DDPG-HER-B3-Context-Count-Only-2M_s1_5k_correctness_cuda"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

PYTHONPATH="$ROOT/code/src:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
"$PYTHON" "$ROOT/code/src/train_drawer_open_lite_v0.py" \
  --task_name drawer_open --reward_type sparse --metaworld_env_name drawer-open-v3 \
  --metaworld_reward_function_version v2 --max_episode_steps 200 --total_steps 5000 \
  --start_steps 1000 --warmup_steps 1000 --updates_per_step 1 --batch_size 256 \
  --buffer_capacity 300000 --her_ratio 0.8 --gamma 0.98 --tau 0.005 \
  --actor_lr 0.0003 --critic_lr 0.0003 --hidden_dim 256 --noise_std 0.2 \
  --policy_delay 2 --target_policy_noise 0.2 --target_noise_clip 0.5 \
  --eval_freq 1000 --eval_episodes 5 --checkpoint_every_steps 1000 --log_every_steps 1000 \
  --device cuda --env_seed 1 --torch_seed 1 --numpy_seed 1 --action_noise_seed 1 \
  --weight_seed 20001 --eval_seed 10001 --no_save_step_checkpoints \
  --M_B 16 --M_Q_min 4 --N_local_min 4 --N_gap_min 128 --local_search_max 512 \
  --action_bins_per_dim 5 --lambda_support 1 --drawer_success_threshold 0.05 \
  --drawer_d_reach 0.08 --drawer_d_goal_near 0.05 --drawer_d_opening_mid 0.12 \
  --weighting_mode batch_norm --policy_support_mode bin --support_query_mode context_count_only \
  --support_scan_mode deterministic_tail_v1 --support_cache_enabled \
  --support_cache_max_entries 200000 --local_scan_cache_enabled \
  --local_scan_cache_max_entries 50000 --support_short_circuit --numba_support_scan \
  --support_predicate_early_exit --support_diagnostic_audit_interval 100 \
  --method b3_support_runtime_opt --weight_scope critic_only \
  --run_name "$RUN_NAME" --output_root "$OUTPUT_ROOT" \
  > "$LOG_ROOT/$RUN_NAME.log" 2>&1

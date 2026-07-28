#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../../.." && pwd)"
TRAIN="$EXP_DIR/code/src/train_drawer_open_lite_v0.py"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  GIT_COMMON_DIR="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
  SHARED_PYTHON="$(dirname "$GIT_COMMON_DIR")/.venv/bin/python"
  if [[ -x "$SHARED_PYTHON" ]]; then
    PYTHON="$SHARED_PYTHON"
  else
    PYTHON="$(command -v python3)"
  fi
fi

read -r -a SEED_ARRAY <<< "${SEEDS:-1 2 3 4 5}"
if [[ "${#SEED_ARRAY[@]}" -ne 5 ]]; then
  echo "SEEDS must contain exactly five IDs; got: ${SEED_ARRAY[*]}" >&2
  exit 2
fi
declare -A SEEN=()
for seed in "${SEED_ARRAY[@]}"; do
  if [[ ! "$seed" =~ ^([1-9]|10)$ ]] || [[ -n "${SEEN[$seed]:-}" ]]; then
    echo "SEEDS must be five unique IDs in 1..10; got: ${SEED_ARRAY[*]}" >&2
    exit 2
  fi
  SEEN[$seed]=1
done

STEPS="${STEPS:-2000000}"
STEP_TAG="${STEP_TAG:-2m_fixed_target_diag}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EXP_DIR/results_2m}"
LOG_ROOT="${LOG_ROOT:-$EXP_DIR/logs_2m}"
ORCH_LOG_DIR="$EXP_DIR/orchestrator_logs"
ORCH_LOG="$ORCH_LOG_DIR/fixed_target_diag_${SEED_ARRAY[0]}_${SEED_ARRAY[4]}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$ORCH_LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$ORCH_LOG"
}

if ! "$PYTHON" -c 'import llvmlite, numba, numpy, torch; assert numpy.__version__ == "2.3.5"; assert numba.__version__ == "0.66.0"; assert llvmlite.__version__ == "0.48.0"; assert torch.cuda.is_available()'; then
  echo "Install the pinned runtime first: $PYTHON -m pip install -r $EXP_DIR/requirements-runtime.txt" >&2
  exit 2
fi
PYTHONPATH="$EXP_DIR/code/src:${PYTHONPATH:-}" "$PYTHON" -m unittest \
  discover -s "$EXP_DIR/code/src" -p 'test_drawer_open_fixed_target_diagnostic.py'

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

saber_args() {
  printf "%s\0" \
    --method b3_support_runtime_opt \
    --weight_scope critic_only \
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

validation_args=()
while IFS= read -r -d "" arg; do validation_args+=("$arg"); done < <(common_args "1")
while IFS= read -r -d "" arg; do validation_args+=("$arg"); done < <(saber_args)
PYTHONPATH="$EXP_DIR/code/src:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
"$PYTHON" "$TRAIN" \
  "${validation_args[@]}" \
  --run_name drawer_open_fixed_target_preflight \
  --output_root /tmp \
  --validate_only

log "BEGIN DrawerOpen DDPG+HER+SABER fixed-target diagnostic"
log "seeds=${SEED_ARRAY[*]} steps=$STEPS python=$PYTHON cuda=$CUDA_DEVICE"
pids=()
run_dirs=()
for seed in "${SEED_ARRAY[@]}"; do
  run_name="drawer_open_DDPG-HER-SABER-fixed-target-diagnostic_s${seed}_${STEP_TAG}_cuda"
  run_dir="$OUTPUT_ROOT/$run_name"
  log_file="$LOG_ROOT/${run_name}.log"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to mix runs: output already exists: $run_dir" >&2
    exit 2
  fi
  args=()
  while IFS= read -r -d "" arg; do args+=("$arg"); done < <(common_args "$seed")
  while IFS= read -r -d "" arg; do args+=("$arg"); done < <(saber_args)

  log "START seed=$seed run=$run_name log=$log_file"
  (
    cd "$EXP_DIR"
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$EXP_DIR/code/src:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
    "$PYTHON" "$TRAIN" \
      "${args[@]}" \
      --run_name "$run_name" \
      --output_root "$OUTPUT_ROOT"
  ) >"$log_file" 2>&1 &
  pids+=("$!")
  run_dirs+=("$run_dir")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [[ "$status" -ne 0 ]]; then
  log "FAILED: at least one seed exited nonzero"
  exit "$status"
fi
"$PYTHON" "$EXP_DIR/validate_completed_runs.py" --steps "$STEPS" "${run_dirs[@]}"
log "END: all five seeds completed and validated"

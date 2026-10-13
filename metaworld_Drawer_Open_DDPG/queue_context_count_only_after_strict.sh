#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
UPSTREAM_SESSION="${UPSTREAM_SESSION:-drawer-open-ddpg-b3-strict-ctx-countmatched-2m-after-sac}"
LAUNCHER="$ROOT/DDPG-HER-B3-Context-Count-Only-2M.sh"
SMOKE_ROOT="$ROOT/context_count_only_smoke"
FULL_ROOT="$ROOT/results_2m/DDPG-HER-B3-Context-Count-Only-2M"
QUEUE_LOG_DIR="$ROOT/orchestrator_logs"
QUEUE_LOG="$QUEUE_LOG_DIR/DDPG-HER-B3-Context-Count-Only-2M_queue.log"
mkdir -p "$QUEUE_LOG_DIR"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$QUEUE_LOG"; }

log "waiting for strict upstream session: $UPSTREAM_SESSION"
while tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; do sleep 60; done
while pgrep -af '[t]rain_drawer_open_lite_v0.py' >/dev/null 2>&1; do sleep 60; done

if [[ -e "$FULL_ROOT" ]]; then
  log "refusing to overwrite existing full-run root: $FULL_ROOT"
  exit 2
fi
if [[ -e "$SMOKE_ROOT" ]]; then
  log "refusing to overwrite existing smoke root: $SMOKE_ROOT"
  exit 2
fi

log "running static, parity, unit, regression, and environment gates"
"$PYTHON" "$ROOT/freeze_context_count_only_source.py" --verify
bash -n "$LAUNCHER"
"$PYTHON" -m py_compile \
  "$ROOT/code/src/train_drawer_open_lite_v0.py" \
  "$ROOT/code/src/metacog_lite_v0/config.py" \
  "$ROOT/code/src/metacog_lite_v0/weights.py" \
  "$ROOT/code/src/metacog_lite_v0/numba_support_scan_drawer_open.py" \
  "$ROOT/code/src/metacog_lite_v0/trainer.py"
PYTHONPATH="$ROOT/code/src" "$PYTHON" "$ROOT/code/src/test_context_count_only_drawer_open.py"
"$PYTHON" "$ROOT/test_context_count_only_config_parity.py"
"$PYTHON" "$ROOT/validate_context_count_only_config_parity.py" > "$SMOKE_ROOT.parity.json"
PYTHONPATH="$ROOT/code/src" "$PYTHON" "$ROOT/strict_context_eligible_count_matched_2m/code/src/test_strict_ddpg_loss_scope.py"
PYTHONPATH="$ROOT/code/src" "$PYTHON" "$ROOT/strict_context_eligible_count_matched_2m/code/src/test_strict_drawer_open_scanner.py"
for query_mode in current_policy context_count_only; do
  PYTHONPATH="$ROOT/code/src" "$PYTHON" "$ROOT/code/src/train_drawer_open_lite_v0.py" \
    --task_name drawer_open --reward_type sparse --metaworld_env_name drawer-open-v3 \
    --metaworld_reward_function_version v2 --method b3_support_runtime_opt \
    --weight_scope critic_only --weighting_mode batch_norm --M_B 16 --M_Q_min 4 \
    --N_local_min 4 --local_search_max 512 --action_bins_per_dim 5 --lambda_support 1 \
    --support_query_mode "$query_mode" --device cpu --validate_only
done

log "running seed-1 5k correctness smoke"
CUDA_DEVICE=0 OUTPUT_ROOT="$SMOKE_ROOT/results_5k/DDPG-HER-B3-Context-Count-Only-2M" LOG_ROOT="$SMOKE_ROOT/logs_5k" bash "$ROOT/run_context_count_only_correctness_5k.sh"
"$PYTHON" "$ROOT/validate_context_count_only_run.py" \
  --root "$SMOKE_ROOT/results_5k" --steps 5000 --eval-freq 1000 --eval-episodes 5 --seeds 1

log "running seed-1 20k CUDA smoke"
SEEDS=1 STEPS=20000 STEP_TAG=20k_smoke CUDA_DEVICE=0 \
OUTPUT_ROOT="$SMOKE_ROOT/results_20k" LOG_ROOT="$SMOKE_ROOT/logs_20k" bash "$LAUNCHER"
"$PYTHON" "$ROOT/validate_context_count_only_run.py" \
  --root "$SMOKE_ROOT/results_20k" --steps 20000 --eval-freq 10000 --eval-episodes 20 --seeds 1

"$PYTHON" "$ROOT/freeze_context_count_only_source.py" --verify
log "smoke gates passed; starting seeds 1-5 x 2M"
SEEDS="1 2 3 4 5" STEPS=2000000 STEP_TAG=2m CUDA_DEVICE=0 \
OUTPUT_ROOT="$ROOT/results_2m" LOG_ROOT="$ROOT/logs_2m" bash "$LAUNCHER"
"$PYTHON" "$ROOT/validate_context_count_only_run.py" \
  --root "$ROOT/results_2m" --steps 2000000 --eval-freq 10000 --eval-episodes 20 --seeds 1 2 3 4 5
"$PYTHON" "$ROOT/analyze_context_count_only_results.py" --output-dir "$ROOT/context_count_only_analysis_2m"
"$PYTHON" "$ROOT/build_context_count_only_artifact.py"
"$PYTHON" "$ROOT/verify_context_count_only_artifact.py" --artifact "$ROOT/paper_artifacts/drawer_open_ddpg_context_count_only_2m"
log "context-count-only full cohort passed completion gate"

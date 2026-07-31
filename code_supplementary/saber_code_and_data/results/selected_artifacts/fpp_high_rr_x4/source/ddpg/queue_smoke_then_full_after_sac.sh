#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${RAW_REPOSITORY}}"
WAIT_SESSION="${WAIT_SESSION:-fpp_sac_controls_after_td3_controls}"
EXP_DIR="$REPO_ROOT/SABER-FetchPickAndPlace-active/FetchPickAndPlace/B0-high-replay-ratio-x4"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/B0-high-replay-ratio-x4_queue}"
SMOKE_ROOT="${SMOKE_ROOT:-$REPO_ROOT/experiment_FPP/runs/B0-high-replay-ratio-x4_smoke_after_sac_$(date +%Y%m%d_%H%M%S)}"
FULL_ROOT="${FULL_ROOT:-$REPO_ROOT/experiment_FPP/runs/B0-high-replay-ratio-x4}"
FULL_LOG_DIR="${FULL_LOG_DIR:-$REPO_ROOT/experiment_FPP/logs/B0-high-replay-ratio-x4_1m}"
RUN_NAME_SMOKE="fetch_pick_and_place_b0_high_rr_x4_s1_50000_ls1024"
SMOKE_RUN_DIR="$SMOKE_ROOT/$RUN_NAME_SMOKE"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$QUEUE_LOG_DIR"

echo "[$(date '+%F %T %Z')] queued FPP DDPG high-RR x4 smoke/full"
echo "wait_session=$WAIT_SESSION"
echo "smoke_root=$SMOKE_ROOT"
echo "full_root=$FULL_ROOT"

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  echo "[dry-run] would wait for tmux session: $WAIT_SESSION"
  echo "[dry-run] smoke command: python3 $EXP_DIR/run.py --seed 1 --steps 50000 --device cuda --local_search_max 1024 --output_root $SMOKE_ROOT"
  echo "[dry-run] full command: SEED_LIST='1 2 3 4 5' STEPS=1000000 DEVICE=cuda LOCAL_SEARCH_MAX=1024 OUTPUT_ROOT=$FULL_ROOT LOG_DIR=$FULL_LOG_DIR bash $EXP_DIR/run_fpp_ddpg_high_rr_x4_1m_5parallel.sh"
  exit 0
fi

while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  echo "[$(date '+%F %T %Z')] waiting for tmux session $WAIT_SESSION to finish"
  sleep 60
done

echo "[$(date '+%F %T %Z')] wait session finished; starting 50k smoke"
python3 "$EXP_DIR/run.py" \
  --seed 1 \
  --steps 50000 \
  --device cuda \
  --local_search_max 1024 \
  --output_root "$SMOKE_ROOT" \
  > "$QUEUE_LOG_DIR/smoke_seed1.log" 2>&1

echo "[$(date '+%F %T %Z')] smoke finished; validating $SMOKE_RUN_DIR"
RUN_DIR="$SMOKE_RUN_DIR" EXP_DIR="$EXP_DIR" python3 - <<'PY_VALIDATE'
import json
import os
import sys
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR'])
exp_dir = Path(os.environ['EXP_DIR'])
missing = [name for name in ('summary.json', 'config.json', 'env_snapshot.json') if not (run_dir / name).exists()]
if missing:
    raise SystemExit(f"missing smoke artifacts in {run_dir}: {missing}")

cfg = json.loads((run_dir / 'config.json').read_text())
folder_cfg = json.loads((exp_dir / 'config.json').read_text())
checks = {
    'runtime method=b0_ddpg_her': cfg.get('method') == 'b0_ddpg_her',
    'runtime updates_per_step=4': int(cfg.get('updates_per_step', -1)) == 4,
    'runtime lambda_support=0.0': float(cfg.get('lambda_support', 999.0)) == 0.0,
    'runtime weight_scope=none': cfg.get('weight_scope') == 'none',
    'runtime total_steps=50000': int(cfg.get('total_steps', -1)) == 50000,
    'runtime reward_type=sparse': cfg.get('reward_type') == 'sparse',
    'runtime batch_size=256': int(cfg.get('batch_size', -1)) == 256,
    'runtime her_ratio=0.8': abs(float(cfg.get('her_ratio', -1.0)) - 0.8) < 1e-12,
    'runtime run_name smoke': cfg.get('run_name') == 'fetch_pick_and_place_b0_high_rr_x4_s1_50000_ls1024',
    'folder support_lookup=false': folder_cfg.get('support_lookup') is False,
    'folder trainer_method=b0_ddpg_her': folder_cfg.get('trainer_method') == 'b0_ddpg_her',
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    for name, ok in checks.items():
        print(f"{name}: {ok}")
    raise SystemExit(f"smoke validation failed: {failed}")
print('smoke validation passed')
for key in ('method', 'updates_per_step', 'lambda_support', 'weight_scope', 'total_steps', 'run_name'):
    print(f"{key}={cfg.get(key)}")
PY_VALIDATE

echo "[$(date '+%F %T %Z')] validation passed; starting 1M seeds 1-5 full run"
SEED_LIST="1 2 3 4 5" \
STEPS=1000000 \
DEVICE=cuda \
LOCAL_SEARCH_MAX=1024 \
OUTPUT_ROOT="$FULL_ROOT" \
LOG_DIR="$FULL_LOG_DIR" \
bash "$EXP_DIR/run_fpp_ddpg_high_rr_x4_1m_5parallel.sh"

echo "[$(date '+%F %T %Z')] full high-RR x4 run finished"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
UPSTREAM_SESSION="${UPSTREAM_SESSION:-fpp-sac-annealing-er-main1m-after-er-s1-5}"
RESULT_ROOT="$REPO_ROOT/metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Strict-Context-Eligible-Count-Matched-2M"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }

if ! tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; then
  log "required upstream session is not present: $UPSTREAM_SESSION"
  exit 2
fi

log "waiting for upstream SAC-AnnealingER session: $UPSTREAM_SESSION"
while tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; do
  sleep 60
done

log "waiting for all earlier GPU training workers to exit"
while pgrep -af '[t]rain_fetch_pick_and_place_lite_v0.py|[t]rain_drawer_open_lite_v0.py' >/dev/null 2>&1; do
  sleep 60
done

log "running source/config/unit gates"
"$PYTHON" "$ROOT/validate_config_parity.py" --output "$RESULT_ROOT/config_parity.json"
"$PYTHON" "$ROOT/validate_source_parity.py" --output "$RESULT_ROOT/source_parity.json"
"$PYTHON" "$ROOT/audit_checkpoint_resume.py" --output "$RESULT_ROOT/checkpoint_resume_audit.json"
"$PYTHON" -m py_compile \
  "$ROOT/code/src/metacog_lite_v0/config.py" \
  "$ROOT/code/src/metacog_lite_v0/weights.py" \
  "$ROOT/code/src/metacog_lite_v0/numba_support_scan_drawer_open.py" \
  "$ROOT/code/src/metacog_lite_v0/assignment_diagnostics.py" \
  "$ROOT/code/src/metacog_lite_v0/trainer.py" \
  "$ROOT/code/src/train_drawer_open_lite_v0.py"
for test_file in \
  "$ROOT/code/src/test_strict_context_eligible_count_matched.py" \
  "$ROOT/code/src/test_strict_drawer_open_scanner.py" \
  "$ROOT/code/src/test_strict_ddpg_loss_scope.py" \
  "$ROOT/code/src/test_strict_config_resume_contract.py"; do
  "$PYTHON" "$test_file"
done
"$PYTHON" "$ROOT/run.py" --mode main2m --seeds 1 --validate-only

log "freezing strict source before any GPU run"
"$PYTHON" "$ROOT/freeze_source.py" --write
"$PYTHON" "$ROOT/freeze_source.py" --verify

log "running 5k correctness (seed 1)"
"$PYTHON" "$ROOT/run.py" --mode correctness5k
"$PYTHON" "$ROOT/validate_integration.py" --mode correctness5k

log "running 20k correctness smoke (seed 1)"
"$PYTHON" "$ROOT/run.py" --mode smoke20k_seed1
"$PYTHON" "$ROOT/validate_integration.py" --mode smoke20k_seed1

log "running 20k correctness smoke (seeds 1..5)"
"$PYTHON" "$ROOT/run.py" --mode smoke20k_seeds1_5
"$PYTHON" "$ROOT/validate_integration.py" --mode smoke20k_seeds1_5

"$PYTHON" "$ROOT/freeze_source.py" --verify

log "launching production strict control seeds 1..5"
"$PYTHON" "$ROOT/run.py" --mode main2m

log "validating completed production cohort"
"$PYTHON" "$ROOT/validate_completion.py" --strict-root "$RESULT_ROOT" --output "$RESULT_ROOT/completion_validation.json"

log "queue completed through production completion gate"

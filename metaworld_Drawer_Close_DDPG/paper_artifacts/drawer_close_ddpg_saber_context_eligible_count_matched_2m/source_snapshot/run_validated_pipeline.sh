#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
EXPERIMENT_REL="${ROOT#"$REPO_ROOT"/}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WAIT_FOR_TMUX_SESSION="${WAIT_FOR_TMUX_SESSION:-}"
NUMBA_RUNTIME="${NUMBA_RUNTIME:-/tmp/saber-numba-0.66.0}"
RESULT_ROOT="$REPO_ROOT/metaworld_Drawer_Close_DDPG/results_2m/DDPG-HER-B3-Strict-Context-Eligible-Count-Matched-2M"
ANALYSIS_ROOT="$RESULT_ROOT/analysis"
ARTIFACT_ROOT="$REPO_ROOT/metaworld_Drawer_Close_DDPG/paper_artifacts/drawer_close_ddpg_saber_context_eligible_count_matched_2m"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }

if [[ ! -x "$PYTHON" ]]; then
  log "python executable is missing: $PYTHON"
  exit 2
fi

if [[ ! -d "$NUMBA_RUNTIME" ]]; then
  log "pinned scanner runtime is missing: $NUMBA_RUNTIME"
  log "install it with: $PYTHON -m pip install --target $NUMBA_RUNTIME -r $ROOT/requirements-numba.txt"
  exit 2
fi
export PYTHONPATH="$NUMBA_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "$WAIT_FOR_TMUX_SESSION" ]]; then
  if ! tmux has-session -t "$WAIT_FOR_TMUX_SESSION" 2>/dev/null; then
    log "requested upstream tmux session is not present: $WAIT_FOR_TMUX_SESSION"
    exit 2
  fi
  log "waiting for tmux session: $WAIT_FOR_TMUX_SESSION"
  while tmux has-session -t "$WAIT_FOR_TMUX_SESSION" 2>/dev/null; do
    sleep 60
  done
fi

log "waiting for existing lite training workers to exit"
while pgrep -af '[t]rain_.*_lite_v0.py' >/dev/null 2>&1; do
  sleep 60
done

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- "$EXPERIMENT_REL")" ]]; then
  log "experiment directory is not clean; commit the final source before production"
  exit 2
fi

log "running DrawerClose source, config, RNG, and unit gates"
"$PYTHON" -c 'import llvmlite, numba, numpy; assert (numpy.__version__, numba.__version__, llvmlite.__version__) == ("2.3.5", "0.66.0", "0.48.0")'
"$PYTHON" "$ROOT/validate_config_parity.py" --output "$ROOT/config_parity.json"
"$PYTHON" "$ROOT/validate_source_parity.py" --output "$ROOT/source_parity.json"
"$PYTHON" "$ROOT/audit_checkpoint_resume.py" --output "$ROOT/checkpoint_resume_audit.json"
"$PYTHON" -m compileall -q "$ROOT"
"$PYTHON" "$ROOT/test_config_parity.py"
for test_file in \
  "$ROOT/code/src/test_strict_context_eligible_count_matched.py" \
  "$ROOT/code/src/test_strict_drawer_close_scanner.py" \
  "$ROOT/code/src/test_strict_ddpg_loss_scope.py" \
  "$ROOT/code/src/test_strict_config_resume_contract.py"; do
  "$PYTHON" "$test_file"
done
"$PYTHON" "$ROOT/run.py" --mode main2m --seeds 1 --validate-only
"$PYTHON" "$ROOT/run.py" --mode main2m --dry-run

if [[ -f "$ROOT/freeze_manifest.json" ]]; then
  "$PYTHON" "$ROOT/freeze_source.py" --verify
else
  "$PYTHON" "$ROOT/freeze_source.py" --write
  "$PYTHON" "$ROOT/freeze_source.py" --verify
fi

log "running 5k correctness gate"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" "$ROOT/run.py" --mode correctness5k --cuda-device "$CUDA_DEVICE"
"$PYTHON" "$ROOT/validate_integration.py" --mode correctness5k

log "running 20k seed-1 smoke gate"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" "$ROOT/run.py" --mode smoke20k_seed1 --cuda-device "$CUDA_DEVICE"
"$PYTHON" "$ROOT/validate_integration.py" --mode smoke20k_seed1

log "running 20k seeds-1--5 smoke gate"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" "$ROOT/run.py" --mode smoke20k_seeds1_5 --cuda-device "$CUDA_DEVICE"
"$PYTHON" "$ROOT/validate_integration.py" --mode smoke20k_seeds1_5
"$PYTHON" "$ROOT/freeze_source.py" --verify

log "launching DrawerClose strict production seeds 1--5"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" "$ROOT/run.py" --mode main2m --cuda-device "$CUDA_DEVICE"

log "validating and analyzing the completed production cohort"
"$PYTHON" "$ROOT/validate_completion.py" --strict-root "$RESULT_ROOT" --output "$RESULT_ROOT/completion_validation.json"
"$PYTHON" "$ROOT/analyze_results.py" --strict-root "$RESULT_ROOT" --output-dir "$ANALYSIS_ROOT"

log "building and verifying the compact paper artifact"
"$PYTHON" "$ROOT/build_artifact.py" --strict-root "$RESULT_ROOT" --output-dir "$ARTIFACT_ROOT"
"$PYTHON" "$ROOT/verify_artifact.py" --artifact-dir "$ARTIFACT_ROOT"

log "DrawerClose strict pipeline completed successfully"

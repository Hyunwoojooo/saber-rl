#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
NUMBA_TARGET="${NUMBA_TARGET:-${XDG_CACHE_HOME:-$HOME/.cache}/saber-diversity-numba-0.66}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DEVICE="${DEVICE:-cuda}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

RUNS_ROOT="$REPO_ROOT/experiment_FPP/runs"
PROBE_ROOT="$RUNS_ROOT/td3_saber_diversity_probe_d1_50k"
PROBE_RUN="$PROBE_ROOT/fetch_pick_and_place_td3_saber_diversity_probe_d1_50k_s1_50000_ls512"
SELECTED_THRESHOLD="$PROBE_ROOT/selected_threshold.json"
ORCHESTRATOR_ROOT="$RUNS_ROOT/td3_saber_action_diversity_orchestration"
LOG="$ORCHESTRATOR_ROOT/orchestrator.log"
SOURCE_DIGEST_FILE="$ORCHESTRATOR_ROOT/source_sha256.txt"

mkdir -p "$ORCHESTRATOR_ROOT"
exec > >(tee -a "$LOG") 2>&1

timestamp() {
  date -u "+%Y-%m-%d %H:%M:%S UTC"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

fail() {
  log "ABORTED: $*"
  exit 1
}

on_exit() {
  status=$?
  if (( status == 0 )); then
    log "FINISHED: all staged actions completed"
  else
    log "FAILED: exit_status=$status"
  fi
}
trap on_exit EXIT

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$NUMBA_TARGET${PYTHONPATH:+:$PYTHONPATH}"

source_digest() {
  "$PYTHON" - "$SCRIPT_DIR" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
digest = hashlib.sha256()
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo", ".nbc", ".nbi"}:
        continue
    relative = path.relative_to(root).as_posix().encode("utf-8")
    digest.update(len(relative).to_bytes(8, "big"))
    digest.update(relative)
    data = path.read_bytes()
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)
print(digest.hexdigest())
PY
}

assert_source_unchanged() {
  current="$(source_digest)"
  if [[ "$current" != "$EXPECTED_SOURCE_DIGEST" ]]; then
    fail "source snapshot changed: expected=$EXPECTED_SOURCE_DIGEST current=$current"
  fi
}

validate_completed_run() {
  run_dir="$1"
  expected_steps="$2"
  expected_seed="$3"
  "$PYTHON" - "$run_dir" "$expected_steps" "$expected_seed" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected_steps = int(sys.argv[2])
expected_seed = int(sys.argv[3])
metrics_path = run_dir / "metrics.csv"
summary_path = run_dir / "summary.json"
config_path = run_dir / "config.json"
for path in (metrics_path, summary_path, config_path):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty required artifact: {path}")

config = json.loads(config_path.read_text(encoding="utf-8"))
if int(config.get("total_steps", -1)) != expected_steps:
    raise SystemExit(f"unexpected config total_steps in {config_path}")
if int(config.get("seeds", {}).get("env_seed", -1)) != expected_seed:
    raise SystemExit(f"unexpected env_seed in {config_path}")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
native_auc = float(summary.get("eval_AUC_success", float("nan")))
if not math.isfinite(native_auc):
    raise SystemExit(f"missing finite eval_AUC_success in {summary_path}")

final_values = []
with metrics_path.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        raw_step = row.get("step") or row.get("global_step") or row.get("env_step")
        if raw_step in (None, "") or int(float(raw_step)) != expected_steps:
            continue
        raw_value = row.get("eval/env_success_mean")
        if raw_value in (None, ""):
            raw_value = row.get("eval/success_mean")
        if raw_value not in (None, ""):
            value = float(raw_value)
            if math.isfinite(value):
                final_values.append(value)
if not final_values:
    raise SystemExit(f"missing finite final evaluation at step {expected_steps}: {metrics_path}")
if any(not math.isclose(value, final_values[0], rel_tol=0.0, abs_tol=1e-12) for value in final_values[1:]):
    raise SystemExit(f"final evaluation rows disagree: {metrics_path}")
print(f"VALIDATED: run={run_dir} step={expected_steps} seed={expected_seed} final={final_values[0]:.6f}")
PY
}

ensure_run() {
  condition="$1"
  seed="$2"
  expected_steps="$3"
  run_dir="$4"
  selected_path="${5:-}"

  assert_source_unchanged
  if [[ -d "$run_dir" ]]; then
    if validate_completed_run "$run_dir" "$expected_steps" "$seed"; then
      log "REUSING: condition=$condition seed=$seed run=$run_dir"
      return
    fi
    fail "partial or invalid canonical run exists; refusing timestamped replacement: $run_dir"
  fi

  log "STARTING: condition=$condition seed=$seed steps=$expected_steps"
  args=(
    "$PYTHON" "$SCRIPT_DIR/run.py"
    --condition "$condition"
    --seed "$seed"
    --device "$DEVICE"
  )
  if [[ -n "$selected_path" ]]; then
    args+=(--selected-threshold "$selected_path")
  fi
  "${args[@]}"
  validate_completed_run "$run_dir" "$expected_steps" "$seed"
  log "COMPLETED: condition=$condition seed=$seed"
}

[[ -x "$PYTHON" ]] || fail "missing Python environment: $PYTHON"
[[ -d "$NUMBA_TARGET" ]] || fail "missing Numba target: $NUMBA_TARGET"

"$PYTHON" - <<'PY'
import numba
import numpy
import torch

if not torch.cuda.is_available():
    raise SystemExit("Torch CUDA is unavailable")
print(f"PREFLIGHT: numpy={numpy.__version__} numba={numba.__version__} torch={torch.__version__}")
print(f"PREFLIGHT: cuda_device={torch.cuda.get_device_name(0)}")
PY

EXPECTED_SOURCE_DIGEST="$(source_digest)"
printf '%s\n' "$EXPECTED_SOURCE_DIGEST" > "$SOURCE_DIGEST_FILE"
log "SOURCE: digest=$EXPECTED_SOURCE_DIGEST head=$(git -C "$REPO_ROOT" rev-parse HEAD)"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  log "PREFLIGHT_ONLY: no training was started"
  exit 0
fi

ensure_run "probe" 1 50000 "$PROBE_RUN"
assert_source_unchanged

log "SELECTING: Dmin from exact 20k-50k probe counters"
"$PYTHON" "$SCRIPT_DIR/select_diversity_threshold.py" \
  --metrics "$PROBE_RUN/metrics.csv" \
  --output "$SELECTED_THRESHOLD"

SELECTED_DMIN="$("$PYTHON" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["selected_dmin"]))' "$SELECTED_THRESHOLD")"
case "$SELECTED_DMIN" in
  2|4|8) ;;
  *) fail "selector returned invalid Dmin=$SELECTED_DMIN" ;;
esac
log "SELECTED: Dmin=$SELECTED_DMIN artifact=$SELECTED_THRESHOLD"

DIVERSE_SLUG="td3_saber_diverse_d${SELECTED_DMIN}_critic_only_v4"
DIVERSE_ROOT="$RUNS_ROOT/${DIVERSE_SLUG}_ls512"
for seed in 1 2 3 4 5; do
  run_dir="$DIVERSE_ROOT/fetch_pick_and_place_${DIVERSE_SLUG}_s${seed}_1000000_ls512"
  ensure_run "diverse" "$seed" 1000000 "$run_dir" "$SELECTED_THRESHOLD"
done

RANDOM_SLUG="td3_saber_random_diversity_count_d${SELECTED_DMIN}_critic_only_v4"
RANDOM_ROOT="$RUNS_ROOT/${RANDOM_SLUG}_ls512"
for seed in 1 2 3 4 5; do
  run_dir="$RANDOM_ROOT/fetch_pick_and_place_${RANDOM_SLUG}_s${seed}_1000000_ls512"
  ensure_run "random_count" "$seed" 1000000 "$run_dir" "$SELECTED_THRESHOLD"
done

assert_source_unchanged
REPORT_ROOT="$REPO_ROOT/experiment_FPP/reports/td3_saber_action_diversity_d${SELECTED_DMIN}"
log "ANALYZING: output=$REPORT_ROOT"
"$PYTHON" "$SCRIPT_DIR/analyze_results.py" \
  --repo-root "$REPO_ROOT" \
  --diverse-root "$DIVERSE_ROOT" \
  --random-count-root "$RANDOM_ROOT" \
  --output-dir "$REPORT_ROOT"
log "COMPLETE: report=$REPORT_ROOT/report.md"

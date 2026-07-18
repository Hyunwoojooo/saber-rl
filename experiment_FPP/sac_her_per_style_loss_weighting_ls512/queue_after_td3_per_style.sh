#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

TD3_SESSION="${TD3_SESSION:-fpp_td3_per_style_1m}"
QUEUE_SESSION="${QUEUE_SESSION:-fpp_sac_per_style_after_td3}"
SAC_SESSION="${SAC_SESSION:-fpp_sac_per_style_1m}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DEVICE="${DEVICE:-cuda:0}"
STEPS=1000000
EXPECTED_UPDATES=995001
SEEDS=(1 2 3 4 5)

TD3_OUTPUT_ROOT="$REPO_ROOT/experiment_FPP/runs/td3_her_per_style_loss_weighting_ls512"
TD3_LAUNCH_LOG="$TD3_OUTPUT_ROOT/launcher_logs/launcher.log"
SAC_OUTPUT_ROOT="$REPO_ROOT/experiment_FPP/runs/sac_her_per_style_loss_weighting_ls512"
SAC_LOG_ROOT="$REPO_ROOT/experiment_FPP/logs/sac_her_per_style_loss_weighting_ls512"
SAC_LAUNCHER="$SCRIPT_DIR/run_main_1m_seeds1_5_5parallel.sh"
QUEUE_LOG="$SAC_LOG_ROOT/queue_after_td3_per_style.log"
LOCK_FILE="/tmp/saber-rl-fpp-sac-per-style-after-td3.lock"
CLAIM_DIR="$SAC_OUTPUT_ROOT/.launch_claim"

mkdir -p "$SAC_LOG_ROOT"
exec >>"$QUEUE_LOG" 2>&1

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

for command_name in tmux flock rg pgrep; do
  command -v "$command_name" >/dev/null 2>&1 \
    || fail "required command not found: $command_name"
done
[[ -x "$PYTHON_BIN" ]] || fail "python environment not found: $PYTHON_BIN"
[[ -x "$SAC_LAUNCHER" ]] || fail "SAC launcher is not executable: $SAC_LAUNCHER"
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] \
  || fail "POLL_SECONDS must be a positive integer: $POLL_SECONDS"

exec 9>"$LOCK_FILE"
flock -n 9 || fail "another SAC PER-style queue already holds $LOCK_FILE"

log "queue armed: wait_session=$TD3_SESSION target_session=$SAC_SESSION"
log "source_commit=c95952c steps=$STEPS seeds=${SEEDS[*]} device=$DEVICE"

while tmux has-session -t "$TD3_SESSION" 2>/dev/null; do
  log "waiting for TD3 session: $TD3_SESSION"
  sleep "$POLL_SECONDS"
done

log "TD3 session ended; validating successful completion"
[[ -s "$TD3_LAUNCH_LOG" ]] || fail "missing TD3 launcher log: $TD3_LAUNCH_LOG"
last_launcher_status="$(rg 'all seeds exited aggregate_status=' "$TD3_LAUNCH_LOG" | tail -n 1 || true)"
[[ "$last_launcher_status" == *"all seeds exited aggregate_status=0" ]] \
  || fail "TD3 launcher did not finish successfully: ${last_launcher_status:-missing status}"

TD3_TRAINER_PATTERN='td3_her_per_style_loss_weighting_ls512/code/src/train_fetch_pick_and_place_lite_v0.py'
if pgrep -af "$TD3_TRAINER_PATTERN" >/dev/null 2>&1; then
  pgrep -af "$TD3_TRAINER_PATTERN" || true
  fail "TD3 trainer process still exists after session exit"
fi

for seed in "${SEEDS[@]}"; do
  run_dir="$TD3_OUTPUT_ROOT/fetch_pick_and_place_td3_her_per_style_loss_weighting_ls512_s${seed}_${STEPS}_ls512"
  summary="$run_dir/summary.json"
  metrics="$run_dir/metrics.csv"
  run_log="$run_dir/run.log"

  [[ -s "$summary" ]] || fail "missing TD3 seed $seed summary: $summary"
  [[ -s "$metrics" ]] || fail "missing TD3 seed $seed metrics: $metrics"
  [[ -s "$run_log" ]] || fail "missing TD3 seed $seed run log: $run_log"
  [[ -s "$run_dir/checkpoint.pt" ]] \
    || fail "missing TD3 seed $seed rolling checkpoint"
  [[ -s "$run_dir/checkpoint_step_${STEPS}.pt" ]] \
    || fail "missing TD3 seed $seed final checkpoint"
  rg -F -q 'finished run at' "$run_log" \
    || fail "TD3 seed $seed run log lacks finished marker"

  "$PYTHON_BIN" - "$summary" "$metrics" "$seed" "$STEPS" "$EXPECTED_UPDATES" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
seed = int(sys.argv[3])
expected_steps = int(sys.argv[4])
expected_updates = int(sys.argv[5])

summary = json.loads(summary_path.read_text(encoding="utf-8"))
checks = {
    "method": summary.get("method") == "td3_her_per_style_loss_weighting",
    "total_steps": int(summary.get("total_steps", -1)) == expected_steps,
    "total_updates": int(summary.get("total_updates", -1)) == expected_updates,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(
        f"TD3 seed {seed} summary mismatch: {failed}; "
        f"method={summary.get('method')!r} "
        f"steps={summary.get('total_steps')!r} "
        f"updates={summary.get('total_updates')!r}"
    )

last_step = None
with metrics_path.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        value = row.get("step")
        if value not in (None, ""):
            last_step = int(float(value))
if last_step != expected_steps:
    raise SystemExit(
        f"TD3 seed {seed} metrics final step mismatch: "
        f"{last_step!r} != {expected_steps}"
    )
PY
  log "validated TD3 seed=$seed"
done

git -C "$REPO_ROOT" merge-base --is-ancestor c95952c HEAD \
  || fail "current HEAD no longer contains source commit c95952c"
git -C "$REPO_ROOT" diff --quiet -- "$SCRIPT_DIR" \
  || fail "tracked SAC source has uncommitted worktree changes"
git -C "$REPO_ROOT" diff --cached --quiet -- "$SCRIPT_DIR" \
  || fail "tracked SAC source has staged changes"

if tmux has-session -t "$SAC_SESSION" 2>/dev/null; then
  fail "target tmux session already exists: $SAC_SESSION"
fi

for seed in "${SEEDS[@]}"; do
  target="$SAC_OUTPUT_ROOT/fetch_pick_and_place_sac_her_per_style_loss_weighting_ls512_s${seed}_${STEPS}"
  [[ ! -e "$target" ]] || fail "refusing to overwrite SAC seed $seed: $target"
done

mkdir -p "$SAC_OUTPUT_ROOT"
mkdir "$CLAIM_DIR" 2>/dev/null \
  || fail "SAC launch was already claimed: $CLAIM_DIR"
printf 'source_commit=c95952c\nqueued_at=%s\nsteps=%s\nseeds=%s\ndevice=%s\n' \
  "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" \
  "$STEPS" "${SEEDS[*]}" "$DEVICE" >"$CLAIM_DIR/metadata.txt"

log "TD3 completion gate passed; starting SAC PER-style seeds ${SEEDS[*]}"
tmux rename-session -t "$QUEUE_SESSION" "$SAC_SESSION" \
  || fail "could not rename queue session $QUEUE_SESSION to $SAC_SESSION"
log "session renamed to $SAC_SESSION; executing SAC launcher"
exec env DEVICE="$DEVICE" bash "$SAC_LAUNCHER"

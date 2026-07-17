#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TD3_SESSION="${TD3_SESSION:-fpp_td3_per_full_1m}"
SAC_SESSION="${SAC_SESSION:-fpp_sac_per_full_1m}"
POLL_SECONDS="${POLL_SECONDS:-60}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"

TD3_RESULT_ROOT="$REPO_ROOT/experiment_FPP/runs/td3_her_per_full_ls512"
SAC_RESULT_ROOT="$REPO_ROOT/experiment_FPP/runs/sac_her_per_full_ls512"
SAC_LOG_ROOT="$SAC_RESULT_ROOT/launcher_logs"
SAC_RUNNER="$REPO_ROOT/experiment_FPP/sac_her_per_full_ls512/run.py"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

mkdir -p "$SAC_LOG_ROOT"
QUEUE_LOG="$SAC_LOG_ROOT/queue_after_td3.log"
exec >>"$QUEUE_LOG" 2>&1

log() {
  printf '%s | %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"
}

td3_summary_path() {
  local seed="$1"
  printf '%s/fetch_pick_and_place_td3_her_per_full_ls512_s%s_%s_ls512/summary.json' \
    "$TD3_RESULT_ROOT" "$seed" "$TOTAL_STEPS"
}

sac_run_dir() {
  local seed="$1"
  printf '%s/fetch_pick_and_place_sac_her_per_full_ls512_s%s_%s_ls512' \
    "$SAC_RESULT_ROOT" "$seed" "$TOTAL_STEPS"
}

summary_is_complete() {
  local summary="$1"
  [[ -s "$summary" ]] || return 1
  "$PYTHON_BIN" -c \
    'import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
ok = (
    payload.get("method") == "td3_her_per_full"
    and int(payload.get("total_steps", -1)) == int(sys.argv[2])
    and int(payload.get("total_updates", 0)) > 0
)
raise SystemExit(0 if ok else 1)' \
    "$summary" "$TOTAL_STEPS"
}

log "queue started; waiting for TD3 seeds 1-5 at ${TOTAL_STEPS} steps"
last_complete=-1

while true; do
  complete=0
  live_windows=""
  if tmux has-session -t "$TD3_SESSION" 2>/dev/null; then
    live_windows="$(tmux list-windows -t "$TD3_SESSION" -F '#{window_name}')"
  fi

  for seed in 1 2 3 4 5; do
    summary="$(td3_summary_path "$seed")"
    if summary_is_complete "$summary"; then
      complete=$((complete + 1))
      continue
    fi

    if ! grep -qx "seed${seed}" <<<"$live_windows"; then
      # Recheck after observing the window exit so a just-written summary wins the race.
      if summary_is_complete "$summary"; then
        complete=$((complete + 1))
      else
        log "ERROR: TD3 seed ${seed} stopped without a valid completion summary; SAC was not launched"
        exit 1
      fi
    fi
  done

  if (( complete != last_complete )); then
    log "TD3 completion progress: ${complete}/5"
    last_complete="$complete"
  fi
  if (( complete == 5 )); then
    break
  fi
  sleep "$POLL_SECONDS"
done

if tmux has-session -t "$SAC_SESSION" 2>/dev/null; then
  log "ERROR: SAC session ${SAC_SESSION} already exists; refusing duplicate launch"
  exit 1
fi

for seed in 1 2 3 4 5; do
  run_dir="$(sac_run_dir "$seed")"
  if [[ -e "$run_dir" ]]; then
    log "ERROR: SAC seed ${seed} output already exists at ${run_dir}; refusing overwrite"
    exit 1
  fi
done

log "all TD3 seeds completed normally; launching SAC seeds 1-5 in parallel"
for seed in 1 2 3 4 5; do
  log_file="$SAC_LOG_ROOT/seed${seed}.log"
  command="env PYTHONUNBUFFERED=1 \"$PYTHON_BIN\" \"$SAC_RUNNER\" --seed $seed --steps $TOTAL_STEPS --device cuda > \"$log_file\" 2>&1"
  if (( seed == 1 )); then
    tmux new-session -d -s "$SAC_SESSION" -n "seed${seed}" -c "$REPO_ROOT" "$command"
  else
    tmux new-window -d -t "$SAC_SESSION" -n "seed${seed}" -c "$REPO_ROOT" "$command"
  fi
  log "launched SAC seed ${seed}; log=${log_file}"
done

log "SAC parallel session ready: ${SAC_SESSION}"

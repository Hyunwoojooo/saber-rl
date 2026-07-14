#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="$REPO_ROOT/experiment_FPP/runs/td3_saber_context_eligible_count_matched_ls512"
ORCHESTRATION_ROOT="$REPO_ROOT/experiment_FPP/runs/td3_saber_context_eligible_count_matched_orchestration"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$ORCHESTRATION_ROOT"
printf 'seed,pid,exit_code\n' > "$ORCHESTRATION_ROOT/status.csv"

declare -a pids=()
declare -a seeds=()

for seed in 1 2 3 4 5; do
  expected="$OUTPUT_ROOT/fetch_pick_and_place_td3_saber_context_eligible_count_matched_s${seed}_1000000_ls512"
  if [[ -e "$expected" ]]; then
    printf 'Refusing to overwrite existing production run: %s\n' "$expected" >&2
    exit 2
  fi
  log="$ORCHESTRATION_ROOT/seed_${seed}.log"
  printf 'starting seed=%s log=%s\n' "$seed" "$log"
  "$PYTHON_BIN" "$SCRIPT_DIR/run.py" \
    --stage production \
    --seed "$seed" \
    --device "$DEVICE" >"$log" 2>&1 &
  pids+=("$!")
  seeds+=("$seed")
done

failed=0
for index in "${!pids[@]}"; do
  seed="${seeds[$index]}"
  pid="${pids[$index]}"
  if wait "$pid"; then
    status=0
  else
    status=$?
    failed=1
  fi
  printf '%s,%s,%s\n' "$seed" "$pid" "$status" >> "$ORCHESTRATION_ROOT/status.csv"
done

exit "$failed"

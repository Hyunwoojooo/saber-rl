#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {k3|k7|both} [run.py args...]" >&2
  exit 2
fi
VARIANT="$1"
shift
case "$VARIANT" in
  k3|k7|both) ;;
  *)
    echo "variant must be k3, k7, or both" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="${PYTHON_BIN:-python3}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"

if [[ ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer; got $MAX_PARALLEL" >&2
  exit 2
fi
if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
  echo "Python launcher not found: $PYTHON" >&2
  exit 2
fi

"$PYTHON" -B "$SCRIPT_DIR/test_experiment_contract.py"

variants=()
case "$VARIANT" in
  k3) variants=(k3) ;;
  k7) variants=(k7) ;;
  both) variants=(k3 k7) ;;
esac

# Interleave K=3 and K=7 by seed so a combined launch does not privilege one
# variant when fewer than ten worker slots are available.
jobs=()
for seed in 1 2 3 4 5; do
  for variant in "${variants[@]}"; do
    jobs+=("${variant}:${seed}")
  done
done

for job in "${jobs[@]}"; do
  "$PYTHON" -B "$SCRIPT_DIR/run.py" \
    --variant "${job%%:*}" \
    --seed "${job##*:}" \
    --dry-run \
    "$@" >/dev/null
done

timestamp="$(date -u +'%Y%m%dT%H%M%SZ')"
EXECUTION_ROOT="${EXECUTION_ROOT:-${REPO_ROOT}/experiment_FPP/runs/td3_saber_action_bin_sensitivity_ls512_1m/_execution}"
EXECUTION_DIR="${EXECUTION_ROOT}/${VARIANT}_${timestamp}"
LOG_DIR="${EXECUTION_DIR}/logs"
mkdir -p "$LOG_DIR"

printf '%s\n' \
  "timestamp_utc=${timestamp}" \
  "repo_head=$(git -C "$REPO_ROOT" rev-parse HEAD)" \
  "variant=${VARIANT}" \
  "seeds=1,2,3,4,5" \
  "steps=1000000" \
  "max_parallel=${MAX_PARALLEL}" \
  "runner_python=${PYTHON}" \
  "saber_python=${SABER_PYTHON:-auto}" \
  "extra_args=$*" >"${EXECUTION_DIR}/launch_metadata.txt"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

declare -A LABEL_BY_PID=()
running=0
failed=0

reap_one() {
  local finished_pid=""
  local status=0
  if wait -n -p finished_pid; then
    status=0
  else
    status=$?
    failed=1
  fi
  local label="${LABEL_BY_PID[${finished_pid}]:-unknown}"
  if (( status == 0 )); then
    echo "Completed ${label}."
  else
    echo "Failed ${label} with status ${status}; inspect ${LOG_DIR}/${label}.log" >&2
  fi
  unset "LABEL_BY_PID[${finished_pid}]"
  running=$((running - 1))
}

launch_job() {
  local job="$1"
  shift
  local variant="${job%%:*}"
  local seed="${job##*:}"
  local label="${variant}_seed_${seed}"
  local log_path="${LOG_DIR}/${label}.log"
  echo "Launching ${label}; log: ${log_path}"
  (
    "$PYTHON" -B "$SCRIPT_DIR/run.py" \
      --variant "$variant" \
      --seed "$seed" \
      "$@"
  ) >"$log_path" 2>&1 &
  local pid=$!
  LABEL_BY_PID["$pid"]="$label"
  running=$((running + 1))
}

for job in "${jobs[@]}"; do
  while (( running >= MAX_PARALLEL )); do
    reap_one
  done
  launch_job "$job" "$@"
done

while (( running > 0 )); do
  reap_one
done

if (( failed != 0 )); then
  exit 1
fi

echo "All ${VARIANT} action-bin sensitivity runs completed."

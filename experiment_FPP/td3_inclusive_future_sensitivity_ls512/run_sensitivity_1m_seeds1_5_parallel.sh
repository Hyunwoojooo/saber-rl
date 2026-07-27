#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
PYTHON="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" && -x "${REPO_ROOT}/../../.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/../../.venv/bin/python"
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "No experiment Python found; set PYTHON_BIN." >&2
  exit 2
fi

DEVICE="${DEVICE:-cuda}"
MAX_PARALLEL="${MAX_PARALLEL:-7}"
STEPS=1000000
OUTPUT_ROOT="${REPO_ROOT}/experiment_FPP/runs/td3_inclusive_future_sensitivity_ls512"
EXECUTION_DIR="${OUTPUT_ROOT}/_execution"
LOG_DIR="${EXECUTION_DIR}/logs"

if [[ ! "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer; got ${MAX_PARALLEL}." >&2
  exit 2
fi

"${PYTHON}" -B "${SCRIPT_DIR}/validate_experiment.py"
"${PYTHON}" -B "${SCRIPT_DIR}/test_inclusive_future_her.py"

for arm in her saber; do
  "${PYTHON}" -B "${SCRIPT_DIR}/run.py" \
    --arm "${arm}" --seed 1 --steps "${STEPS}" --device "${DEVICE}" --dry-run >/dev/null
done

for arm in her saber; do
  slug="td3_her_inclusive_future_ls512"
  if [[ "${arm}" == "saber" ]]; then
    slug="td3_saber_inclusive_future_v4_ls512"
  fi
  for seed in 1 2 3 4 5; do
    target="${OUTPUT_ROOT}/fetch_pick_and_place_${slug}_s${seed}_${STEPS}_ls512"
    if [[ -e "${target}" ]]; then
      echo "Refusing to overwrite existing run: ${target}" >&2
      exit 2
    fi
  done
done

mkdir -p "${LOG_DIR}"
REPO_HEAD="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
ORIGIN_MAIN="$(git -C "${REPO_ROOT}" rev-parse origin/main)"
SOURCE_BLOB="$(git -C "${REPO_ROOT}" hash-object "${SCRIPT_DIR}/code/src/metacog_lite_v0/replay.py")"
TIMESTAMP_UTC="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
printf '%s\n' \
  "timestamp_utc=${TIMESTAMP_UTC}" \
  "repo_head=${REPO_HEAD}" \
  "origin_main=${ORIGIN_MAIN}" \
  "inclusive_replay_blob=${SOURCE_BLOB}" \
  "steps=${STEPS}" \
  "seeds=1,2,3,4,5" \
  "max_parallel=${MAX_PARALLEL}" \
  "device=${DEVICE}" >"${EXECUTION_DIR}/launch_metadata.txt"

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
  local arm="$1"
  local seed="$2"
  local label="${arm}_seed_${seed}"
  local log_path="${LOG_DIR}/${label}.log"
  echo "Launching ${label}; log: ${log_path}"
  (
    PYTHON_BIN="${PYTHON}" "${PYTHON}" -B "${SCRIPT_DIR}/run.py" \
      --arm "${arm}" \
      --seed "${seed}" \
      --steps "${STEPS}" \
      --device "${DEVICE}" \
      --output_root "${OUTPUT_ROOT}"
  ) >"${log_path}" 2>&1 &
  local pid=$!
  LABEL_BY_PID["${pid}"]="${label}"
  running=$((running + 1))
}

jobs=(
  saber:1 saber:2 saber:3 saber:4 saber:5
  her:1 her:2 her:3 her:4 her:5
)

for job in "${jobs[@]}"; do
  while (( running >= MAX_PARALLEL )); do
    reap_one
  done
  launch_job "${job%%:*}" "${job##*:}"
done

while (( running > 0 )); do
  reap_one
done

if (( failed != 0 )); then
  exit 1
fi

echo "All paired inclusive-future FPP TD3 runs completed."

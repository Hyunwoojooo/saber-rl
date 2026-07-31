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
PREPARE_ONLY="${PREPARE_ONLY:-0}"
STEPS=1000000
SEEDS=(6 7 8 9 10)
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiment_FPP/runs/td3_inclusive_future_sensitivity_ls512_seeds6_10}"
EXECUTION_DIR="${OUTPUT_ROOT}/_execution"
LOG_DIR="${EXECUTION_DIR}/logs"
SOURCE_REFERENCE_COMMIT="47343f4cb295f354df16733dca17f3a882a2b34a"

if [[ ! "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer; got ${MAX_PARALLEL}." >&2
  exit 2
fi
if [[ "${PREPARE_ONLY}" != "0" && "${PREPARE_ONLY}" != "1" ]]; then
  echo "PREPARE_ONLY must be 0 or 1; got ${PREPARE_ONLY}." >&2
  exit 2
fi

"${PYTHON}" -B "${SCRIPT_DIR}/validate_experiment.py"
"${PYTHON}" -B "${SCRIPT_DIR}/test_inclusive_future_her.py"

for arm in her saber; do
  for seed in "${SEEDS[@]}"; do
    "${PYTHON}" -B "${SCRIPT_DIR}/run.py" \
      --arm "${arm}" \
      --seed "${seed}" \
      --steps "${STEPS}" \
      --device "${DEVICE}" \
      --output_root "${OUTPUT_ROOT}" \
      --dry-run >/dev/null
  done
done

for arm in her saber; do
  slug="td3_her_inclusive_future_ls512"
  if [[ "${arm}" == "saber" ]]; then
    slug="td3_saber_inclusive_future_v4_ls512"
  fi
  for seed in "${SEEDS[@]}"; do
    target="${OUTPUT_ROOT}/fetch_pick_and_place_${slug}_s${seed}_${STEPS}_ls512"
    if [[ -e "${target}" ]]; then
      echo "Refusing to overwrite existing run: ${target}" >&2
      exit 2
    fi
  done
done

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "Preparation checks passed; production was not started."
  echo "Seeds: ${SEEDS[*]}"
  echo "Paired arms: her saber"
  echo "Steps per run: ${STEPS}"
  echo "Maximum concurrent runs: ${MAX_PARALLEL}"
  echo "Output root: ${OUTPUT_ROOT}"
  exit 0
fi

mkdir -p "${LOG_DIR}"
REPO_HEAD="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
ORIGIN_MAIN="$(git -C "${REPO_ROOT}" rev-parse origin/main)"
SOURCE_BLOB="$(git -C "${REPO_ROOT}" hash-object "${SCRIPT_DIR}/code/src/metacog_lite_v0/replay.py")"
LAUNCHER_BLOB="$(git -C "${REPO_ROOT}" hash-object "${BASH_SOURCE[0]}")"
TIMESTAMP_UTC="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
printf '%s\n' \
  "timestamp_utc=${TIMESTAMP_UTC}" \
  "repo_head=${REPO_HEAD}" \
  "origin_main=${ORIGIN_MAIN}" \
  "source_reference_commit=${SOURCE_REFERENCE_COMMIT}" \
  "inclusive_replay_blob=${SOURCE_BLOB}" \
  "launcher_blob=${LAUNCHER_BLOB}" \
  "steps=${STEPS}" \
  "seeds=6,7,8,9,10" \
  "paired_arms=her,saber" \
  "max_parallel=${MAX_PARALLEL}" \
  "device=${DEVICE}" \
  "output_root=${OUTPUT_ROOT}" >"${EXECUTION_DIR}/launch_metadata.txt"

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
  saber:6 saber:7 saber:8 saber:9 saber:10
  her:6 her:7 her:8 her:9 her:10
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

echo "All paired inclusive-future FPP TD3 seeds 6-10 runs completed."

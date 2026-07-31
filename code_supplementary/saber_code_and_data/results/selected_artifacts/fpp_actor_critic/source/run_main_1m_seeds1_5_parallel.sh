#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"; then
  echo "Unable to locate the repository root from ${SCRIPT_DIR}." >&2
  exit 2
fi

DEVICE="${DEVICE:-cuda}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
STEPS=1000000
SEEDS=(1 2 3 4 5)
MAIN_ROOT="${REPO_ROOT}/experiment_FPP/runs/ddpg_saber_actor_critic_v4_ls512"
EXECUTION_DIR="${MAIN_ROOT}/_execution"
METADATA_PATH="${EXECUTION_DIR}/launch_metadata.json"

if [[ ! "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer; got ${MAX_PARALLEL}." >&2
  exit 2
fi

resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${PYTHON_BIN}" ]]; then
      printf '%s\n' "${PYTHON_BIN}"
      return
    fi
    if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      command -v "${PYTHON_BIN}"
      return
    fi
    echo "PYTHON_BIN is not executable or available on PATH: ${PYTHON_BIN}" >&2
    exit 2
  fi
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    printf '%s\n' "${REPO_ROOT}/.venv/bin/python"
    return
  fi
  if [[ -x "${REPO_ROOT}/../.venv/bin/python" ]]; then
    printf '%s\n' "${REPO_ROOT}/../.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "No Python interpreter found; set PYTHON_BIN explicitly." >&2
  exit 2
}

PYTHON="$(resolve_python)"

"${PYTHON}" -B "${SCRIPT_DIR}/validate_config_parity.py"

for seed in "${SEEDS[@]}"; do
  run_name="fetch_pick_and_place_ddpg_saber_actor_critic_v4_ls512_s${seed}_${STEPS}_ls512"
  target_run="${MAIN_ROOT}/${run_name}"
  if [[ -e "${target_run}" ]]; then
    echo "Refusing to start: target run already exists: ${target_run}" >&2
    exit 2
  fi
done

if ! repository_snapshot_id="$(git -C "${REPO_ROOT}" rev-parse HEAD)"; then
  echo "Unable to resolve repository HEAD." >&2
  exit 2
fi
if ! canonical_snapshot_id="$(git -C "${REPO_ROOT}" rev-parse origin/main)"; then
  echo "Unable to resolve origin/main; update repository refs before launching." >&2
  exit 2
fi
if ! WORKTREE_STATUS="$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)"; then
  echo "Unable to inspect the repository worktree." >&2
  exit 2
fi
if [[ -z "${WORKTREE_STATUS}" ]]; then
  WORKTREE_CLEAN=true
else
  WORKTREE_CLEAN=false
fi
if [[ "${repository_snapshot_id}" == "${canonical_snapshot_id}" ]]; then
  HEAD_MATCHES_canonical_snapshot_id=true
else
  HEAD_MATCHES_canonical_snapshot_id=false
fi
TIMESTAMP_UTC="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

mkdir -p "${EXECUTION_DIR}"
METADATA_TMP="${METADATA_PATH}.tmp.$$"
{
  printf '{\n'
  printf '  "schema_version": 1,\n'
  printf '  "experiment": "ddpg_saber_actor_critic_v4_ls512",\n'
  printf '  "timestamp_utc": "%s",\n' "${TIMESTAMP_UTC}"
  printf '  "repository_snapshot_id": "%s",\n' "${repository_snapshot_id}"
  printf '  "canonical_snapshot_id": "%s",\n' "${canonical_snapshot_id}"
  printf '  "head_matches_canonical_snapshot_id": %s,\n' "${HEAD_MATCHES_canonical_snapshot_id}"
  printf '  "worktree_clean": %s,\n' "${WORKTREE_CLEAN}"
  printf '  "steps": %s,\n' "${STEPS}"
  printf '  "seeds": [1, 2, 3, 4, 5],\n'
  printf '  "max_parallel": %s\n' "${MAX_PARALLEL}"
  printf '}\n'
} >"${METADATA_TMP}"
mv "${METADATA_TMP}" "${METADATA_PATH}"

if [[ "${HEAD_MATCHES_canonical_snapshot_id}" != true ]]; then
  echo "Refusing to launch: HEAD (${repository_snapshot_id}) does not match origin/main (${canonical_snapshot_id})." >&2
  echo "Launch metadata recorded at ${METADATA_PATH}." >&2
  exit 2
fi
if [[ "${WORKTREE_CLEAN}" != true ]]; then
  echo "Refusing to launch: the repository worktree was not clean at preflight." >&2
  echo "Launch metadata recorded at ${METADATA_PATH}." >&2
  exit 2
fi

LOG_DIR="${EXECUTION_DIR}/logs"
mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

pids=()
pid_seeds=()

wait_batch() {
  local failed=0
  local index
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "Seed ${pid_seeds[$index]} completed."
    else
      echo "Seed ${pid_seeds[$index]} failed; see ${LOG_DIR}/seed_${pid_seeds[$index]}.log" >&2
      failed=1
    fi
  done
  pids=()
  pid_seeds=()
  return "${failed}"
}

for seed in "${SEEDS[@]}"; do
  log_path="${LOG_DIR}/seed_${seed}.log"
  echo "Launching seed ${seed}; log: ${log_path}"
  (
    PYTHON_BIN="${PYTHON}" "${PYTHON}" -B "${SCRIPT_DIR}/run.py" \
      --seed "${seed}" \
      --steps "${STEPS}" \
      --device "${DEVICE}" \
      --output_root "${MAIN_ROOT}"
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
  pid_seeds+=("${seed}")

  if (( ${#pids[@]} >= MAX_PARALLEL )); then
    if ! wait_batch; then
      exit 1
    fi
  fi
done

if (( ${#pids[@]} > 0 )); then
  if ! wait_batch; then
    exit 1
  fi
fi

"${PYTHON}" -B "${SCRIPT_DIR}/validate_results.py" \
  --output-root "${MAIN_ROOT}" \
  --steps "${STEPS}" \
  --seeds "${SEEDS[@]}" \
  --launcher-log-dir "${LOG_DIR}" \
  --write-selected-10k

echo "All five 1M runs and result validation passed."

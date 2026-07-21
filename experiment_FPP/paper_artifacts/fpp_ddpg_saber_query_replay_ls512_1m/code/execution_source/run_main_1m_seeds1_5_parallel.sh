#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"; then
  echo "Unable to locate repository root from ${SCRIPT_DIR}." >&2
  exit 2
fi

DEVICE="${DEVICE:-cuda}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
STEPS=1000000
SEEDS=(1 2 3 4 5)
MAIN_ROOT="${REPO_ROOT}/experiment_FPP/runs/ddpg_saber_query_replay_ls512"
EXECUTION_DIR="${MAIN_ROOT}/_execution"
METADATA_PATH="${EXECUTION_DIR}/launch_metadata.json"

if [[ ! "${MAX_PARALLEL}" =~ ^[12]$ ]]; then
  echo "MAX_PARALLEL is deliberately bounded to 1 or 2; got ${MAX_PARALLEL}." >&2
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
    echo "PYTHON_BIN is not executable or on PATH: ${PYTHON_BIN}" >&2
    exit 2
  fi
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    printf '%s\n' "${REPO_ROOT}/.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "No Python interpreter found; set PYTHON_BIN." >&2
  exit 2
}

PYTHON="$(resolve_python)"

"${PYTHON}" -B "${SCRIPT_DIR}/validate_config_parity.py"

for seed in "${SEEDS[@]}"; do
  run_name="fetch_pick_and_place_ddpg_saber_query_replay_ls512_s${seed}_${STEPS}_ls512"
  shopt -s nullglob
  matches=("${MAIN_ROOT}/${run_name}"*)
  shopt -u nullglob
  if (( ${#matches[@]} > 0 )); then
    echo "Refusing to start: existing exact/timestamped target for seed ${seed}: ${matches[*]}" >&2
    exit 2
  fi
done

if [[ -e "${METADATA_PATH}" ]]; then
  echo "Refusing to overwrite existing launch provenance: ${METADATA_PATH}" >&2
  exit 2
fi
if ! REPO_HEAD="$(git -C "${REPO_ROOT}" rev-parse HEAD)"; then
  echo "Unable to resolve repository HEAD." >&2
  exit 2
fi
if ! ORIGIN_MAIN="$(git -C "${REPO_ROOT}" rev-parse origin/main)"; then
  echo "Unable to resolve origin/main; fetch it before launching." >&2
  exit 2
fi
if ! WORKTREE_STATUS="$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)"; then
  echo "Unable to inspect worktree status." >&2
  exit 2
fi
if [[ -n "${WORKTREE_STATUS}" ]]; then
  echo "Refusing to launch: repository worktree is not clean." >&2
  printf '%s\n' "${WORKTREE_STATUS}" >&2
  exit 2
fi
if [[ "${REPO_HEAD}" != "${ORIGIN_MAIN}" ]]; then
  echo "Refusing to launch: HEAD ${REPO_HEAD} != origin/main ${ORIGIN_MAIN}." >&2
  exit 2
fi

SOURCE_TREE="$(git -C "${REPO_ROOT}" rev-parse "${REPO_HEAD}:experiment_FPP/ddpg_saber_query_replay_ls512/code/src")"
CONFIG_BLOB="$(git -C "${REPO_ROOT}" rev-parse "${REPO_HEAD}:experiment_FPP/ddpg_saber_query_replay_ls512/config.json")"
TIMESTAMP_UTC="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

mkdir -p "${EXECUTION_DIR}"
METADATA_TMP="${METADATA_PATH}.tmp.$$"
{
  printf '{\n'
  printf '  "schema_version": 1,\n'
  printf '  "experiment": "ddpg_saber_query_replay_ls512",\n'
  printf '  "timestamp_utc": "%s",\n' "${TIMESTAMP_UTC}"
  printf '  "source_commit": "%s",\n' "${REPO_HEAD}"
  printf '  "origin_main": "%s",\n' "${ORIGIN_MAIN}"
  printf '  "head_matches_origin_main": true,\n'
  printf '  "worktree_clean": true,\n'
  printf '  "source_tree_git_oid": "%s",\n' "${SOURCE_TREE}"
  printf '  "config_git_blob_oid": "%s",\n' "${CONFIG_BLOB}"
  printf '  "support_query_mode": "replay_action",\n'
  printf '  "weight_scope": "critic_only",\n'
  printf '  "steps": %s,\n' "${STEPS}"
  printf '  "seeds": [1, 2, 3, 4, 5],\n'
  printf '  "device": "%s",\n' "${DEVICE}"
  printf '  "max_parallel": %s\n' "${MAX_PARALLEL}"
  printf '}\n'
} >"${METADATA_TMP}"
mv "${METADATA_TMP}" "${METADATA_PATH}"

LOG_DIR="${EXECUTION_DIR}/logs"
mkdir -p "${LOG_DIR}"

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

echo "All five replay-action-query 1M runs and completion gates passed."

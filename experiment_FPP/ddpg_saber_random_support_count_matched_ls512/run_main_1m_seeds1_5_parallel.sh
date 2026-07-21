#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
PYTHON="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
DRY_RUN="${DRY_RUN:-1}"
STEPS=1000000
SEEDS=(1 2 3 4 5)
MAIN_ROOT="${REPO_ROOT}/experiment_FPP/runs/ddpg_saber_random_support_count_matched_ls512"
EXECUTION_DIR="${MAIN_ROOT}/_execution"
METADATA="${EXECUTION_DIR}/launch_metadata.json"
FINGERPRINT="${EXECUTION_DIR}/source_fingerprint.json"

if [[ ! "${MAX_PARALLEL}" =~ ^[1-5]$ ]]; then
  echo "MAX_PARALLEL must be an integer from 1 through 5; got ${MAX_PARALLEL}." >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1; got ${DRY_RUN}." >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 2
fi
"${PYTHON}" -B "${SCRIPT_DIR}/validate_config_parity.py" >/dev/null
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Production dry-run only; no trainer processes will be started."
  for seed in "${SEEDS[@]}"; do
    "${PYTHON}" -B "${SCRIPT_DIR}/run.py" \
      --stage production --seed "${seed}" --device "${DEVICE}" --dry-run
  done
  exit 0
fi
if [[ "${SABER_PRODUCTION_AUTHORIZED:-}" != "YES" ]]; then
  echo "Production is locked pending explicit user approval." >&2
  echo "After approval, set DRY_RUN=0 SABER_PRODUCTION_AUTHORIZED=YES." >&2
  exit 2
fi
"${PYTHON}" -B "${SCRIPT_DIR}/run.py" --stage production --seed 1 --device "${DEVICE}" --dry-run >/dev/null

for seed in "${SEEDS[@]}"; do
  name="fetch_pick_and_place_ddpg_saber_random_support_count_matched_ls512_s${seed}_${STEPS}_ls512"
  shopt -s nullglob
  matches=("${MAIN_ROOT}/${name}"*)
  shopt -u nullglob
  if (( ${#matches[@]} > 0 )); then
    echo "Refusing existing exact/timestamped seed ${seed} target: ${matches[*]}" >&2
    exit 2
  fi
done
if [[ -e "${METADATA}" ]]; then
  echo "Refusing to overwrite launch provenance: ${METADATA}" >&2
  exit 2
fi

# A cached remote-tracking ref is insufficient for a confirmatory launch.
# Network/fetch failure is fatal under set -e.
git -C "${REPO_ROOT}" fetch origin main
REPO_HEAD="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
ORIGIN_MAIN="$(git -C "${REPO_ROOT}" rev-parse origin/main)"
WORKTREE_STATUS="$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)"
if [[ -n "${WORKTREE_STATUS}" ]]; then
  echo "Refusing production launch from a dirty worktree." >&2
  printf '%s\n' "${WORKTREE_STATUS}" >&2
  exit 2
fi
if [[ "${REPO_HEAD}" != "${ORIGIN_MAIN}" ]]; then
  echo "Refusing production launch: HEAD ${REPO_HEAD} != origin/main ${ORIGIN_MAIN}." >&2
  exit 2
fi
"${PYTHON}" -B "${SCRIPT_DIR}/source_fingerprint.py" verify-git \
  --freeze-manifest "${SCRIPT_DIR}/freeze_manifest.json" --commit "${REPO_HEAD}"
if [[ -e "${FINGERPRINT}" ]]; then
  echo "Refusing to overwrite production source fingerprint: ${FINGERPRINT}" >&2
  exit 2
fi
"${PYTHON}" -B "${SCRIPT_DIR}/source_fingerprint.py" write \
  --stage production --output "${FINGERPRINT}"
SOURCE_TREE="$(git -C "${REPO_ROOT}" rev-parse "${REPO_HEAD}:experiment_FPP/ddpg_saber_random_support_count_matched_ls512/code/src")"
CONFIG_BLOB="$(git -C "${REPO_ROOT}" rev-parse "${REPO_HEAD}:experiment_FPP/ddpg_saber_random_support_count_matched_ls512/config.json")"
FREEZE_BLOB="$(git -C "${REPO_ROOT}" rev-parse "${REPO_HEAD}:experiment_FPP/ddpg_saber_random_support_count_matched_ls512/freeze_manifest.json")"
TIMESTAMP="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

mkdir -p "${EXECUTION_DIR}/logs"
TMP="${METADATA}.tmp.$$"
{
  printf '{\n'
  printf '  "schema_version": 1,\n'
  printf '  "experiment": "ddpg_saber_random_support_count_matched_ls512",\n'
  printf '  "timestamp_utc": "%s",\n' "${TIMESTAMP}"
  printf '  "source_commit": "%s",\n' "${REPO_HEAD}"
  printf '  "origin_main": "%s",\n' "${ORIGIN_MAIN}"
  printf '  "head_matches_origin_main": true,\n'
  printf '  "worktree_clean": true,\n'
  printf '  "explicit_production_authorization": true,\n'
  printf '  "source_tree_git_oid": "%s",\n' "${SOURCE_TREE}"
  printf '  "config_git_blob_oid": "%s",\n' "${CONFIG_BLOB}"
  printf '  "freeze_manifest_git_blob_oid": "%s",\n' "${FREEZE_BLOB}"
  printf '  "support_query_mode": "current_policy",\n'
  printf '  "support_assignment_mode": "batch_wide_count_matched",\n'
  printf '  "assignment_seed_rule": "30000+seed",\n'
  printf '  "weight_scope": "critic_only",\n'
  printf '  "steps": 1000000,\n'
  printf '  "seeds": [1, 2, 3, 4, 5],\n'
  printf '  "device": "%s",\n' "${DEVICE}"
  printf '  "max_parallel": %s\n' "${MAX_PARALLEL}"
  printf '}\n'
} >"${TMP}"
mv "${TMP}" "${METADATA}"

pids=()
pid_seeds=()
wait_batch() {
  local failed=0 index
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "Seed ${pid_seeds[$index]} completed."
    else
      echo "Seed ${pid_seeds[$index]} failed; see ${EXECUTION_DIR}/logs/seed_${pid_seeds[$index]}.log" >&2
      failed=1
    fi
  done
  pids=()
  pid_seeds=()
  return "${failed}"
}

for seed in "${SEEDS[@]}"; do
  log="${EXECUTION_DIR}/logs/seed_${seed}.log"
  echo "Launching Random-Support seed ${seed}; log: ${log}"
  (
    PYTHON_BIN="${PYTHON}" "${PYTHON}" -B "${SCRIPT_DIR}/run.py" \
      --stage production --seed "${seed}" --device "${DEVICE}" \
      --output_root "${MAIN_ROOT}"
  ) >"${log}" 2>&1 &
  pids+=("$!")
  pid_seeds+=("${seed}")
  if (( ${#pids[@]} >= MAX_PARALLEL )); then
    wait_batch || exit 1
  fi
done
if (( ${#pids[@]} > 0 )); then
  wait_batch || exit 1
fi

"${PYTHON}" -B "${SCRIPT_DIR}/source_fingerprint.py" verify \
  --fingerprint "${FINGERPRINT}" --expected-stage production
"${PYTHON}" -B "${SCRIPT_DIR}/validate_results.py" \
  --stage production --output-root "${MAIN_ROOT}" \
  --seeds "${SEEDS[@]}" --launcher-log-dir "${EXECUTION_DIR}/logs" \
  --source-fingerprint "${FINGERPRINT}" \
  --write-selected-10k --output "${EXECUTION_DIR}/completion_validation.json"
echo "All five Random-Support 1M runs and completion gates passed."

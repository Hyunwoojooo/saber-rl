#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiment_FPP/runs/td3_runtime_profile_her_vs_saber_100k}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
SEED="${SEED:-1}"

# Sequential execution on one visible GPU is part of the comparison protocol.
OUTPUT_ROOT="${OUTPUT_ROOT}" TOTAL_STEPS="${TOTAL_STEPS}" SEED="${SEED}" \
  bash "${SCRIPT_DIR}/run_fpp_td3_her_runtime_profile_100k.sh"
OUTPUT_ROOT="${OUTPUT_ROOT}" TOTAL_STEPS="${TOTAL_STEPS}" SEED="${SEED}" \
  bash "${SCRIPT_DIR}/run_fpp_td3_saber_runtime_profile_100k.sh"

HER_SUMMARY="$(find "${OUTPUT_ROOT}" -maxdepth 2 -type f -path "*/fetch_pick_and_place_td3_her_runtime_profile_s${SEED}_${TOTAL_STEPS}*/runtime_profile_summary.json" -print | sort | tail -n 1)"
SABER_SUMMARY="$(find "${OUTPUT_ROOT}" -maxdepth 2 -type f -path "*/fetch_pick_and_place_td3_saber_runtime_profile_ls512_s${SEED}_${TOTAL_STEPS}*/runtime_profile_summary.json" -print | sort | tail -n 1)"
if [[ -z "${HER_SUMMARY}" || -z "${SABER_SUMMARY}" ]]; then
  echo "Could not locate both runtime profile summaries under ${OUTPUT_ROOT}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_runtime_profile.py" \
  --her-summary "${HER_SUMMARY}" \
  --saber-summary "${SABER_SUMMARY}" \
  --output-dir "${OUTPUT_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
PYTHON="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
ROOT="${REPO_ROOT}/experiment_FPP/runs/ddpg_saber_context_eligible_count_matched_smoke_5k"
RUN="${ROOT}/fetch_pick_and_place_ddpg_saber_context_eligible_count_matched_smoke_s1_5000_ls512"
FINGERPRINT="${ROOT}/_execution/source_fingerprint.json"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON}" bash "${SCRIPT_DIR}/run_tests.sh"
if [[ -e "${FINGERPRINT}" ]]; then
  echo "Refusing to overwrite smoke source fingerprint: ${FINGERPRINT}" >&2
  exit 2
fi
"${PYTHON}" -B "${SCRIPT_DIR}/source_fingerprint.py" write \
  --stage smoke --output "${FINGERPRINT}"
"${PYTHON}" -B "${SCRIPT_DIR}/run.py" --stage smoke --seed 1 --device "${DEVICE}"
# Exercise strict checkpoint restore. A completed-run resume is intentional;
# unit tests independently prove the following assignment sequence is exact.
"${PYTHON}" -B "${SCRIPT_DIR}/run.py" --stage smoke --seed 1 --device "${DEVICE}" --resume
"${PYTHON}" -B "${SCRIPT_DIR}/source_fingerprint.py" verify \
  --fingerprint "${FINGERPRINT}" --expected-stage smoke
"${PYTHON}" -B "${SCRIPT_DIR}/validate_results.py" \
  --stage smoke --output-root "${ROOT}" --seeds 1 \
  --source-fingerprint "${FINGERPRINT}" \
  --output "${RUN}/smoke_validation.json"
echo "Strict seed-1 5k smoke and checkpoint-resume gate passed."

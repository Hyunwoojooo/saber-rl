#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
PYTHON="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
SMOKE_RUN="${REPO_ROOT}/experiment_FPP/runs/ddpg_saber_context_eligible_count_matched_smoke_5k/fetch_pick_and_place_ddpg_saber_context_eligible_count_matched_smoke_s1_5000_ls512"
SMOKE_FINGERPRINT="${REPO_ROOT}/experiment_FPP/runs/ddpg_saber_context_eligible_count_matched_smoke_5k/_execution/source_fingerprint.json"
GATE_ROOT="${REPO_ROOT}/experiment_FPP/runs/ddpg_saber_context_eligible_count_matched_gate_50k"
GATE_RUN="${GATE_ROOT}/fetch_pick_and_place_ddpg_saber_context_eligible_count_matched_gate_s1_50000_ls512"
GATE_FINGERPRINT="${GATE_ROOT}/_execution/source_fingerprint.json"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${SMOKE_RUN}/smoke_validation.json" ]]; then
  echo "Validated 5k smoke is required first: ${SMOKE_RUN}/smoke_validation.json" >&2
  exit 2
fi
if [[ ! -f "${SMOKE_FINGERPRINT}" ]]; then
  echo "Smoke source fingerprint is required first: ${SMOKE_FINGERPRINT}" >&2
  exit 2
fi
"${PYTHON}" -B "${SCRIPT_DIR}/source_fingerprint.py" verify \
  --fingerprint "${SMOKE_FINGERPRINT}" --expected-stage smoke
PYTHON_BIN="${PYTHON}" bash "${SCRIPT_DIR}/run_tests.sh"
"${PYTHON}" -B "${SCRIPT_DIR}/source_fingerprint.py" verify \
  --fingerprint "${SMOKE_FINGERPRINT}" --expected-stage smoke
if [[ -e "${GATE_FINGERPRINT}" ]]; then
  echo "Refusing to overwrite gate source fingerprint: ${GATE_FINGERPRINT}" >&2
  exit 2
fi
"${PYTHON}" -B "${SCRIPT_DIR}/source_fingerprint.py" write \
  --stage gate --output "${GATE_FINGERPRINT}"
"${PYTHON}" -B "${SCRIPT_DIR}/run.py" --stage gate --seed 1 --device "${DEVICE}"
# Exercise restoration of the exact 50k assignment RNG and diagnostic state.
"${PYTHON}" -B "${SCRIPT_DIR}/run.py" --stage gate --seed 1 --device "${DEVICE}" --resume
"${PYTHON}" -B "${SCRIPT_DIR}/source_fingerprint.py" verify \
  --fingerprint "${GATE_FINGERPRINT}" --expected-stage gate
"${PYTHON}" -B "${SCRIPT_DIR}/validate_results.py" \
  --stage gate --output-root "${GATE_ROOT}" --seeds 1 \
  --source-fingerprint "${GATE_FINGERPRINT}" \
  --write-selected-10k --output "${GATE_RUN}/gate_validation.json"
"${PYTHON}" -B "${SCRIPT_DIR}/freeze_experiment.py" \
  --smoke-validation "${SMOKE_RUN}/smoke_validation.json" \
  --gate-validation "${GATE_RUN}/gate_validation.json"
echo "Strict seed-1 50k technical gate passed; production source freeze written."

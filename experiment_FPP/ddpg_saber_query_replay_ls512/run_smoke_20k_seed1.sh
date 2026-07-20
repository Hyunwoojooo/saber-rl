#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEVICE="${DEVICE:-cuda}"
STEPS=20000
SEED=1
SMOKE_ROOT="${REPO_ROOT}/experiment_FPP/runs/ddpg_saber_query_replay_ls512_smoke20k"
RUN_NAME="fetch_pick_and_place_ddpg_saber_query_replay_ls512_s1_20000_ls512"
TARGET_RUN="${SMOKE_ROOT}/${RUN_NAME}"

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

if [[ -e "${TARGET_RUN}" ]]; then
  echo "Refusing to overwrite existing smoke run: ${TARGET_RUN}" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON}" bash "${SCRIPT_DIR}/run_tests.sh"

PYTHON_BIN="${PYTHON}" "${PYTHON}" -B "${SCRIPT_DIR}/run.py" \
  --seed "${SEED}" \
  --steps "${STEPS}" \
  --device "${DEVICE}" \
  --output_root "${SMOKE_ROOT}"

"${PYTHON}" -B "${SCRIPT_DIR}/validate_results.py" \
  --output-root "${SMOKE_ROOT}" \
  --steps "${STEPS}" \
  --seeds "${SEED}" \
  --write-selected-10k

echo "20k seed-1 replay-action-query smoke gate passed: ${TARGET_RUN}"

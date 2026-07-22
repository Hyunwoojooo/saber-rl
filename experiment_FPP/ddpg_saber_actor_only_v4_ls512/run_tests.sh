#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

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
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "No Python interpreter found; set PYTHON_BIN explicitly." >&2
  exit 2
}

PYTHON="$(resolve_python)"

for script in \
  run_smoke_20k_seed1.sh \
  run_main_1m_seeds1_5_parallel.sh \
  run_tests.sh \
  run_all_seeds.sh \
  run_screening_3seeds_parallel.sh; do
  bash -n "${SCRIPT_DIR}/${script}"
done

"${PYTHON}" -B "${SCRIPT_DIR}/validate_config_parity.py"
"${PYTHON}" -B -m unittest discover -s "${SCRIPT_DIR}/analysis" -p 'test_*.py' -v
"${PYTHON}" -B "${SCRIPT_DIR}/code/src/test_actor_only_scope.py"
"${PYTHON}" -B "${SCRIPT_DIR}/code/src/test_b3_runtime_opt_lookup.py"
"${PYTHON}" -B "${SCRIPT_DIR}/code/src/test_numba_runtime_opt_fpp.py"
"${PYTHON}" -B "${SCRIPT_DIR}/code/src/test_numba_support_scan_fpp.py"

echo "All static, parity, actor-only, and canonical runtime tests passed."

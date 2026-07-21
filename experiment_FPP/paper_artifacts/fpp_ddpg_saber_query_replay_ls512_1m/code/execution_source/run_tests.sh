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
SOURCE_ROOT="${SCRIPT_DIR}/code/src"

for script in \
  run_smoke_20k_seed1.sh \
  run_main_1m_seeds1_5_parallel.sh \
  run_tests.sh \
  run_all_seeds.sh \
  run_screening_3seeds_parallel.sh; do
  bash -n "${SCRIPT_DIR}/${script}"
done

"${PYTHON}" -B -m py_compile \
  "${SCRIPT_DIR}/run.py" \
  "${SCRIPT_DIR}/validate_config_parity.py" \
  "${SCRIPT_DIR}/validate_results.py" \
  "${SCRIPT_DIR}/analysis/query_ablation.py" \
  "${SCRIPT_DIR}/analysis/analyze_results.py" \
  "${SCRIPT_DIR}/analysis/build_artifact.py" \
  "${SCRIPT_DIR}/analysis/verify_artifact.py" \
  "${SCRIPT_DIR}/analysis/test_analysis.py" \
  "${SCRIPT_DIR}/analysis/test_artifact.py"
"${PYTHON}" -B "${SCRIPT_DIR}/validate_config_parity.py"

PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" -B "${SOURCE_ROOT}/test_support_query_modes.py"
PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" -B "${SOURCE_ROOT}/test_ddpg_query_loss_isolation.py"
PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" -B "${SOURCE_ROOT}/test_current_policy_legacy_parity.py"
PARITY_NUMBA=1 PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" -B "${SOURCE_ROOT}/test_current_policy_legacy_parity.py"
PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" -B "${SOURCE_ROOT}/test_b3_runtime_opt_lookup.py"
PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" -B "${SOURCE_ROOT}/test_numba_runtime_opt_fpp.py"
PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" -B "${SOURCE_ROOT}/test_numba_support_scan_fpp.py"

"${PYTHON}" -B "${SCRIPT_DIR}/analysis/test_analysis.py"
"${PYTHON}" -B "${SCRIPT_DIR}/analysis/test_artifact.py"

PYTHON_BIN="${PYTHON}" "${PYTHON}" -B "${SCRIPT_DIR}/run.py" \
  --seed 1 --steps 20000 --device cpu --validate_only >/dev/null
"${PYTHON}" -B "${SCRIPT_DIR}/run.py" --seed 1 --steps 20000 --dry-run >/dev/null
"${PYTHON}" -B "${SCRIPT_DIR}/run.py" --seed 5 --steps 1000000 --dry-run >/dev/null

echo "All static, source/config parity, query semantics, legacy parity, cache, and scanner tests passed."

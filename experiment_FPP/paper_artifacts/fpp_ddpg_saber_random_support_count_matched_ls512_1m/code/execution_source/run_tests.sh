#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

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
  run_smoke_5k_seed1.sh \
  run_technical_gate_50k_seed1.sh \
  run_main_1m_seeds1_5_parallel.sh \
  run_smoke_20k_seed1.sh \
  run_all_seeds.sh \
  run_screening_3seeds_parallel.sh \
  run_tests.sh; do
  bash -n "${SCRIPT_DIR}/${script}"
done

"${PYTHON}" -B -c \
  'import ast, pathlib, sys; [ast.parse(pathlib.Path(path).read_text(encoding="utf-8"), filename=path) for path in sys.argv[1:]]' \
  "${SCRIPT_DIR}/run.py" \
  "${SCRIPT_DIR}/validate_config_parity.py" \
  "${SCRIPT_DIR}/validate_results.py" \
  "${SCRIPT_DIR}/freeze_experiment.py" \
  "${SCRIPT_DIR}/source_fingerprint.py" \
  "${SCRIPT_DIR}/test_source_fingerprint.py" \
  "${SCRIPT_DIR}/test_validation_optional_nan.py" \
  "${SCRIPT_DIR}/analysis/random_support_analysis.py" \
  "${SCRIPT_DIR}/analysis/analyze_results.py" \
  "${SCRIPT_DIR}/analysis/build_artifact.py" \
  "${SCRIPT_DIR}/analysis/verify_artifact.py" \
  "${SCRIPT_DIR}/analysis/test_analysis.py" \
  "${SCRIPT_DIR}/analysis/test_artifact.py"

"${PYTHON}" -B "${SCRIPT_DIR}/validate_config_parity.py"
(
  cd "${SCRIPT_DIR}"
  "${PYTHON}" -B test_source_fingerprint.py
)

run_source_test() {
  PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" -B "${SOURCE_ROOT}/$1"
}

run_source_test test_batch_wide_count_matched.py
run_source_test test_context_eligible_count_matched.py
run_source_test test_ddpg_random_support_loss_scope.py
run_source_test test_trainer_resume_contract.py
run_source_test test_resume_output_rollback.py
run_source_test test_support_query_modes.py
run_source_test test_ddpg_query_loss_isolation.py
run_source_test test_current_policy_legacy_parity.py
PARITY_NUMBA=1 run_source_test test_current_policy_legacy_parity.py
run_source_test test_b3_runtime_opt_lookup.py
run_source_test test_numba_runtime_opt_fpp.py
run_source_test test_numba_support_scan_fpp.py

(
  cd "${SCRIPT_DIR}"
  "${PYTHON}" -B test_validation_optional_nan.py
)
"${PYTHON}" -B "${SCRIPT_DIR}/analysis/test_analysis.py"
"${PYTHON}" -B "${SCRIPT_DIR}/analysis/test_artifact.py"

"${PYTHON}" -B "${SCRIPT_DIR}/run.py" \
  --stage smoke --seed 1 --device cpu --validate_only >/dev/null
"${PYTHON}" -B "${SCRIPT_DIR}/run.py" \
  --stage gate --seed 1 --device cpu --validate_only >/dev/null
if [[ -f "${SCRIPT_DIR}/freeze_manifest.json" ]]; then
  "${PYTHON}" -B "${SCRIPT_DIR}/run.py" \
    --stage production --seed 5 --device cpu --dry-run >/dev/null
fi

echo "All batch-wide assignment, RNG/resume, DDPG-loss, legacy-parity, scanner, raw-NaN, analysis, and artifact tests passed."

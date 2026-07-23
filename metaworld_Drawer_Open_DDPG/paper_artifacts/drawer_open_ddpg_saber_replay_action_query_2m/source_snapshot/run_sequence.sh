#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"

"$PYTHON" -m py_compile "$EXP_DIR/run.py" "$EXP_DIR/validate_config_parity.py" "$EXP_DIR/validate_integration.py" "$EXP_DIR/code/src/train_drawer_open_lite_v0.py"
"$PYTHON" "$EXP_DIR/validate_config_parity.py"
"$PYTHON" "$EXP_DIR/test_config_parity.py"
"$PYTHON" "$EXP_DIR/test_analysis_artifact_contract.py"
"$PYTHON" "$EXP_DIR/test_freeze_source_head_binding.py"

test_files=("$EXP_DIR/code/src"/test_*.py)
if [[ ! -e "${test_files[0]}" ]]; then
  printf '%s\n' "No direct test files found under $EXP_DIR/code/src" >&2
  exit 1
fi
for test_file in "${test_files[@]}"; do
  printf 'RUN %s\n' "$test_file"
  PYTHONPATH="$EXP_DIR/code/src:${PYTHONPATH:-}" "$PYTHON" "$test_file"
done

"$PYTHON" "$EXP_DIR/validate_source_parity.py" --output "$EXP_DIR/source_parity.json"
"$PYTHON" "$EXP_DIR/audit_checkpoint_resume.py" --output "$EXP_DIR/checkpoint_resume_audit.json"

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  printf '%s\n' 'All static, unit, source-parity, and checkpoint-resume prechecks passed.'
  exit 0
fi

MODE=correctness5k SEEDS="1" VALIDATE_ONLY=1 CUDA_DEVICE="$CUDA_DEVICE" MAX_PARALLEL=1 PYTHON="$PYTHON" "$EXP_DIR/launch.sh"
MODE=correctness5k SEEDS="1" CUDA_DEVICE="$CUDA_DEVICE" MAX_PARALLEL=1 PYTHON="$PYTHON" "$EXP_DIR/launch.sh"
"$PYTHON" "$EXP_DIR/validate_integration.py" --mode correctness5k --seeds 1

MODE=smoke20k_seed1 SEEDS="1" CUDA_DEVICE="$CUDA_DEVICE" MAX_PARALLEL=1 PYTHON="$PYTHON" "$EXP_DIR/launch.sh"
"$PYTHON" "$EXP_DIR/validate_integration.py" --mode smoke20k_seed1 --seeds 1

MODE=smoke20k_seeds1_5 SEEDS="1 2 3 4 5" CUDA_DEVICE="$CUDA_DEVICE" MAX_PARALLEL="$MAX_PARALLEL" PYTHON="$PYTHON" "$EXP_DIR/launch.sh"
"$PYTHON" "$EXP_DIR/validate_integration.py" --mode smoke20k_seeds1_5

printf '%s\n' \
  'All pre-production correctness gates passed.' \
  'STOP: freeze and locally commit the verified source before starting main2m.' \
  'After recording the frozen commit SHA, run:' \
  '  MODE=main2m SEEDS="1 2 3 4 5" MAX_PARALLEL=1 bash metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Replay-Action-Query-2M.sh'

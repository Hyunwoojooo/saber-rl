#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiment_FPP/runs/td3_runtime_profile_her_vs_saber_100k}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
START_STEPS="${START_STEPS:-10000}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
PROFILE_START_STEP="${PROFILE_START_STEP:-20000}"
PROFILE_END_STEP="${PROFILE_END_STEP:-${TOTAL_STEPS}}"
PROFILE_SAMPLE_EVERY_UPDATES="${PROFILE_SAMPLE_EVERY_UPDATES:-100}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run.py" \
  --method td3_her \
  --seed "${SEED}" \
  --steps "${TOTAL_STEPS}" \
  --start_steps "${START_STEPS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --eval_freq "${EVAL_FREQ}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --profile_start_step "${PROFILE_START_STEP}" \
  --profile_end_step "${PROFILE_END_STEP}" \
  --profile_sample_every_updates "${PROFILE_SAMPLE_EVERY_UPDATES}" \
  --device cuda \
  --output_root "${OUTPUT_ROOT}"

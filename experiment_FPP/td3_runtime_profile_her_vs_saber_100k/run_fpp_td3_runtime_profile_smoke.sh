#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiment_FPP/runs/td3_runtime_profile_her_vs_saber_smoke}"

OUTPUT_ROOT="${OUTPUT_ROOT}" \
TOTAL_STEPS=2000 \
START_STEPS=256 \
WARMUP_STEPS=256 \
PROFILE_START_STEP=512 \
PROFILE_END_STEP=2000 \
PROFILE_SAMPLE_EVERY_UPDATES=10 \
EVAL_FREQ=1000 \
EVAL_EPISODES=2 \
bash "${SCRIPT_DIR}/run_fpp_td3_runtime_profile_pair_100k.sh"

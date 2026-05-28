#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${WORKSPACE_ROOT}"

INPUT_ROOT="${INPUT_ROOT:-${EXPERIMENT_ROOT}/experiment/FetchSlide}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_ROOT}/analysis/FetchSlide}"

python3 "${EXPERIMENT_ROOT}/src/saber_fetch_slide/analyze.py" \
  --input_root "${INPUT_ROOT}" \
  --output_root "${OUTPUT_ROOT}"

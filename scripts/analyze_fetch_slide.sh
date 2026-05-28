#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

INPUT_ROOT="${INPUT_ROOT:-SABER/experiment/FetchSlide}"
OUTPUT_ROOT="${OUTPUT_ROOT:-SABER/analysis/FetchSlide}"

python3 SABER/src/saber_fetch_slide/analyze.py \
  --input_root "${INPUT_ROOT}" \
  --output_root "${OUTPUT_ROOT}"


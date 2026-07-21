#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/smoke}"

exec python3 "$SCRIPT_DIR/run.py" \
  --seed 1 \
  --smoke \
  --device "$DEVICE" \
  --output-root "$OUTPUT_ROOT" \
  --support-query-mode shuffled_policy \
  "$@"

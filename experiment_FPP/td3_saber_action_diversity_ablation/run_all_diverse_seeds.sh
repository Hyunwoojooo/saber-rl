#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 SELECTED_THRESHOLD_JSON [run.py args...]" >&2
  exit 2
fi
SELECTED_THRESHOLD="$1"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for seed in 1 2 3 4 5; do
  python3 "$SCRIPT_DIR/run.py" \
    --condition diverse \
    --selected-threshold "$SELECTED_THRESHOLD" \
    --seed "$seed" \
    "$@"
done

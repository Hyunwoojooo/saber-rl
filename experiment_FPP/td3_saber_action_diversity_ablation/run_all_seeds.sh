#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 {diverse|random_count} SELECTED_THRESHOLD_JSON [run.py args...]" >&2
  exit 2
fi
CONDITION="$1"
SELECTED_THRESHOLD="$2"
shift 2
case "$CONDITION" in
  diverse|random_count) ;;
  *)
    echo "condition must be diverse or random_count" >&2
    exit 2
    ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for seed in 1 2 3 4 5; do
  python3 "$SCRIPT_DIR/run.py" \
    --condition "$CONDITION" \
    --selected-threshold "$SELECTED_THRESHOLD" \
    --seed "$seed" \
    "$@"
done

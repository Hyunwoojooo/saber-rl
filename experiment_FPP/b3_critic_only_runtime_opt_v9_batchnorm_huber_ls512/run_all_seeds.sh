#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for seed in 1 2 3 4 5; do
  python3 "$SCRIPT_DIR/run.py" --seed "$seed" "$@"
done

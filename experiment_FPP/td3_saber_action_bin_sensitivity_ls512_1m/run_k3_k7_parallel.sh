#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
export MAX_PARALLEL="${MAX_PARALLEL:-7}"
exec "$SCRIPT_DIR/run_all_seeds.sh" both "$@"

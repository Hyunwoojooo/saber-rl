#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$ROOT/.venv}"
SKIP_TORCH="${2:-}"

if [[ "$SKIP_TORCH" == "--skip-torch" ]]; then
  python3 -m venv --system-site-packages "$VENV"
else
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip wheel

if [[ "$SKIP_TORCH" == "--skip-torch" ]]; then
  grep -v '^torch==' "$ROOT/environment/requirements-lock.txt" \
    | "$VENV/bin/python" -m pip install -r /dev/stdin
else
  "$VENV/bin/python" -m pip install -r "$ROOT/environment/requirements-lock.txt"
fi

"$VENV/bin/python" "$ROOT/reviewer_tools/check_environment.py"

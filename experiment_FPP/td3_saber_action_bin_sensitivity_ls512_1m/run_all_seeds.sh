#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {k3|k7} [run.py args...]" >&2
  exit 2
fi
VARIANT="$1"
shift
case "$VARIANT" in
  k3|k7) ;;
  *)
    echo "variant must be k3 or k7" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
for seed in 1 2 3 4 5; do
  python3 "$SCRIPT_DIR/run.py" \
    --variant "$VARIANT" \
    --seed "$seed" \
    "$@"
done

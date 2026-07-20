#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS:-0,1,2,3,4}"
if (( ${#GPU_ARRAY[@]} < 5 )); then
  echo "GPU_IDS must contain at least five comma-separated device IDs" >&2
  exit 2
fi

mkdir -p "$SCRIPT_DIR/logs"
pids=()
for index in 0 1 2 3 4; do
  seed=$((index + 1))
  gpu="${GPU_ARRAY[$index]}"
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$SCRIPT_DIR/run.py"     --seed "$seed" --device cuda "$@"     >"$SCRIPT_DIR/logs/seed_${seed}.log" 2>&1 &
  pids+=("$!")
  echo "seed=$seed gpu=$gpu pid=${pids[-1]}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"

#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$EXPERIMENT_DIR/TD3-ER-2M.sh"
bash "$EXPERIMENT_DIR/TD3-AnnealingER-2M.sh"
bash "$EXPERIMENT_DIR/TD3-AER-2M.sh"

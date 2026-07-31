#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m unittest discover -s tests -p 'test_*.py' -v

for preset in \
  fpp_ddpg fpp_td3 fpp_sac \
  drawer_open_ddpg drawer_open_td3 \
  drawer_close_ddpg drawer_close_td3
do
  python reviewer_tools/launch_experiment.py \
    --preset "$preset" --seed 1 --steps 5000 --device cpu --dry-run >/dev/null
done

PYTHONPATH="$ROOT/source/fpp_ddpg" \
  python "$ROOT/source/fpp_ddpg/test_b3_runtime_opt_lookup.py"

run_source_test() {
  local source_name="$1"
  local test_file="$2"
  PYTHONPATH="$ROOT/source/$source_name" \
    python "$ROOT/source/$source_name/$test_file"
}

run_source_test control_replay_action test_ddpg_query_loss_isolation.py
run_source_test control_shuffled_policy test_ddpg_query_invariants.py
run_source_test control_random_support test_batch_wide_count_matched.py
run_source_test control_random_support test_ddpg_random_support_loss_scope.py
run_source_test control_strict_count_matched test_context_eligible_count_matched.py
run_source_test control_strict_count_matched test_ddpg_strict_loss_scope.py

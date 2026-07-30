# DrawerOpen DDPG SABER Fixed-Target Diagnostic, Seeds 6-10

This artifact contains the completed 2M-step fixed-target diagnostic runs for
seeds 6-10. It is a separate diagnostic cohort and must not be presented as a
paired HER comparison.

## Scope

- Task: Meta-World `drawer-open-v3`, sparse reward function v2
- Backbone: DDPG + future-goal HER
- Method: SABER Critic-Only
- Seeds: 6-10
- Budget: 2,000,000 environment steps per seed
- Execution source: `22015c0a3743d4a10f72197752022ed0e08ee730`
- Completed: 2026-07-29T22:39:11+09:00

## Packaged data

- `data/seeds/seed_XX/metrics.csv.gz`: complete original metrics CSV,
  deterministically compressed; checkpoints and duplicate JSONL are excluded.
- `data/seeds/seed_XX/config.json`: sanitized run configuration.
- `data/seeds/seed_XX/env_snapshot.json`: sanitized environment definition.
- `data/seeds/seed_XX/summary.json`: sanitized completed-run summary.
- `analysis/eval_curves.csv`: exact 10k-step evaluation grid.
- `analysis/performance_summary.json`: per-seed and aggregate performance.
- `validation/validation_summary.json`: completion and diagnostic-contract checks.
- `provenance/`: execution-source hashes and post-run environment snapshot.

## Performance summary

- Mean AUC: 0.381558 +/- 0.381141
- Mean final success: 0.600 +/- 0.548
- Hit90 coverage: 3/5
- Median Hit90 among reached seeds: 480,000 steps

All five runs contain exactly 9,976 fixed-target diagnostic snapshots on the
5k-2M/200-step grid, an exact top-10% count of 26 for batch size 256, no Numba
scanner fallback, and no support-audit mismatch.

The software environment was collected immediately after the runs from the
same machine and virtual environment. It was not embedded independently in
each process at launch; this limitation is recorded in `provenance/environment.json`.

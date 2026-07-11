# DrawerOpen DDPG-HER-B3-Critic-Only 2M Diagnostic Results, Seeds 6-10

Diagnostic logging run:

- Experiment: `DDPG-HER-B3-Critic-Only-2M`
- Task: `drawer_open`
- Method: `b3_support_runtime_opt`
- Weight scope: `critic_only`
- Seeds: `6,7,8,9,10`
- Steps per seed: `2,000,000`
- Start: `2026-07-09 23:14:12 UTC`
- End: `2026-07-11 06:19:20 UTC`
- Output folder: `metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Critic-Only-2M`

Per-seed final metrics:

| seed | final step | final success | best success | AUC | Hit90 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 2,000,000 | 1.00 | 1.00 | 0.766 | 470k |
| 7 | 2,000,000 | 0.00 | 0.00 | 0.000 | - |
| 8 | 2,000,000 | 1.00 | 1.00 | 0.576 | 830k |
| 9 | 2,000,000 | 0.95 | 1.00 | 0.571 | 850k |
| 10 | 2,000,000 | 1.00 | 1.00 | 0.837 | 340k |

Aggregate:

- Success seeds: `4/5`
- Final success: `0.79`
- Best success: `0.80`
- AUC: `0.550`
- Mean Hit90 over reached seeds: `623k`
- Median Hit90 over reached seeds: `650k`

Included artifacts:

- `metrics.csv`
- `summary.json`
- `config.json`
- `env_snapshot.json`
- `run.log`
- `best_checkpoint.json`

Excluded from git because they are too large for regular GitHub repository storage:

- `checkpoint.pt`
- `best_checkpoint.pt`
- `metrics.jsonl`

This run includes online diagnostic logging for support-positive rate, TD-error enrichment, and one-step residual correction metrics. The diagnostic logging does not change replay sampling, loss weights, optimizer steps, or policy updates.

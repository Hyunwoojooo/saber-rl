# FPP TD3+HER confirmatory seeds 6-10 (1M)

This artifact contains the completed FetchPickAndPlace TD3+HER confirmatory
extension for seeds 6-10. Each run used 1,000,000 environment steps and the
frozen protocol recorded in `experiment_config.json`.

This is a partial experiment artifact for the HER arm only. The paired
TD3+HER+SABER seeds 6-10 runs and the combined seeds 1-10 comparison are not
included here and must not be inferred from these files.

## Frozen HER arm

- Environment: `FetchPickAndPlace-v4`, sparse reward
- Method: `td3_her`
- Seeds: 6, 7, 8, 9, 10
- Steps per seed: 1,000,000
- Evaluation: 20 episodes every 10,000 steps
- HER ratio: 0.8
- TD3 policy delay: 2
- Support weighting: disabled (`lambda_support=0`, `weight_scope=none`)
- Local-search setting retained for protocol matching: 512

## Included data

Each seed directory under `data/td3_her/` contains:

- `summary.json`
- `config.json`
- `env_snapshot.json`
- `run.log`
- `metrics_selected_10k.csv`

`metrics_selected_10k.csv` contains the exact evaluation grid
`{10k, 20k, ..., 1M}`. No interpolation, smoothing, or forward filling was
used. Checkpoints, raw `metrics.csv`, and raw `metrics.jsonl` are excluded.

## Seeds 6-10 summary

| Metric | Mean | Sample std |
|---|---:|---:|
| Final success | 0.8800 | 0.2139 |
| Best success | 0.9300 | 0.1304 |
| AUC (10k-1M, normalized by 990k) | 0.3936 | 0.0498 |
| Best-final drop | 0.0500 | 0.0866 |

Hit90 was reached by 4/5 seeds; the reached-only median was 930k steps.
Detailed values are in `per_seed_results.csv` and `aggregate_results.csv`.

`artifact_manifest.json` records SHA256 and size for every payload file. The
manifest excludes itself from its payload list by construction.

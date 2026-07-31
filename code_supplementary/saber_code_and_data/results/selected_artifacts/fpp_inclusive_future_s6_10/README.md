# FPP TD3 Inclusive-Future HER Sensitivity: Seeds 6--10

This artifact contains the completed paired seeds 6--10 sensitivity extension
between TD3 HER and TD3 HER+SABER under the inclusive future-goal indexing
convention. For a transition at step `t`, the eligible stored transition
indices include `u=t`, and relabeling uses the selected transition's
`next_achieved_goal`. This admits `m(s_{t+1})` and makes the final
transition eligible for relabeling.

Both arms use seeds 6--10, one million environment steps, UTD 1, uniform replay,
the same inclusive relabeling implementation, and the same 10k--1M evaluation
grid. The intended arm differences are SABER's method identifier,
`lambda_support`, critic-only weight scope, and disabled audit logging.
No sensitivity-specific tuning was performed.

## Results

| Metric | HER mean +/- sample SD | SABER mean +/- sample SD | Paired mean difference | 95% paired-bootstrap CI | Positive pairs |
|---|---:|---:|---:|---:|---:|
| Normalized success AUC | 0.3274 +/- 0.0897 | 0.3388 +/- 0.0181 | +0.0114 | [-0.0507, 0.0950] | 2/5 |
| Final-quarter AUC | 0.5350 +/- 0.1254 | 0.5086 +/- 0.0294 | -0.0264 | [-0.1422, 0.0724] | 3/5 |
| Final success | 0.6200 +/- 0.2280 | 0.5300 +/- 0.1304 | -0.0900 | [-0.3500, 0.1300] | 3/5 |

The extension has a small positive mean SABER-minus-HER AUC difference, but
the paired interval includes zero and only two of five AUC pairs are positive.
Final-quarter AUC and final-success means favor HER. These results were retained
without tuning and do not establish a universal effect in either direction.

## Contents

- `data/evaluation_curves.csv`: exact 100-point success curves for all ten runs.
- `results/per_seed_metrics.csv`: seed-level endpoints for both arms.
- `results/paired_seed_metrics.csv`: within-seed SABER-minus-HER differences.
- `results/aggregate_metrics.csv`: mean and sample SD by arm.
- `results/paired_statistics.csv`: paired mean differences, bootstrap CIs,
  exact sign-flip tests, and win/tie/loss counts.
- `provenance/execution.json`: sanitized launch provenance.
- `provenance/run_contracts.json`: shared configuration and intended arm
  differences, with source-file hashes.
- `validation_summary.json`: grid, AUC, configuration, and completion checks.
- `artifact_manifest.json`: SHA-256 hashes for every artifact file.

No checkpoints, replay buffers, full training logs, absolute local paths, or
raw high-frequency metrics are included.

## Rebuild and verify

From the repository root:

```bash
python3 \
  experiment_FPP/paper_artifacts/fpp_td3_inclusive_future_her_sensitivity_seeds6_10_1m/code/build_artifact.py \
  --run-root /path/to/td3_inclusive_future_sensitivity_ls512_seeds6_10

python3 \
  experiment_FPP/paper_artifacts/fpp_td3_inclusive_future_her_sensitivity_seeds6_10_1m/code/verify_artifact.py
```

The execution checkout was
`70fb7041f4b5142ce91bfd207142e361cced17b8`. Its training source is
byte-identical to the validated source commit
`47343f4cb295f354df16733dca17f3a882a2b34a`. The executed launcher is bound
by Git blob `2b4a6d7f960eb56d9013f4694abe0e6d14d2c441`. The experiment source is
tracked at
`experiment_FPP/td3_inclusive_future_sensitivity_ls512/`.

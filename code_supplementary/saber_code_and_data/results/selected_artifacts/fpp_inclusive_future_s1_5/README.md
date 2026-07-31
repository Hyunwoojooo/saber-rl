# FPP TD3 Inclusive-Future HER Sensitivity

This artifact contains the completed paired five-seed sensitivity comparison
between TD3 HER and TD3 HER+SABER under an inclusive future-goal indexing
convention. For a transition at step `t`, the eligible stored transition
indices include `u=t`, and relabeling uses the selected transition's
`next_achieved_goal`. This admits `m(s_{t+1})` and makes the final
transition eligible for relabeling.

Both arms use seeds 1--5, one million environment steps, UTD 1, uniform replay,
the same inclusive relabeling implementation, and the same 10k--1M evaluation
grid. The intended arm differences are SABER's method identifier,
`lambda_support`, critic-only weight scope, and disabled audit logging.
No sensitivity-specific tuning was performed.

## Results

| Metric | HER mean +/- sample SD | SABER mean +/- sample SD | Paired mean difference | 95% paired-bootstrap CI | Positive pairs |
|---|---:|---:|---:|---:|---:|
| Normalized success AUC | 0.3261 +/- 0.0409 | 0.3935 +/- 0.1005 | +0.0674 | [-0.0068, 0.1416] | 3/5 |
| Final-quarter AUC | 0.5206 +/- 0.0773 | 0.7060 +/- 0.2301 | +0.1854 | [-0.0212, 0.3920] | 4/5 |
| Final success | 0.5600 +/- 0.2485 | 0.7800 +/- 0.2797 | +0.2200 | [-0.2000, 0.5200] | 4/5 |

The observed mean SABER-over-HER AUC direction is preserved under the
inclusive convention. The primary paired interval includes zero, so this
artifact supports a sensitivity statement about the observed direction rather
than a claim of independently conclusive five-seed evidence.

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
  experiment_FPP/paper_artifacts/fpp_td3_inclusive_future_her_sensitivity_1m/code/build_artifact.py \
  --run-root /path/to/td3_inclusive_future_sensitivity_ls512

python3 \
  experiment_FPP/paper_artifacts/fpp_td3_inclusive_future_her_sensitivity_1m/code/verify_artifact.py
```

The execution source snapshot is `snapshot_reference_010`.
The frozen experiment source is tracked at
`experiment_FPP/td3_inclusive_future_sensitivity_ls512/`.

# FPP TD3 HER PER-style Loss-Weighting 1M Artifact

> **Excluded result notice (2026-07-27).** This artifact is retained for
> provenance but is excluded from the paper and supplementary analyses. Its
> TD-error weights and optimized critic residuals used independently sampled
> TD3 target-policy smoothing noise, so the packaged runs are not treated as
> an exact TD-error critic-loss weighting control.

This artifact packages completed FetchPickAndPlace-v4 sparse-reward
TD3+HER+PER-style loss-weighting runs for seeds 1-5.

| Method | Final success mean +/- std | Success AUC mean +/- std |
|---|---:|---:|
| TD3+HER+PER-style | 0.490000 +/- 0.171026 | 0.351364 +/- 0.021377 |

## Contents

- `data/`: sanitized config, environment snapshot, summary, and compact
  evaluation-only metrics (100 points) for every seed.
- `results/`: per-seed and aggregate CSVs, report, and machine-readable validation.
- `input_manifest.json`: SHA256 provenance for raw inputs and experiment source.
- `artifact_manifest.json`: SHA256 inventory of the packaged artifact.

The runnable source is versioned in the same repository under
`experiment_FPP/td3_her_per_style_loss_weighting_ls512`. Regenerate from the repository root with:

```bash
.venv/bin/python experiment_FPP/package_td3_her_per_style_results.py
```

Raw checkpoints, replay state, full metrics, JSONL, and logs are excluded.
See `results/report.md` for tables and evaluation semantics.

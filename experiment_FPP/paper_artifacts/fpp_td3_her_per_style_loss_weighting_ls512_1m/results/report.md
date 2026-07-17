# FPP TD3 HER PER-style Loss-Weighting 1M Results

All five runs completed 1,000,000 environment steps with 995,001 updates
and a final replay size of 300,000. Aggregate values are arithmetic means
and sample standard deviations (ddof=1) across seeds 1-5.

## Aggregate Results

| Method | Final success | Best success | Success AUC | Final return | Runtime (h) | Steps/s | Hit90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TD3+HER+PER-style | 0.490000 +/- 0.171026 | 0.790000 +/- 0.082158 | 0.351364 +/- 0.021377 | -32.910000 +/- 5.870094 | 4.588774 +/- 0.004594 | 60.534249 +/- 0.060576 | 1/5 |

## Per-Seed Results

| Seed | Final | Best (step) | AUC | Final return | Hit90 | Runtime (h) | Steps/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.40 | 0.75 (630000) | 0.327525 | -34.30 | - | 4.5858 | 60.5735 |
| 2 | 0.30 | 0.70 (630000) | 0.333586 | -40.65 | - | 4.5859 | 60.5715 |
| 3 | 0.55 | 0.75 (920000) | 0.380808 | -31.30 | - | 4.5918 | 60.4938 |
| 4 | 0.45 | 0.90 (630000) | 0.356313 | -33.90 | 630000 | 4.5954 | 60.4474 |
| 5 | 0.75 | 0.85 (630000) | 0.358586 | -24.40 | - | 4.5849 | 60.5850 |

## Evaluation and Packaging

Evaluation used the native 10,000-step grid from 10,000 through 1,000,000
(100 points). Success AUC is verified by trapezoidal integration on that
exact grid and normalized by the distance from the first to final evaluation.
No interpolation, smoothing, extrapolation, or forward filling is used.

The fail-closed validation also verifies the exact method, run/seed identity,
1M-step/update/replay counts, HER ratio 0.8, critic-only batch-normalized
PER-style loss weighting, PER alpha 0.6, and U_PER_max 2.0.

Checkpoints, full per-update metrics, JSONL files, and logs are intentionally
excluded. Packaged JSON replaces repository-local absolute paths with
`${REPO_ROOT}`, redacts any other absolute path, and emits non-finite values
as strict JSON `null`.

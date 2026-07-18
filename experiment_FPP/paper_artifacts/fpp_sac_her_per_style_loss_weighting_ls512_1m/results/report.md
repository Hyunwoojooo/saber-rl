# FPP SAC HER PER-style Loss-Weighting 1M Results

All five runs completed 1,000,000 environment steps with 995,001 updates
and a final replay size of 300,000. Aggregate values are arithmetic means
and sample standard deviations (ddof=1) across seeds 1-5.

## Aggregate Results

| Method | Final success | Best success | Success AUC | Final return | Runtime (h) | Steps/s | Hit90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAC+HER+PER-style | 0.990000 +/- 0.022361 | 0.990000 +/- 0.022361 | 0.532949 +/- 0.073484 | -9.970000 +/- 3.084356 | 7.281001 +/- 0.015189 | 38.151176 +/- 0.079527 | 5/5 |

## Per-Seed Results

| Seed | Final | Best (step) | AUC | Final return | Hit90 | Runtime (h) | Steps/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 (1000000) | 0.551282 | -8.10 | 775000 | 7.3019 | 38.0420 |
| 2 | 1.00 | 1.00 (925000) | 0.505769 | -11.00 | 800000 | 7.2692 | 38.2132 |
| 3 | 1.00 | 1.00 (1000000) | 0.600641 | -7.65 | 700000 | 7.2768 | 38.1732 |
| 4 | 1.00 | 1.00 (1000000) | 0.587821 | -8.15 | 725000 | 7.2660 | 38.2299 |
| 5 | 0.95 | 0.95 (1000000) | 0.419231 | -14.95 | 1000000 | 7.2912 | 38.0976 |

## Evaluation and Packaging

Evaluation used the native 25,000-step grid from 25,000 through 1,000,000
(40 points). Success AUC is verified by trapezoidal integration on that
exact grid and normalized by the distance from the first to final evaluation.
No interpolation, smoothing, extrapolation, or forward filling is used.

The fail-closed validation also verifies the exact method, run/seed identity,
1M-step/update/replay counts, HER ratio 0.8, critic-only batch-normalized
PER-style loss weighting, PER alpha 0.6, U_PER_max 2.0, SAC auto-alpha,
and the min-absolute twin-critic priority reduction.

Checkpoints, full per-update metrics, JSONL files, and logs are intentionally
excluded. Packaged JSON replaces repository-local absolute paths with
`${REPO_ROOT}`, redacts any other absolute path, and emits non-finite values
as strict JSON `null`.

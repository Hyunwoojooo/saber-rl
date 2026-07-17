# FPP TD3/SAC HER Full-PER 1M Results

All ten runs completed 1,000,000 environment steps with 995,001 updates
and a final replay size of 300,000. Values below are arithmetic means
and sample standard deviations (ddof=1) across seeds 1-5.

## Aggregate Results

| Method | Final success | Best success | Success AUC | Final return | Runtime (h) | Steps/s | Hit90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TD3+HER+PER(full) | 0.660000 +/- 0.204328 | 0.820000 +/- 0.097468 | 0.368889 +/- 0.033609 | -25.870000 +/- 8.453742 | 5.469775 +/- 0.007449 | 50.784209 +/- 0.069255 | 1/5 |
| SAC+HER+PER(full) | 0.990000 +/- 0.022361 | 1.000000 +/- 0.000000 | 0.674872 +/- 0.042158 | -8.540000 +/- 1.322592 | 7.983024 +/- 0.011500 | 34.796117 +/- 0.050106 | 5/5 |

## Per-Seed Results

| Method | Seed | Final | Best (step) | AUC | Final return | Hit90 | Runtime (h) | Steps/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TD3+HER+PER(full) | 1 | 0.55 | 0.70 (960000) | 0.345455 | -29.35 | - | 5.4714 | 50.7692 |
| TD3+HER+PER(full) | 2 | 0.95 | 0.95 (1000000) | 0.408586 | -14.05 | 1000000 | 5.4766 | 50.7204 |
| TD3+HER+PER(full) | 3 | 0.80 | 0.85 (630000) | 0.394192 | -20.10 | - | 5.4702 | 50.7804 |
| TD3+HER+PER(full) | 4 | 0.50 | 0.85 (630000) | 0.369192 | -33.70 | - | 5.4735 | 50.7498 |
| TD3+HER+PER(full) | 5 | 0.50 | 0.75 (730000) | 0.327020 | -32.15 | - | 5.4572 | 50.9012 |
| SAC+HER+PER(full) | 1 | 1.00 | 1.00 (1000000) | 0.740385 | -7.95 | 450000 | 7.9768 | 34.8234 |
| SAC+HER+PER(full) | 2 | 1.00 | 1.00 (1000000) | 0.631410 | -8.05 | 550000 | 7.9826 | 34.7978 |
| SAC+HER+PER(full) | 3 | 1.00 | 1.00 (1000000) | 0.686538 | -7.80 | 525000 | 7.9690 | 34.8575 |
| SAC+HER+PER(full) | 4 | 1.00 | 1.00 (800000) | 0.668590 | -8.00 | 550000 | 7.9996 | 34.7240 |
| SAC+HER+PER(full) | 5 | 0.95 | 1.00 (925000) | 0.647436 | -10.90 | 575000 | 7.9872 | 34.7778 |

## Descriptive Comparison

SAC - TD3 mean AUC difference: +0.305983.
SAC - TD3 mean final-success difference: +0.330000.
All five paired seeds have positive SAC-minus-TD3 AUC and final-success differences.
These five-seed summaries are descriptive; this artifact makes no formal
statistical-significance or task-generalization claim.

## Evaluation and Packaging

TD3 was evaluated every 10,000 steps (100 points); SAC was evaluated every
25,000 steps (40 points). Native success AUC is verified by trapezoidal
integration over each method's exact evaluation grid, normalized by the
distance from its first to final evaluation step. No interpolation, smoothing,
extrapolation, or forward filling is used.

Checkpoints, full per-update metrics, JSONL files, and logs are intentionally
excluded. Packaged JSON replaces repository-local absolute paths with
`${REPO_ROOT}` and non-finite diagnostic values with JSON `null`.

# FPP Support vs TD-error Existing-Artifact Diagnostic

Generated: 2026-06-24T04:43:58+00:00

## Scope

This diagnostic uses existing `metrics_selected_10k.csv` files from FPP SABER runs.
It is a checkpoint-level aggregate diagnostic, not a sample-level diagnostic.

Available:

- checkpoint-level `lite/psi_q_rank_rate`
- checkpoint-level `loss/td_error_abs_mean` and `loss/td_error_abs_p90`
- checkpoint-level support coverage and success curves

Not available in current artifacts:

- per-sample `psi_i`
- per-sample `|delta_i|`
- top-k support vs TD-error overlap
- support-positive vs support-negative residual distributions
- before/after update residual reduction by support group

## Aggregate Correlations

| Method | n | support rate | TD abs mean | corr(support, TD mean) | Spearman | corr(support, TD p90) | corr(support, eval success) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FPP-DDPG-SABER | 5 | 0.367 +/- 0.011 | 0.560 +/- 0.050 | 0.670 +/- 0.078 | 0.679 +/- 0.060 | 0.682 +/- 0.064 | 0.442 +/- 0.123 |
| FPP-SAC-SABER | 5 | 0.197 +/- 0.013 | 0.276 +/- 0.019 | -0.463 +/- 0.138 | -0.359 +/- 0.104 | -0.453 +/- 0.126 | 0.758 +/- 0.045 |
| FPP-TD3-SABER | 5 | 0.219 +/- 0.009 | 0.356 +/- 0.027 | 0.290 +/- 0.092 | 0.325 +/- 0.168 | 0.362 +/- 0.086 | 0.678 +/- 0.152 |

## Per-seed Correlations

| Method | Seed | rows | support mean | TD mean | Pearson | Spearman | TD p90 Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FPP-DDPG-SABER | 1 | 100 | 0.368 | 0.631 | 0.556 | 0.594 | 0.603 |
| FPP-DDPG-SABER | 2 | 100 | 0.366 | 0.552 | 0.654 | 0.701 | 0.649 |
| FPP-DDPG-SABER | 3 | 100 | 0.375 | 0.580 | 0.759 | 0.730 | 0.776 |
| FPP-DDPG-SABER | 4 | 100 | 0.349 | 0.496 | 0.655 | 0.642 | 0.684 |
| FPP-DDPG-SABER | 5 | 100 | 0.376 | 0.543 | 0.725 | 0.730 | 0.701 |
| FPP-SAC-SABER | 1 | 100 | 0.217 | 0.244 | -0.691 | -0.516 | -0.665 |
| FPP-SAC-SABER | 2 | 100 | 0.188 | 0.281 | -0.466 | -0.383 | -0.442 |
| FPP-SAC-SABER | 3 | 100 | 0.190 | 0.281 | -0.443 | -0.366 | -0.441 |
| FPP-SAC-SABER | 4 | 100 | 0.202 | 0.294 | -0.335 | -0.264 | -0.354 |
| FPP-SAC-SABER | 5 | 100 | 0.188 | 0.280 | -0.378 | -0.265 | -0.363 |
| FPP-TD3-SABER | 1 | 100 | 0.217 | 0.377 | 0.333 | 0.549 | 0.389 |
| FPP-TD3-SABER | 2 | 100 | 0.219 | 0.347 | 0.256 | 0.217 | 0.322 |
| FPP-TD3-SABER | 3 | 100 | 0.233 | 0.350 | 0.147 | 0.111 | 0.233 |
| FPP-TD3-SABER | 4 | 100 | 0.208 | 0.317 | 0.331 | 0.347 | 0.419 |
| FPP-TD3-SABER | 5 | 100 | 0.218 | 0.387 | 0.384 | 0.399 | 0.447 |

## Interpretation

- Existing artifacts do not close the reviewer-critical sample-level claim.
- The aggregate checkpoint correlations are useful as a quick sanity check: support coverage and mean TD-error are not the same logged quantity.
- Because both support rate and TD-error drift over training, these correlations are confounded by learning progress and should not be presented as the main mechanism evidence.
- To close the mechanism claim, an instrumented rerun should log per-sample `psi_i`, `|delta_i|`, weights, phase, support count, and residual before/after the critic update for a small diagnostic stream.

## Recommended Next Diagnostic

Run an instrumented FPP DDPG SABER job for seeds 1--3 or 1--5 that writes a small sampled diagnostic table every 10k or 20k steps. The diagnostic table should contain per-sample rows, capped to a small random subset per checkpoint to avoid large artifacts.

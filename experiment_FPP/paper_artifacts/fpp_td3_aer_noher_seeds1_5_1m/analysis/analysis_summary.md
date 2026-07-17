# FPP TD3 no-HER AER paired analysis

Primary endpoint: trapezoidal `eval/success_mean` AUC on the exact 10k, 20k, ..., 1M grid, divided by 990k.

No interpolation, smoothing, forward filling, or best-checkpoint selection was used.

## Method summaries

| Method | n | AUC mean | AUC SD | Final mean | Best mean |
|---|---:|---:|---:|---:|---:|
| td3_er | 5 | 0.039343 | 0.001918 | 0.000000 | 0.170000 |
| td3_annealing_er | 5 | 0.039394 | 0.001847 | 0.000000 | 0.170000 |
| td3_aer | 5 | 0.039293 | 0.001969 | 0.000000 | 0.170000 |

## Paired AUC comparisons

Positive differences favor `td3_aer`.

| Comparison | Mean difference | 95% paired bootstrap CI | Exact sign-flip p | W/T/L |
|---|---:|---:|---:|---:|
| td3_aer_vs_td3_annealing_er | -0.000101 | [-0.000455, 0.000253] | 0.75 | 1/2/2 |
| td3_aer_vs_td3_er | -0.000051 | [-0.000152, 0.000000] | 1 | 0/4/1 |

The exact sign-flip test is two-sided and enumerates every paired sign assignment. The bootstrap resamples paired seed differences, not unpaired method values.

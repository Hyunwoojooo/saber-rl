# FPP SAC no-HER AER paired analysis

Primary endpoint: trapezoidal `eval/success_mean` AUC on the exact 10k, 20k, ..., 1M grid, divided by 990k.

No interpolation, smoothing, forward filling, or best-checkpoint selection was used.

## Method summaries

| Method | n | AUC mean | AUC SD | Final mean | Best mean |
|---|---:|---:|---:|---:|---:|
| sac_er | 5 | 0.063990 | 0.042061 | 0.130000 | 0.280000 |
| sac_annealing_er | 5 | 0.040808 | 0.002497 | 0.000000 | 0.170000 |
| sac_aer | 5 | 0.040000 | 0.002910 | 0.000000 | 0.170000 |

## Paired AUC comparisons

Positive differences favor `sac_aer`.

| Comparison | Mean difference | 95% paired bootstrap CI | Exact sign-flip p | W/T/L |
|---|---:|---:|---:|---:|
| sac_aer_vs_sac_annealing_er | -0.000808 | [-0.002929, 0.001010] | 0.625 | 1/1/3 |
| sac_aer_vs_sac_er | -0.023990 | [-0.062929, -0.001970] | 0.0625 | 0/0/5 |

The exact sign-flip test is two-sided and enumerates every paired sign assignment. The bootstrap resamples paired seed differences, not unpaired method values.

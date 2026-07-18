# FPP DDPG no-HER AER paired analysis

Primary endpoint: trapezoidal `eval/success_mean` AUC on the exact 10k, 20k, ..., 1M grid, divided by 990k.

No interpolation, smoothing, forward filling, or best-checkpoint selection was used.

## Method summaries

| Method | n | AUC mean | AUC SD | Final mean | Best mean |
|---|---:|---:|---:|---:|---:|
| ddpg_er | 5 | 0.147677 | 0.039279 | 0.460000 | 0.590000 |
| ddpg_annealing_er | 5 | 0.128586 | 0.111223 | 0.340000 | 0.450000 |
| ddpg_aer | 5 | 0.044444 | 0.002020 | 0.000000 | 0.190000 |

## Paired AUC comparisons

Positive differences favor `ddpg_aer`.

| Comparison | Mean difference | 95% paired bootstrap CI | Exact sign-flip p | W/T/L |
|---|---:|---:|---:|---:|
| ddpg_aer_vs_ddpg_annealing_er | -0.084141 | [-0.185859, -0.017778] | 0.0625 | 0/0/5 |
| ddpg_aer_vs_ddpg_er | -0.103232 | [-0.134293, -0.070000] | 0.0625 | 0/0/5 |

The exact sign-flip test is two-sided and enumerates every paired sign assignment. The bootstrap resamples paired seed differences, not unpaired method values.

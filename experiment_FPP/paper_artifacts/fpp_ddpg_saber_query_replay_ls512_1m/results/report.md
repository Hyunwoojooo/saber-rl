# FPP DDPG SABER Replay-Action Query results

The endpoint is normalized trapezoidal success AUC on exact observed 
10k, 20k, ..., 1M evaluations (100 points; denominator 990,000). No 
interpolation, smoothing, extrapolation, or forward filling is used.

Replay-action querying retains the sampled transition in the local replay pool and can therefore self-support.

| Method | AUC mean ± sample SD | Final mean ± sample SD | Best mean ± sample SD | Hit90 |
|---|---:|---:|---:|---:|
| DDPG+HER+SABER Main | 0.609091 ± 0.037778 | 0.960000 ± 0.041833 | 1.000000 ± 0.000000 | 5/5 |
| DDPG+HER+SABER Replay-Action Query | 0.597677 ± 0.026695 | 0.970000 ± 0.044721 | 1.000000 ± 0.000000 | 5/5 |

## Paired Main − Replay AUC

Mean delta: +0.011414; 95% paired percentile bootstrap CI [-0.028081, 0.058802] (10,000 resamples, seed 27027); positive seeds: 3/5; exact two-sided sign-flip p=0.812500 over 32 assignments.

## Query/support diagnostics

Blank legacy values are unavailable and are not imputed.

| Method | Context eligible | Support positive | Query present given context |
|---|---:|---:|---:|
| DDPG+HER+SABER Main |  | 0.366781 |  |
| DDPG+HER+SABER Replay-Action Query | 0.921937 | 0.504023 | 0.546504 |

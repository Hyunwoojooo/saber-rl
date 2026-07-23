# FPP DDPG+HER+SABER Actor-Only Scope Ablation

Primary endpoint: normalized trapezoidal `eval/success_mean` AUC on the exact observed 
10k–1M grid (100 points; denominator 990k). No interpolation, smoothing, or forward-fill.

| Arm | n | AUC mean +/- sample SD | Final mean +/- sample SD | Best mean +/- sample SD | Hit90 |
|---|---:|---:|---:|---:|---:|
| actor_only | 5 | 0.588889 +/- 0.031878 | 0.960000 +/- 0.022361 | 1.000000 +/- 0.000000 | 5/5 |
| critic_only | 5 | 0.609091 +/- 0.037778 | 0.960000 +/- 0.041833 | 1.000000 +/- 0.000000 | 5/5 |

## Paired primary result

Delta is actor-only minus canonical critic-only on seeds 1–5.

- Mean AUC delta: -0.020202
- Sample SD of paired deltas: 0.056385
- 95% paired bootstrap CI (10,000 resamples, RNG seed 27012027): [-0.063030, 0.024242]
- Positive/negative/zero seed deltas: 2/3/0
- Exact two-sided sign-flip p: 0.500000 (32 assignments)

Hit90 means the first observed 10k-grid step with success >= 0.90. Unreached runs remain missing; Hit90 step inference is complete-case and reports the omitted seeds explicitly.

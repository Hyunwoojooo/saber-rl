# B3 Critic-Only Runtime Optimization v8 Positive MSE LS512

FPP-only Stage 1 loss ablation built from `b3_critic_only_runtime_opt_v7_positive_huber_ls512`.

This variant keeps the positive-only additive SABER support bonus but removes Huber clipping:

```text
critic_td_loss = mse
saber_loss_mode = positive_huber_bonus
support_quality_mode = none
local_search_max = 512
numba_support_scan = true
```

The intended comparison is against v7 positive-huber and the stronger v4 ls512 baselines.

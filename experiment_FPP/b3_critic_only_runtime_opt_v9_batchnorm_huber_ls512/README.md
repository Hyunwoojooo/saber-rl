# B3 Critic-Only Runtime Optimization v9 BatchNorm Huber LS512

FPP-only Stage 1 loss ablation.

This variant keeps the original batch-normalized SABER weighting but changes the critic TD loss from MSE to Huber:

```text
critic_td_loss = huber
saber_loss_mode = batch_norm
support_quality_mode = none
local_search_max = 512
numba_support_scan = true
```

The intended comparison is v7 positive-huber and v8 positive-MSE, isolating whether Huber clipping itself hurts FPP.

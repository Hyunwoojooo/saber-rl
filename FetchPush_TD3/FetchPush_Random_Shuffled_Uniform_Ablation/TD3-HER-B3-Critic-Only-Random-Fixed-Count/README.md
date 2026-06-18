# TD3-HER-B3-Fully-Random-Fixed-Count

FetchPush TD3 + HER + critic-only fully random fixed-count reweighting.

This is a fully random control, not a support-matched control:

```text
method = b3_fully_random_fixed_count
weight_scope = critic_only
random_fixed_count_k = 60
lambda_support = 1.0
weighting_mode = batch_norm
```

At every update, exactly `k=60` samples in the batch are selected uniformly at
random and assigned raw weight `1 + lambda_support = 2.0`; all other samples get
raw weight `1.0`. The batch is then normalized with `batch_norm`, and the weight
is applied only to the critic loss.

This variant does not call the SABER/B3 support scanner and does not use SABER
support-positive counts.

Run from the repository root with the shared runner:

```bash
bash FetchPush_TD3/FetchPush_Random_Shuffled_Uniform_Ablation/run_td3_random_fixed_count_5parallel.sh
```

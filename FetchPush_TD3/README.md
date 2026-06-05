# FetchPush TD3+HER experiment snapshots

This directory contains TD3 versions of the FetchPush comparison matrix.

## Variants

- `TD3-HER-B0`: TD3 + HER baseline.
- `TD3-HER-B3-Critic-Only`: TD3 + HER + SABER/B3 critic-only local support weighting.
- `TD3-HER-PER-Style`: TD3 + HER + PER-style batch-local TD-error critic loss weighting.
- `TD3-HER-PER-Full`: TD3 + HER + full prioritized replay sampling with IS critic weighting.

All variants keep the FetchPush task, HER relabeling, sparse reward, replay size, batch size, evaluation protocol, and 1M-step training budget aligned with the DDPG FetchPush experiments.

## TD3 defaults

```text
policy_delay = 2
target_policy_noise = 0.2
target_noise_clip = 0.5
```

TD3 uses twin critics. SABER and PER critic weights are applied to both critic losses. Actor loss remains unweighted for the critic-only comparison variants.

## Run

Each folder has:

```powershell
cmd /c call start_1m_cuda_parallel2.cmd
cmd /c call start_100k_cuda_smoke.cmd
```

CUDA parallelism is fixed to 2 by default.

## Completed FetchPush TD3 Results

Completed 1M-step FetchPush runs used 5 seeds, sparse rewards, HER, CUDA, and
20-episode evaluations every 10k environment steps. The B3-Critic-Only row below
uses the original `TD3-HER-B3-Critic-Only/results` runs as the final TD3 B3
result.

| Method | Final success | Best success | Success AUC | Hit90 step | Speed | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TD3+HER B0 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.638 +/- 0.030 | 462k +/- 29k | 47.91 steps/s | 5.80 h |
| TD3+HER+B3-Critic-Only | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.662 +/- 0.039 | 438k +/- 35k | 37.79 steps/s | 7.35 h |
| TD3+HER+PER-Style | 0.990 +/- 0.022 | 1.000 +/- 0.000 | 0.683 +/- 0.022 | 436k +/- 34k | 39.85 steps/s | 6.97 h |
| TD3+HER+PER-Full | 0.960 +/- 0.022 | 1.000 +/- 0.000 | 0.684 +/- 0.023 | 390k +/- 24k | 41.64 steps/s | 6.67 h |

Metric notes:

- `Final success`: final evaluation success rate at 1M environment steps.
- `Best success`: best evaluation success rate observed during training.
- `Success AUC`: trapezoidal area under the evaluation success curve, normalized
  by the 1M-step budget.
- `Hit90 step`: first evaluation step where success reached at least 0.90.
- `Speed` and `Runtime`: mean wall-clock throughput and runtime across seeds.

Summary:

- B3-Critic-Only improves over the TD3+HER baseline in sample efficiency while
  preserving perfect final success: AUC improves from 0.638 to 0.662 and Hit90
  improves from 462k to 438k.
- PER-style baselines show the strongest sample-efficiency on FetchPush TD3.
  PER-Full reaches Hit90 earliest at 390k, but its final success is lower
  (0.960 +/- 0.022) than B0 and B3-Critic-Only.
- B3-Critic-Only adds moderate wall-clock overhead relative to B0: 37.79
  steps/s versus 47.91 steps/s.

Recommended interpretation:

```text
B3-Critic-Only improves over the TD3+HER baseline while preserving final
success, although PER-style prioritization achieves the strongest
sample-efficiency on FetchPush under TD3.
```

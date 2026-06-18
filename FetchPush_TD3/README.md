# FetchPush TD3+HER experiment snapshots

This directory contains TD3 versions of the FetchPush comparison matrix.

## Variants

- `TD3-HER-B0`: TD3 + HER baseline.
- `TD3-HER-B3-Critic-Only`: TD3 + HER + SABER/B3 critic-only local support weighting.
- `TD3-HER-B3-Actor-Only`: TD3 + HER + SABER/B3 actor-only local support weighting.
- `TD3-HER-B3-Actor-Critic`: TD3 + HER + SABER/B3 actor+critic local support weighting.
- `TD3-HER-PER-Style`: TD3 + HER + PER-style batch-local TD-error critic loss weighting.
- `TD3-HER-PER-Full`: TD3 + HER + full prioritized replay sampling with IS critic weighting.
- `TD3-HER-B3-Critic-Only-Random-Fixed-Count`: TD3 + HER + fully random fixed-count critic reweighting; no SABER support lookup.

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

## TD3 Support-Control Negative Controls

The support-control ablations keep the FetchPush TD3+HER+B3 critic-only update
path and perturb the mapping between support information and sample weights.
Random-Support preserves the number of B3 high-weight samples in each batch but
randomly reassigns those high weights. Shuffled-Weight preserves the weight
distribution but breaks the original sample-to-weight alignment.

| Method | Final success | Best success | Success AUC | Hit90 step | Speed | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.662 +/- 0.039 | 438k +/- 35k | 37.79 steps/s | 7.35 h |
| TD3+HER+B3-Critic-Only-Random-Support | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.654 +/- 0.041 | 450k +/- 32k | 34.65 steps/s | 8.02 h |
| TD3+HER+B3-Critic-Only-Shuffled-Weight | 0.970 +/- 0.067 | 1.000 +/- 0.000 | 0.679 +/- 0.053 | 428k +/- 79k | 34.70 steps/s | 8.01 h |

Shuffled-Weight seed-level summary:

| Metric | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Final success | 1.00 | 1.00 | 1.00 | 1.00 | 0.85 |
| Success AUC | 0.585 | 0.715 | 0.694 | 0.700 | 0.700 |
| Hit90 step | 560k | 410k | 360k | 380k | 430k |

Interpretation:

- Random-Support performs strongly, which suggests that part of the FetchPush
  TD3 gain comes from support-conditioned reweighting density and stochastic
  critic reweighting. The aligned B3-Critic-Only result remains slightly better
  than Random-Support in AUC and Hit90.
- Shuffled-Weight reaches the best AUC and Hit90 among these FetchPush
  support-control runs, but final success drops to 0.970 because seed 5 ends at
  0.85. This makes it a strong negative control, not a clean replacement for
  aligned B3-Critic-Only.
- Because FetchPush is close to saturated, these controls should be used to
  qualify the claim: FetchPush alone does not isolate semantic support
  alignment; harder tasks such as DrawerOpen are needed to separate the
  mechanism more clearly.

Prepared but not yet run:

```bash
bash FetchPush_TD3/FetchPush_Random_Shuffled_Uniform_Ablation/run_td3_random_fixed_count_5parallel.sh
```

The prepared `Random Fixed-Count` control uses `k=60`, fixed before training
from the completed FetchPush B3-Critic-Only mean high-weight rate. It does not
perform SABER support lookup and does not use support-positive counts.

## TD3 Actor-Scope Ablation

The Actor-Only and Actor+Critic ablations keep the same FetchPush TD3+HER+B3
configuration as the final B3-Critic-Only run and change only `weight_scope`.
Runs used 5 seeds, 1M steps, sparse rewards, HER, CUDA, and 20-episode
evaluations every 10k environment steps.

| Method | Weight scope | Final success | Best success | Success AUC | Hit90 step | Speed | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only | `critic_only` | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.662 +/- 0.039 | 438k +/- 35k | 37.79 steps/s | 7.35 h |
| TD3+HER+B3-Actor-Only | `actor_only` | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.626 +/- 0.032 | 472k +/- 38k | 37.88 steps/s | 7.35 h |
| TD3+HER+B3-Actor+Critic | `actor_critic` | 0.980 +/- 0.027 | 1.000 +/- 0.000 | 0.664 +/- 0.028 | 446k +/- 29k | 37.79 steps/s | 7.37 h |

Seed-level summary:

| Method | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actor-Only final success | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Actor-Only AUC | 0.593 | 0.662 | 0.646 | 0.638 | 0.592 |
| Actor-Only Hit90 | 510k | 430k | 470k | 440k | 510k |
| Actor+Critic final success | 0.95 | 1.00 | 1.00 | 0.95 | 1.00 |
| Actor+Critic AUC | 0.680 | 0.664 | 0.691 | 0.665 | 0.618 |
| Actor+Critic Hit90 | 430k | 440k | 470k | 410k | 480k |

Interpretation:

- Actor-Only reaches perfect final success on FetchPush, but it is weaker than
  Critic-Only in sample efficiency: lower AUC and later Hit90.
- Actor+Critic nearly matches Critic-Only sample efficiency, but its final
  success is slightly lower due to two seeds ending at 0.95.
- Together with DrawerOpen, this supports the actor-scope ablation conclusion
  that applying B3 support weights on the critic side is the more reliable core
  mechanism, while actor-only weighting is not sufficient to explain the gains.


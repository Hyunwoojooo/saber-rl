# FetchPush SAC+HER experiment snapshots

This directory contains SAC versions of the FetchPush comparison matrix.

## Variants

- `SAC-HER-B0`: SAC + HER baseline.
- `SAC-HER-B3-Critic-Only-FPP-Style`: SAC + HER + SABER/B3 critic-only
  local support weighting using the FPP-style optimized support path.
- `SAC-HER-PER-Style`: SAC + HER + PER-style batch-local TD-error critic
  loss weighting.
- `SAC-HER-PER-Full`: SAC + HER + full prioritized replay sampling with IS
  critic weighting.

All variants keep the FetchPush task, HER relabeling, sparse reward, replay
size, batch size, evaluation protocol, and 1M-step training budget aligned with
the TD3 and DDPG FetchPush experiments.

## SAC defaults

```text
actor = tanh Gaussian policy
critics = twin Q critics
entropy_alpha = 0.2
```

SAC uses a stochastic actor and entropy-regularized critic targets. SABER and
PER critic weights are applied to critic losses. Actor loss remains unweighted
for the critic-only comparison variants.

## Run

The full SAC sequence can be launched with:

```bash
bash FetchPush_SAC/run_all_sac_fetchpush_5parallel.sh
```

The completed run used an optimized continuation script after the B3 support
scanner optimization was aligned with the TD3/DDPG code path:

```bash
bash FetchPush_SAC/run_sac_from_b3_optimized_fetchpush_5parallel.sh
```

CUDA parallelism is fixed to 5 seeds per variant.

## Completed FetchPush SAC Results

Completed 1M-step FetchPush runs used 5 seeds, sparse rewards, HER, CUDA, and
20-episode evaluations every 10k environment steps. The B3-Critic-Only row below
uses the optimized `SAC-HER-B3-Critic-Only-FPP-Style/results_optimized256` runs
with `local_search_max=256`.

| Method | Final success | Best success | Success AUC | Hit90 step | Speed | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SAC+HER B0 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.783 +/- 0.018 | 244k +/- 43k | 32.78 steps/s | 8.47 h |
| SAC+HER+B3-Critic-Only opt256 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.765 +/- 0.018 | 276k +/- 17k | 28.14 steps/s | 9.87 h |
| SAC+HER+PER-Style | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.795 +/- 0.013 | 242k +/- 35k | 27.05 steps/s | 10.27 h |
| SAC+HER+PER-Full | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.840 +/- 0.012 | 182k +/- 20k | 29.21 steps/s | 9.51 h |

Metric notes:

- `Final success`: final evaluation success rate at 1M environment steps.
- `Best success`: best evaluation success rate observed during training.
- `Success AUC`: trapezoidal area under the evaluation success curve, normalized
  by the 1M-step budget.
- `Hit90 step`: first evaluation step where success reached at least 0.90.
- `Speed` and `Runtime`: mean wall-clock throughput and runtime across seeds.

Summary:

- All SAC variants reach perfect final success on FetchPush.
- B3-Critic-Only does not improve over the SAC+HER baseline in this setting:
  AUC drops from 0.783 to 0.765 and Hit90 moves later from 244k to 276k.
- PER-Full is the strongest SAC variant in sample efficiency. It reaches the
  highest AUC, 0.840, and the earliest Hit90 point, 182k.
- SAC-B3 adds wall-clock overhead relative to B0: 28.14 steps/s versus
  32.78 steps/s.

Recommended interpretation:

```text
SABER/B3-Critic-Only is not beneficial under the SAC backbone on FetchPush,
where full PER-style prioritization gives the strongest sample-efficiency.
These results suggest the B3 support-weighting effect is backbone-dependent and
is stronger for deterministic actor-critic backbones than for entropy-regularized
SAC.
```

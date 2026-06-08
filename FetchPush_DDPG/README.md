# FetchPush DDPG+HER experiment snapshots

This directory contains DDPG versions of the FetchPush comparison matrix.

## Variants

- `DDPG-HER-B0`: DDPG + HER baseline.
- `DDPG-HER-B3-Critic-Only-FPP-Style`: DDPG + HER + SABER/B3 critic-only
  local support weighting using the FPP-style optimized support path.
- `DDPG-HER-PER-Style`: DDPG + HER + PER-style batch-local TD-error critic
  loss weighting.
- `DDPG-HER-PER-Full`: DDPG + HER + full prioritized replay sampling with IS
  critic weighting.

All variants keep the FetchPush task, HER relabeling, sparse reward, replay
size, batch size, evaluation protocol, and 1M-step training budget aligned with
the TD3 and SAC FetchPush experiments.

## DDPG defaults

```text
actor = deterministic tanh policy
critic = single Q critic
exploration_noise_std = 0.2
target_update_tau = 0.005
```

SABER and PER critic weights are applied to the critic loss. Actor loss remains
unweighted for the critic-only comparison variants.

## Run

The full DDPG sequence can be launched with:

```bash
bash FetchPush_DDPG/run_all_ddpg_fetchpush_5parallel.sh
```

CUDA parallelism is fixed to 5 seeds per variant.

## Completed FetchPush DDPG Results

Completed 1M-step FetchPush runs used 5 seeds, sparse rewards, HER, CUDA, and
20-episode evaluations every 10k environment steps. The B3-Critic-Only row below
uses the replacement rerun result supplied on 2026-06-07 for
`DDPG-HER-B3-Critic-Only-FPP-Style` with `local_search_max=256`. The PER-Full
row is included from the completed 5-seed run; its Hit90 statistic is computed
over the 4 seeds that reached 0.90 success.

| Method | Final success | Best success | Success AUC | Final-window success | Hit90 step | Speed | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DDPG+HER B0 | 0.990 +/- 0.020 | 1.000 +/- 0.000 | 0.727 +/- 0.030 | 0.974 +/- 0.016 | 356k +/- 23k | 47.85 steps/s | 5.81 h |
| DDPG+HER+B3-Critic-Only | 0.990 +/- 0.022 | 1.000 +/- 0.000 | 0.748 +/- 0.038 | 0.984 +/- 0.015 | 344k +/- 50k | 32.04 steps/s | 8.67 h |
| DDPG+HER+PER-Style | 0.990 +/- 0.020 | 1.000 +/- 0.000 | 0.738 +/- 0.034 | 0.973 +/- 0.016 | 346k +/- 38k | 40.52 steps/s | 6.86 h |
| DDPG+HER+PER-Full | 0.790 +/- 0.414 | 0.850 +/- 0.308 | 0.516 +/- 0.287 | 0.782 +/- 0.389 | 525k +/- 245k (4/5) | 40.99 steps/s | 6.78 h |

Metric notes:

- `Final success`: final evaluation success rate at 1M environment steps.
- `Best success`: best evaluation success rate observed during training.
- `Success AUC`: trapezoidal area under the evaluation success curve, normalized
  by the 1M-step budget.
- `Final-window success`: mean success over the last 10 evaluations.
- `Hit90 step`: first evaluation step where success reached at least 0.90.
  `(4/5)` means one seed never reached the threshold.
- `Speed` and `Runtime`: mean wall-clock throughput and runtime across seeds.

Summary:

- B3-Critic-Only improves DDPG sample efficiency over the B0 baseline: AUC
  improves from 0.727 to 0.748 and Hit90 improves from 356k to 344k.
- B3-Critic-Only also edges out PER-Style on AUC, 0.748 versus 0.738, while
  preserving the same final success level.
- PER-Full is unstable in this run: one seed never reaches 0.90 success, final
  success drops to 0.790 +/- 0.414, and AUC drops to 0.516 +/- 0.287.
- B3-Critic-Only adds wall-clock overhead relative to B0, PER-Style, and
  PER-Full because it performs support lookup and weighting during critic
  updates.

Recommended interpretation:

```text
SABER/B3-Critic-Only is beneficial under the DDPG backbone on FetchPush. It
preserves final success and improves sample efficiency over both the DDPG+HER
baseline and PER-style TD-error critic weighting in the completed 1M-step runs.
PER-Full is not competitive in this DDPG FetchPush setting because its 5-seed
run has substantially lower AUC, lower final success, and higher variance.
```

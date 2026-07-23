# MetaWorld Drawer-Close-v3 TD3 Experiments

This folder contains TD3+HER experiment runners for MetaWorld `drawer-close-v3` with sparse reward, `reward_function_version=v2`, 2M steps, and seeds 1-5. The code is kept as an independent snapshot under `B3_v4_512_s1_5/code/src`.

## Experiment Set

| Experiment | Script | Method | Weight Scope |
|---|---|---|---|
| TD3+HER B0 baseline | `TD3-HER-B0-2M.sh` | `b0_ddpg_her` | `none` |
| TD3+HER+B3 Critic-Only | `TD3-HER-B3-Critic-Only-2M.sh` | `b3_support_runtime_opt` | `critic_only` |
| TD3+HER+B3 Random-Fixed-Count | `TD3-HER-B3-Random-Fixed-Count-2M.sh` | `b3_fully_random_fixed_count` | `critic_only` |
| TD3+HER+B3 Random-Support | `TD3-HER-B3-Random-Support-2M.sh` | `b3_support_random_runtime_opt` | `critic_only` |
| TD3+HER+B3 Shuffled-Weight | `TD3-HER-B3-Shuffled-Weight-2M.sh` | `b3_support_shuffled_runtime_opt` | `critic_only` |
| TD3+HER+PER-Style | `TD3-HER-PER-Style-2M.sh` | `per_style_loss_weighting` | `critic_only` |

## Run

From repository root:

```bash
bash metaworld_Drawer_Close_TD3/TD3-HER-B0-2M.sh
bash metaworld_Drawer_Close_TD3/TD3-HER-B3-Critic-Only-2M.sh
bash metaworld_Drawer_Close_TD3/TD3-HER-B3-Random-Fixed-Count-2M.sh
bash metaworld_Drawer_Close_TD3/TD3-HER-B3-Random-Support-2M.sh
bash metaworld_Drawer_Close_TD3/TD3-HER-B3-Shuffled-Weight-2M.sh
bash metaworld_Drawer_Close_TD3/TD3-HER-PER-Style-2M.sh
```

Outputs are written to `metaworld_Drawer_Close_TD3/results_2m/`, logs to `metaworld_Drawer_Close_TD3/logs_2m/`, and orchestrator logs to `metaworld_Drawer_Close_TD3/orchestrator_logs/`.

## Environment

- MetaWorld env: `drawer-close-v3`
- Reward type: `sparse`
- MetaWorld reward function version: `v2`
- Horizon: 200
- Seeds: 1-5
- Steps: 2,000,000
- HER ratio: 0.8
- CUDA parallelism: 5 processes on `CUDA_DEVICE=0` by default

## Results

### Completed Result Summary

| Experiment | Status | Final Success Seeds | Final Success | Best Success | AUC Success | Mean Hit90 | Speed |
|---|---|---:|---:|---:|---:|---:|---:|
| TD3-HER-B0-2M | Completed | 1/5 | 0.35 | 1.00 | 0.329 | 76k | 41.70 steps/s |
| TD3-HER-B3-Critic-Only-2M | Completed | 3/5 | 0.59 | 1.00 | 0.549 | 86k | 22.49 steps/s |
| TD3-HER-B3-Random-Fixed-Count-2M | Completed | 0/5 | 0.00 | 1.00 | 0.133 | 110k | 41.48 steps/s |
| TD3-HER-B3-Random-Support-2M | Completed | 1/5 | 0.20 | 1.00 | 0.296 | 118k | 22.41 steps/s |
| TD3-HER-B3-Shuffled-Weight-2M | Completed | 3/5 | 0.59 | 1.00 | 0.586 | 38k | 23.52 steps/s |
| TD3-HER-PER-Style-2M | Completed | 1/5 | 0.20 | 1.00 | 0.369 | 94k | 38.68 steps/s |
| TD3-HER-PER-Full-2M | Completed | 3/5 | 0.60 | 1.00 | 0.524 | 150k | 34.36 steps/s |

`TD3-HER-B0-2M` finished on 2026-06-28 KST. All five seeds reached 90% success at least once, but only one seed remained at or above 90% success in the final evaluation. Wall-clock runtime was about 13h 26m 37s for five parallel seeds, corresponding to roughly 206-208 aggregate environment steps/s.

`TD3-HER-B3-Critic-Only-2M` finished on 2026-06-29 KST. It improved final success from 0.35 to 0.59 and AUC from 0.329 to 0.549 over B0, but still showed seed instability: seeds 2, 3, and 4 finished successfully while seeds 1 and 5 collapsed by the final evaluation.

### TD3-HER-B0-2M

Completed on 2026-06-28 KST with 5 seeds, 2M steps per seed, sparse DrawerClose-v3, and 20 evaluation episodes every 10k steps.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 0.85 | 1.00 | 0.451 | 80k |
| 2 | 0.00 | 1.00 | 0.058 | 120k |
| 3 | 0.90 | 1.00 | 0.886 | 50k |
| 4 | 0.00 | 1.00 | 0.097 | 60k |
| 5 | 0.00 | 1.00 | 0.151 | 70k |
| Mean | 0.35 | 1.00 | 0.329 | 76k |

The baseline quickly found successful behavior in every seed, but final performance was unstable: seed 3 ended at 0.90, seed 1 ended at 0.85, and seeds 2, 4, and 5 ended at 0.00.

### TD3-HER-B3-Critic-Only-2M

Completed on 2026-06-29 KST with 5 seeds, 2M steps per seed, sparse DrawerClose-v3, and SABER/B3 support weighting applied to critic loss only.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 0.00 | 1.00 | 0.081 | 80k |
| 2 | 1.00 | 1.00 | 0.908 | 90k |
| 3 | 1.00 | 1.00 | 0.740 | 200k |
| 4 | 0.95 | 1.00 | 0.909 | 10k |
| 5 | 0.00 | 1.00 | 0.106 | 50k |
| Mean | 0.59 | 1.00 | 0.549 | 86k |

B3-Critic-Only improved the DrawerClose baseline on final success, final successful seed count, and AUC. The result still has high seed variance, but the successful seeds maintain much stronger learning curves than B0.


### TD3-HER-B3-Random-Fixed-Count-2M

Completed on 2026-06-29 KST with 5 seeds, 2M steps per seed, sparse DrawerClose-v3, and critic-only fixed-count random weighting. This control does not use support lookup; it applies the same number of high-weight samples per batch by random assignment.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 0.00 | 1.00 | 0.135 | 70k |
| 2 | 0.00 | 1.00 | 0.103 | 110k |
| 3 | 0.00 | 1.00 | 0.125 | 200k |
| 4 | 0.00 | 1.00 | 0.129 | 100k |
| 5 | 0.00 | 1.00 | 0.175 | 70k |
| Mean | 0.00 | 1.00 | 0.133 | 110k |

Random-Fixed-Count reached high success transiently in every seed, but all five seeds collapsed to 0.00 final success. Compared with B3-Critic-Only, this control has similar speed to B0 but much lower AUC and no stable final success, suggesting that DrawerClose gains are not explained by fixed-count random critic reweighting alone.

### TD3-HER-B3-Random-Support-2M

Completed on 2026-06-30 KST with 5 seeds, 2M steps per seed, sparse DrawerClose-v3, and critic-only density-matched random support weighting. This control computes the SABER support-positive count, but assigns the high weights to random batch samples rather than the support-positive samples themselves.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 | 0.924 | 80k |
| 2 | 0.00 | 1.00 | 0.239 | 60k |
| 3 | 0.00 | 1.00 | 0.119 | 220k |
| 4 | 0.00 | 1.00 | 0.105 | 50k |
| 5 | 0.00 | 1.00 | 0.091 | 180k |
| Mean | 0.20 | 1.00 | 0.296 | 118k |

Random-Support reached 90% success at least once in all seeds, but only seed 1 maintained high final performance. The mean AUC is below both B3-Critic-Only and B0, suggesting that preserving the support-derived density alone is not sufficient on DrawerClose; assigning critic weights to the aligned support-positive samples is more important for stable learning.

### TD3-HER-B3-Shuffled-Weight-2M

Completed on 2026-07-02 KST with 5 seeds, 2M steps per seed, sparse DrawerClose-v3, and critic-only shuffled SABER weighting. This control computes the same support-derived raw weight vector as B3-Critic-Only, but permutes the weight assignments across the batch before applying them to the critic loss.

This run was resumed after an interruption. The AUC, first-hit, best-success, and runtime values below were recomputed posthoc from the full `metrics.csv` history rather than using the resume-local final-row summary fields.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 0.00 | 1.00 | 0.452 | 10k |
| 2 | 0.95 | 1.00 | 0.774 | 100k |
| 3 | 1.00 | 1.00 | 0.758 | 30k |
| 4 | 1.00 | 1.00 | 0.819 | 30k |
| 5 | 0.00 | 1.00 | 0.127 | 20k |
| Mean | 0.59 | 1.00 | 0.586 | 38k |

Shuffled-Weight matched B3-Critic-Only on final success and produced a slightly higher mean AUC on this DrawerClose TD3 run. Because Shuffled still uses the SABER support lookup and preserves the support-derived weight distribution, this result should be interpreted as evidence that support-conditioned weight density/distribution can matter in addition to exact sample-level alignment. It does not support treating shuffled weighting as a fully random control.

### TD3-HER-PER-Style-2M

Completed on 2026-07-02 KST with 5 seeds, 2M steps per seed, sparse DrawerClose-v3, and critic-only PER-style TD-error loss weighting.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 0.00 | 1.00 | 0.177 | 340k |
| 2 | 0.00 | 1.00 | 0.206 | 30k |
| 3 | 0.00 | 1.00 | 0.219 | 40k |
| 4 | 0.00 | 1.00 | 0.409 | 30k |
| 5 | 1.00 | 1.00 | 0.832 | 30k |
| Mean | 0.20 | 1.00 | 0.369 | 94k |

PER-Style reached high success at least once in every seed, but only seed 5 maintained high final performance. Its mean AUC is higher than B0 and Random-Support, but below B3-Critic-Only and Shuffled-Weight on this DrawerClose TD3 run.

### TD3-HER-PER-Full-2M

Completed on 2026-07-24 KST with 5 seeds, 2M steps per seed, sparse
DrawerClose-v3, and full prioritized experience replay. Replay slots are
sampled by priority before future-goal HER relabeling, importance-sampling
weights are applied to both TD3 critic losses, and priorities are updated from
the mean absolute twin-critic TD error used by the optimizer step.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 0.00 | 1.00 | 0.111 | 220k |
| 2 | 1.00 | 1.00 | 0.845 | 190k |
| 3 | 1.00 | 1.00 | 0.825 | 120k |
| 4 | 0.00 | 1.00 | 0.112 | 120k |
| 5 | 1.00 | 1.00 | 0.725 | 100k |
| Mean | 0.60 | 1.00 | 0.524 | 150k |

Every seed reached 90% success transiently, and seeds 2, 3, and 5 retained
100% success at the final evaluation. Full PER improved over TD3+HER B0 in
both AUC and final success. Relative to B3-Critic-Only, it produced slightly
higher final success (0.60 versus 0.59) but lower AUC (0.524 versus 0.549),
showing that the final snapshot and whole-training efficiency favor different
methods in this cohort.



## Independent Experiment Packages

The following self-contained experiment package is grouped by task and backbone:

- [TD3-HER-PER-Full-2M](experiments/TD3-HER-PER-Full-2M/)

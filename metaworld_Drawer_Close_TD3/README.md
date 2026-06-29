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


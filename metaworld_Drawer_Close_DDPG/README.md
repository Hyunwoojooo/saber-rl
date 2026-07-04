# MetaWorld Drawer-Close-v3 DDPG Experiments

This folder contains an independent DDPG+HER code snapshot for MetaWorld `drawer-close-v3` with sparse reward, `reward_function_version=v2`, 2M steps, and seeds 1-5. It reuses the Drawer-Close TD3 environment/task/HER/SABER code path, but replaces the actor-critic backend with the single-critic DDPG agent used in `metaworld_Drawer_Open_DDPG`.

## Experiment Set

| Experiment | Script | Method | Weight Scope |
|---|---|---|---|
| DDPG+HER B0 baseline | `DDPG-HER-B0-2M.sh` | `b0_ddpg_her` | `none` |
| DDPG+HER+B3 Critic-Only | `DDPG-HER-B3-Critic-Only-2M.sh` | `b3_support_runtime_opt` | `critic_only` |
| DDPG+HER+B3 Random-Fixed-Count | `DDPG-HER-B3-Random-Fixed-Count-2M.sh` | `b3_fully_random_fixed_count` | `critic_only` |
| DDPG+HER+B3 Random-Support | `DDPG-HER-B3-Random-Support-2M.sh` | `b3_support_random_runtime_opt` | `critic_only` |
| DDPG+HER+B3 Shuffled-Weight | `DDPG-HER-B3-Shuffled-Weight-2M.sh` | `b3_support_shuffled_runtime_opt` | `critic_only` |
| DDPG+HER+PER-Style | `DDPG-HER-PER-Style-2M.sh` | `per_style_loss_weighting` | `critic_only` |

## Shared Settings

- MetaWorld env: `drawer-close-v3`
- Reward type: `sparse`
- MetaWorld reward function version: `v2`
- Horizon: 200
- Seeds: 1-5
- Steps: 2,000,000
- HER ratio: 0.8
- Batch size: 256
- Replay capacity: 300000
- CUDA parallelism: 5 processes on `CUDA_DEVICE=0` by default

## B3 Settings

- `M_B=16`
- `M_Q_min=4`
- `N_local_min=4`
- `N_gap_min=128`
- `local_search_max=512`
- `lambda_support=1.0`
- `weighting_mode=batch_norm`
- Numba Drawer support scanner enabled

## Run

From repository root:

```bash
bash metaworld_Drawer_Close_DDPG/DDPG-HER-B0-2M.sh
bash metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Critic-Only-2M.sh
bash metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Random-Fixed-Count-2M.sh
bash metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Random-Support-2M.sh
bash metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Shuffled-Weight-2M.sh
bash metaworld_Drawer_Close_DDPG/DDPG-HER-PER-Style-2M.sh
```

Run the full sequence:

```bash
bash metaworld_Drawer_Close_DDPG/run_all_ddpg_drawer_close_2m_5parallel.sh
```

For a short smoke check:

```bash
STEPS=1000 STEP_TAG=smoke CUDA_DEVICE=0 bash metaworld_Drawer_Close_DDPG/DDPG-HER-B0-2M.sh
```

Outputs are written to `metaworld_Drawer_Close_DDPG/results_2m/`, logs to `metaworld_Drawer_Close_DDPG/logs_2m/`, and orchestrator logs to `metaworld_Drawer_Close_DDPG/orchestrator_logs/`.

## Results

### Completed Result Summary

| Experiment | Status | Final Success Seeds | Final Success | Best Success | AUC Success | Mean Hit90 | Speed |
|---|---|---:|---:|---:|---:|---:|---:|
| DDPG-HER-B0-2M | Completed | 0/5 | 0.16 | 1.00 | 0.251 | 74k | 41.22 steps/s |
| DDPG-HER-B3-Critic-Only-2M | Completed | 0/5 | 0.28 | 1.00 | 0.324 | 30k | 23.96 steps/s |

### DDPG-HER-B0-2M

Completed on 2026-07-03 KST with 5 seeds, 2M steps per seed, sparse DrawerClose-v3, and 20 evaluation episodes every 10k steps.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 0.30 | 1.00 | 0.282 | 90k |
| 2 | 0.10 | 1.00 | 0.269 | 150k |
| 3 | 0.40 | 1.00 | 0.204 | 40k |
| 4 | 0.00 | 1.00 | 0.249 | 70k |
| 5 | 0.00 | 1.00 | 0.252 | 20k |
| Mean | 0.16 | 1.00 | 0.251 | 74k |

DDPG-HER B0 reached 90% success at least once in every seed, but none of the seeds maintained at least 90% success in the final evaluation. This confirms that DrawerClose-v3 sparse remains unstable for the plain DDPG+HER baseline even though transient successful policies appear early in training.

### DDPG-HER-B3-Critic-Only-2M

Completed on 2026-07-04 KST with 5 seeds, 2M steps per seed, sparse DrawerClose-v3, and SABER/B3 support weighting applied to critic loss only.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 0.65 | 1.00 | 0.288 | 10k |
| 2 | 0.20 | 1.00 | 0.337 | 60k |
| 3 | 0.15 | 1.00 | 0.284 | 50k |
| 4 | 0.00 | 1.00 | 0.461 | 20k |
| 5 | 0.40 | 1.00 | 0.252 | 10k |
| Mean | 0.28 | 1.00 | 0.324 | 30k |

B3-Critic-Only improved over DDPG-HER B0 on final success and AUC, but no seed maintained at least 90% success in the final evaluation. The early hit90 values show that support-weighted critic learning finds successful behavior quickly, while final policy stability remains weak on sparse DrawerClose-v3.


# MetaWorld DrawerOpen DDPG Experiments

This folder is an independent DrawerOpen DDPG code snapshot. It reuses the existing sparse MetaWorld DrawerOpen-v3 goal wrapper, HER replay path, and SABER/B3 support-weighting implementation from the TD3 DrawerOpen code, but replaces the actor-critic backend with single-critic DDPG.

## Experiments

- `DDPG-HER-B0-2M.sh`: DDPG + HER baseline with unit actor/critic weights.
- `DDPG-HER-B3-Critic-Only-2M.sh`: DDPG + HER + SABER/B3 support weighting applied to critic loss only.
- `DDPG-HER-B3-Actor-Only-2M.sh`: DDPG + HER + SABER/B3 support weighting applied to actor loss only.
- `DDPG-HER-B3-Actor-Critic-2M.sh`: DDPG + HER + SABER/B3 support weighting applied to both actor and critic losses.
- `DDPG-HER-B3-Random-Fixed-Count-2M.sh`: DDPG + HER + fixed-count random critic reweighting with no SABER support lookup.
- `DDPG-HER-B3-Shuffled-Weight-2M.sh`: DDPG + HER + SABER/B3 support weights shuffled across the batch before critic-only assignment.
- `run_all_ddpg_drawer_open_2m_5parallel.sh`: convenience runner for B0 followed by B3-Critic-Only.
- `run_ddpg_drawer_open_actor_ablation_2m_5parallel.sh`: convenience runner for Actor-Only followed by Actor+Critic.
- `run_ddpg_drawer_open_ablation_2m_5parallel.sh`: convenience runner for Random-Fixed-Count followed by Shuffled-Weight.

## Shared Settings

- Environment: `drawer-open-v3`
- Reward: sparse
- Episode horizon: 200
- Seeds: 1-5
- Training budget: 2M steps by default
- Parallelism: 5 seeds in parallel
- HER ratio: 0.8
- Batch size: 256
- Replay capacity: 300000
- Device: CUDA by default

## B3-Critic-Only Settings

- Method: `b3_support_runtime_opt`
- Weight scope: `critic_only`
- `M_B=16`
- `M_Q_min=4`
- `N_local_min=4`
- `N_gap_min=128`
- `local_search_max=512`
- `lambda_support=1.0`
- `weighting_mode=batch_norm`
- Numba DrawerOpen support scanner enabled

## Results

### Completed Result Summary

| Experiment | Reweighting Signal | Scope | Success Seeds | Final Success | Best Success | AUC Success | Mean Hit90 | Speed |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DDPG-HER-B0-2M | None | - | 2/5 | 0.39 | 0.40 | 0.207 | 960k* | 41.05 steps/s |
| DDPG-HER-B3-Critic-Only-2M | SABER support | Critic only | 5/5 | 1.00 | 1.00 | 0.754 | 482k | 23.33 steps/s |
| DDPG-HER-B3-Random-Fixed-Count-2M | Fixed-count random | Critic only | 4/5 | 0.80 | 0.80 | 0.621 | 423k* | 41.00 steps/s |

`*` Mean Hit90 is computed over successful seeds only. B3-Critic-Only is the only completed DDPG DrawerOpen variant that solved all 5 seeds and has the highest AUC, while Random-Fixed-Count shows that critic reweighting alone can help but remains weaker than support-aligned SABER weighting.

### DDPG-HER-B0-2M

Completed on 2026-06-23 KST with 5 seeds, 2M steps per seed, sparse DrawerOpen-v3, and 20 evaluation episodes every 10k steps.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.00 | 1.00 | 0.460 | 1090k |
| 2 | 0.00 | 0.00 | 0.000 | - |
| 3 | 0.00 | 0.00 | 0.000 | - |
| 4 | 0.00 | 0.00 | 0.000 | - |
| 5 | 0.95 | 1.00 | 0.576 | 830k |
| Mean | 0.39 | 0.40 | 0.207 | - |

DDPG+HER B0 learned in 2 of 5 seeds and failed to acquire success in seeds 2-4 under the sparse DrawerOpen setting.

### DDPG-HER-B3-Critic-Only-2M

Completed on 2026-06-24 KST with the same 5 seeds, 2M steps per seed, sparse DrawerOpen-v3, and 20 evaluation episodes every 10k steps.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.00 | 1.00 | 0.757 | 510k |
| 2 | 1.00 | 1.00 | 0.787 | 420k |
| 3 | 1.00 | 1.00 | 0.705 | 580k |
| 4 | 1.00 | 1.00 | 0.713 | 500k |
| 5 | 1.00 | 1.00 | 0.806 | 400k |
| Mean | 1.00 | 1.00 | 0.754 | 482k |

DDPG+HER+B3-Critic-Only solved all 5 seeds and reached 90% success between 400k and 580k steps, substantially improving both seed reliability and AUC over DDPG+HER B0.

### DDPG-HER-B3-Random-Fixed-Count-2M

Completed on 2026-06-24 KST with 5 seeds, 2M steps per seed, sparse DrawerOpen-v3, and fixed-count random critic reweighting. This control uses `random_fixed_count_k=122`, applies critic-only batch-normalized weights, and does not use SABER support lookup for selecting weighted samples.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.00 | 0.00 | 0.000 | - |
| 2 | 1.00 | 1.00 | 0.818 | 370k |
| 3 | 1.00 | 1.00 | 0.689 | 610k |
| 4 | 1.00 | 1.00 | 0.839 | 280k |
| 5 | 1.00 | 1.00 | 0.760 | 430k |
| Mean | 0.80 | 0.80 | 0.621 | 423k* |

`*` The mean first-hit value is computed over the four successful seeds. Random-Fixed-Count improved over DDPG+HER B0, but remained below SABER/B3-Critic-Only in seed reliability and AUC.

## Actor-Scope Ablation Settings

`DDPG-HER-B3-Actor-Only-2M.sh` and `DDPG-HER-B3-Actor-Critic-2M.sh` match the TD3 actor-scope ablations:

- Method: `b3_support_runtime_opt`
- `DDPG-HER-B3-Actor-Only-2M.sh` uses `weight_scope=actor_only`
- `DDPG-HER-B3-Actor-Critic-2M.sh` uses `weight_scope=actor_critic`
- Same B3 support lookup, support cache, Numba scanner, and local-search settings as B3-Critic-Only.

## Random/Shuffled Ablation Settings

`DDPG-HER-B3-Random-Fixed-Count-2M.sh` matches the TD3 fixed-count control:

- Method: `b3_fully_random_fixed_count`
- Weight scope: `critic_only`
- `random_fixed_count_k=122`
- `lambda_support=1.0`
- `weighting_mode=batch_norm`
- No SABER support lookup is used for selecting weighted samples.

`DDPG-HER-B3-Shuffled-Weight-2M.sh` matches the TD3 shuffled-weight control:

- Method: `b3_support_shuffled_runtime_opt`
- Weight scope: `critic_only`
- Same B3 support lookup, support cache, Numba scanner, and local-search settings as B3-Critic-Only.
- SABER raw support weights are computed first, then shuffled across the batch before loss weighting.

## Run

```bash
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B0-2M.sh
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Critic-Only-2M.sh
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Actor-Only-2M.sh
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Actor-Critic-2M.sh
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Random-Fixed-Count-2M.sh
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Shuffled-Weight-2M.sh
```

Run B0 and B3-Critic-Only sequentially:

```bash
bash metaworld_Drawer_Open_DDPG/run_all_ddpg_drawer_open_2m_5parallel.sh
```

Run the actor-scope ablations sequentially:

```bash
bash metaworld_Drawer_Open_DDPG/run_ddpg_drawer_open_actor_ablation_2m_5parallel.sh
```

Run the random/shuffled ablations sequentially:

```bash
bash metaworld_Drawer_Open_DDPG/run_ddpg_drawer_open_ablation_2m_5parallel.sh
```

To override the budget for smoke checks:

```bash
STEPS=1000 STEP_TAG=smoke CUDA_DEVICE=0 bash metaworld_Drawer_Open_DDPG/DDPG-HER-B0-2M.sh
```

Outputs are written under `metaworld_Drawer_Open_DDPG/results_2m/` and logs under `metaworld_Drawer_Open_DDPG/logs_2m/` by default.

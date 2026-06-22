# MetaWorld DrawerOpen DDPG Experiments

This folder is an independent DrawerOpen DDPG code snapshot. It reuses the existing sparse MetaWorld DrawerOpen-v3 goal wrapper, HER replay path, and SABER/B3 support-weighting implementation from the TD3 DrawerOpen code, but replaces the actor-critic backend with single-critic DDPG.

## Experiments

- `DDPG-HER-B0-2M.sh`: DDPG + HER baseline with unit actor/critic weights.
- `DDPG-HER-B3-Critic-Only-2M.sh`: DDPG + HER + SABER/B3 support weighting applied to critic loss only.
- `run_all_ddpg_drawer_open_2m_5parallel.sh`: convenience runner for B0 followed by B3-Critic-Only.

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

## Run

```bash
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B0-2M.sh
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Critic-Only-2M.sh
```

Run both sequentially:

```bash
bash metaworld_Drawer_Open_DDPG/run_all_ddpg_drawer_open_2m_5parallel.sh
```

To override the budget for smoke checks:

```bash
STEPS=1000 STEP_TAG=smoke CUDA_DEVICE=0 bash metaworld_Drawer_Open_DDPG/DDPG-HER-B0-2M.sh
```

Outputs are written under `metaworld_Drawer_Open_DDPG/results_2m/` and logs under `metaworld_Drawer_Open_DDPG/logs_2m/` by default.

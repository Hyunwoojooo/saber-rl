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

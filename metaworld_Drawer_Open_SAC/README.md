# MetaWorld DrawerOpen SAC Experiments

Independent SAC+HER DrawerOpen experiment snapshot. This folder is intentionally separate from `metaworld_Drawer_Open` so SAC code changes do not mutate the TD3 DrawerOpen code path.

## Variants

The main runner executes five variants sequentially. Each variant runs seeds 1-5 in parallel on CUDA.

| Order | Variant | Method | Weight scope | Steps |
|---:|---|---|---|---:|
| 1 | SAC+HER B0 | `b0_ddpg_her` | `none` | 2M |
| 2 | SAC+HER+B3-Critic-Only | `b3_support_runtime_opt` | `critic_only` | 2M |
| 3 | SAC+HER+B3-Actor-Only | `b3_support_runtime_opt` | `actor_only` | 2M |
| 4 | SAC+HER+B3-Actor+Critic | `b3_support_runtime_opt` | `actor_critic` | 2M |
| 5 | SAC+HER+PER-Full | `per_full_baseline` | `critic_only` | 2M |

## Run

```bash
bash metaworld_Drawer_Open_SAC/run_all_sac_drawer_open_2m_5parallel.sh
```

Optional overrides:

```bash
CUDA_DEVICE=0 STEPS=2000000 bash metaworld_Drawer_Open_SAC/run_all_sac_drawer_open_2m_5parallel.sh
```

## Outputs

- Results: `metaworld_Drawer_Open_SAC/results_2m/<variant>/`
- Logs: `metaworld_Drawer_Open_SAC/logs_2m/<variant>/`
- Orchestrator logs: `metaworld_Drawer_Open_SAC/orchestrator_logs/`

## Notes

- DrawerOpen env/task/HER/B3 support scanner logic is copied from the TD3 DrawerOpen implementation.
- The agent backend is SAC: stochastic Gaussian actor, twin critics, entropy-regularized target, fixed `entropy_alpha=0.2`.
- `PER-Full` uses priority replay sampling plus IS critic weights and updates priorities from SAC TD-error after each update.

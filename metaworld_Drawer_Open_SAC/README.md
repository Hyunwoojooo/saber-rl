# MetaWorld DrawerOpen SAC Experiments

## Completed Auto-Temperature SAC Results

The post-AAAI FPP-reference SAC comparison is complete: **HER, SABER (Ours),
HGR and full PER, seeds 1-5, 2M steps each (20 unique runs)**.

See [the compact results and protocol](paper_artifacts/drawer_open_sac_her_saber_hgr_per_2m/README.md)
for seed-level metrics, full evaluation curves, paired CIs, execution environments,
source provenance and SHA256 verification. Unfavorable and zero-success results are
retained. These results use automatic entropy-temperature tuning and must not be
pooled with the fixed-temperature prototype below.

Exact executed code is preserved in the [HER/SABER source](../ICML_27/experiments/draweropen_drawerclose_sac_her_vs_saber/source/)
and [HGR/PER source](../ICML_27/experiments/draweropen_drawerclose_sac_hgr_per/source/).
The inherited observation/goal representation caveat is documented with the results.

```bash
python3 ICML_27/tools/sac_results.py verify --task drawer_open
```

Run verification from the repository root with Python 3.12 and NumPy. It does not train a model.

## Legacy Fixed-Temperature Prototype

The remaining instructions and results describe the older fixed-alpha prototype.
Its `code/` and launchers are preserved unchanged; they did not produce the
auto-temperature results linked above.

Independent SAC+HER DrawerOpen experiment snapshot. This folder is intentionally separate from `metaworld_Drawer_Open_TD3` so SAC code changes do not mutate the TD3 DrawerOpen code path.

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

## SAC-HER-B0-2M Results

Status as of 2026-06-20 07:08 KST:

- `SAC-HER-B0-2M` completed 5 seeds at 2M steps.
- All seeds used the sparse DrawerOpen goal wrapper, SAC+HER backbone,
  `drawer-open-v3`, `reward_function_version=v2`, and 20 evaluation episodes.
- The run did not solve the task in any seed.

Aggregate metrics use 5 seeds:

| Variant | Final success | Best success | AUC success | Hit90 | Speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAC-HER-B0-2M | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | none, 0/5 seeds | 31.72 +/- 0.25 steps/s |

Seed-level final success:

| seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Seed-level best success:

| seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Interpretation:

- SAC-HER B0 failed to learn DrawerOpen under this 2M sparse-reward setting.
- The mean wall-clock runtime was about 17.5 hours per seed group with 5 seeds
  running in parallel on CUDA.
- These results make SAC-HER B0 a weak backbone on DrawerOpen compared with the
  completed TD3 DrawerOpen runs.

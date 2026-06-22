# AAAI-27 Core Artifacts

This branch is the working artifact branch for the AAAI-27 paper.

Repository source of truth:

- GitHub repository: `https://github.com/Hyunwoojooo/saber-rl`
- Main branch paths: `https://github.com/Hyunwoojooo/saber-rl/tree/main/<path>`
- FetchSlide artifact branch paths: `https://github.com/Hyunwoojooo/saber-rl/tree/fetchslide-runtime-artifacts/<path>`

Use the paths below first when deriving paper tables, plots, and claims.

## FetchPush

Source branch: `main`

| Experiment | Source code | Results |
| --- | --- | --- |
| DDPG-HER-B0 | `FetchPush_DDPG/DDPG-HER-B0/code` | `FetchPush_DDPG/DDPG-HER-B0/results` |
| DDPG-HER-B3-Critic-Only-FPP-Style | `FetchPush_DDPG/DDPG-HER-B3-Critic-Only-FPP-Style/code` | `FetchPush_DDPG/DDPG-HER-B3-Critic-Only-FPP-Style/results` |
| SAC-HER-B0 | `FetchPush_SAC/SAC-HER-B0/code` | no results folder in GitHub checkout |
| SAC-HER-B3-Critic-Only-FPP-Style | `FetchPush_SAC/SAC-HER-B3-Critic-Only-FPP-Style/code` | no results folder in GitHub checkout |
| TD3-HER-B0 | `FetchPush_TD3/TD3-HER-B0/code` | no results folder in GitHub checkout |
| TD3-HER-B3-Critic-Only | `FetchPush_TD3/TD3-HER-B3-Critic-Only/code` | `FetchPush_TD3/TD3-HER-B3-Critic-Only/results` |

## FetchPickAndPlace

Source branch: `main`

| Experiment | Source code | Results |
| --- | --- | --- |
| B0 | `experiment_FPP/B0` | `experiment_FPP/runs/B0` |
| b3_critic_only_runtime_opt_v4_ls512 | `experiment_FPP/b3_critic_only_runtime_opt_v4_ls512` | `experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512` |
| SAC HER Baseline LS512 | `experiment_FPP/sac_her_baseline_ls512` | `experiment_FPP/runs/sac_her_baseline_ls512` |
| SAC SABER Critic-Only V4 LS512 | `experiment_FPP/sac_saber_critic_only_v4_ls512` | `experiment_FPP/runs/sac_saber_critic_only_v4_ls512` |
| TD3 HER Baseline LS512 | `experiment_FPP/td3_her_baseline_ls512` | `experiment_FPP/runs/td3_her_baseline_ls512` |
| TD3 HER SABER Critic-Only V4 LS512 | `experiment_FPP/td3_saber_critic_only_v4_ls512` | `experiment_FPP/runs/td3_saber_critic_only_v4_ls512` |

Aggregate FPP results live under `experiment_FPP/results/`.

## FetchSlide

Source branch: `fetchslide-runtime-artifacts`

Common entrypoint/source:

- `experiment_FetchSlide/src/saber_fetch_slide/experiment.py`
- `experiment_FetchSlide/src/saber_fetch_slide/td3_agent.py`
- `experiment_FetchSlide/src/saber_fetch_slide/sac_agent.py`

| Experiment | Source selector | Results |
| --- | --- | --- |
| FetchSlide B0 / DDPG+HER | method `b0` | `experiment_FetchSlide/experiment/FetchSlide/b0` |
| FetchSlide b3_critic_only_runtime_opt_v5 | method `b3_critic_only_runtime_opt_v5` | `experiment_FetchSlide/experiment/FetchSlideRuntimeOptV5Main/b3_critic_only_runtime_opt_v5` |
| FetchSlide SAC baseline | method `sac` | `experiment_FetchSlide/experiment/FetchSlide/sac` |
| FetchSlide SAC SABER | method `sac_saber` | `experiment_FetchSlide/experiment/FetchSlide/sac_saber` |
| td3_saber_critic_only_v4_ls512 | method `td3_saber_critic_only_v4_ls512` | `experiment_FetchSlide/experiment/FetchSlide/td3_saber_critic_only_v4_ls512` |
| FetchSlide TD3 SABER | method `td3_saber` | `experiment_FetchSlide/experiment/FetchSlide/td3_saber` |

FetchSlide reports:

- `analysis/FetchSlide/fetchslide_selected_1m_results.md`
- `experiment_FetchSlide/reports/FetchSlide/agent_family_results.md`

Additional raw paths imported because the reports reference them directly:

- `experiment/FetchSlide/b0`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5`
- `experiment_FetchSlide/experiment/FetchSlide/td3`

Notes:

- `experiment_FetchSlide/experiment/FetchSlide/b0` contains `summary.json`, `config.json`, `env_snapshot.json`, and logs for the selected 1M runs; the report-level raw `metrics.csv` for B0 is under `experiment/FetchSlide/b0`.
- `experiment_FetchSlide/experiment/FetchSlide/td3_saber_critic_only_v4_ls512` contains selected 1M summaries/configs/logs, but no `metrics.csv` files were present in the source branch for those run directories.
- Some `best_checkpoint.json` files contain absolute machine paths to checkpoint `.pt` files. The checkpoint binaries are not tracked here; use `metrics.csv`, `summary.json`, `config.json`, and reports as the paper artifacts.

## MetaWorld DrawerOpen

Source branch: `main`

| Experiment | Source code | Results |
| --- | --- | --- |
| DrawerOpen TD3 B0 | `metaworld_Drawer_Open_TD3/B3_v4_512_s1_5` plus `metaworld_Drawer_Open_TD3/TD3-HER-B0-2M.sh` | 1M: `metaworld_Drawer_Open_TD3/results/TD3-HER-B0`; 2M: `metaworld_Drawer_Open_TD3/results_2m/TD3-HER-B0-2M` |
| DrawerOpen TD3 v4 512 | `metaworld_Drawer_Open_TD3/B3_v4_512_s1_5` | use with B3 Critic-Only results |
| DrawerOpen TD3 B3 Critic-Only | `metaworld_Drawer_Open_TD3/B3_v4_512_s1_5` plus `metaworld_Drawer_Open_TD3/TD3-HER-B3-Critic-Only-2M.sh` | 1M: `metaworld_Drawer_Open_TD3/results/TD3-HER-B3-Critic-Only`; 2M: `metaworld_Drawer_Open_TD3/results_2m/TD3-HER-B3-Critic-Only-2M` |


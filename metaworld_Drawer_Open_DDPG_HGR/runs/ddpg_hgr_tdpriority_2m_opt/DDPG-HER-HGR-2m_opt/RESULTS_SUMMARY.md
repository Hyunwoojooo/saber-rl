# DrawerOpen DDPG-HER-HGR 2M Results

Final optimized run:

- Experiment: `DDPG-HER-HGR-2m_opt`
- Task: `drawer_open`
- Seeds: `1,2,3,4,5`
- Steps per seed: `2,000,000`
- Source folder: `metaworld_Drawer_Open_DDPG_HGR`
- Artifact folder: `metaworld_Drawer_Open_DDPG_HGR/runs/ddpg_hgr_tdpriority_2m_opt/DDPG-HER-HGR-2m_opt`

Per-seed final metrics:

| seed | final step | final success | best success | AUC | last-quarter AUC | Hit90 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,000,000 | 1.00 | 1.00 | 0.783 | 0.999 | 430k |
| 2 | 2,000,000 | 0.00 | 0.00 | 0.000 | 0.000 | - |
| 3 | 2,000,000 | 1.00 | 1.00 | 0.771 | 0.985 | 470k |
| 4 | 2,000,000 | 0.00 | 0.00 | 0.000 | 0.000 | - |
| 5 | 2,000,000 | 0.00 | 0.00 | 0.000 | 0.000 | - |

Aggregate:

- Final success: `0.40 +/- 0.55`
- Best success: `0.40 +/- 0.55`
- AUC: `0.311 +/- 0.425`
- Last-quarter AUC: `0.397 +/- 0.543`
- Hit90 reached: `2/5`
- Median Hit90 among reached seeds: `450k`
- Mean runtime per seed: `16.05h`
- Mean speed: `34.61 steps/sec`
- Mean HGR sampling time: `0.012 sec/update`
- Mean HGR relabel fraction: `0.797`
- HGR priority clip rate: `0.0`

Interpretation:

HGR succeeded on seeds 1 and 3, reaching Hit90 at 430k and 470k steps. Seeds 2, 4, and 5 did not learn by 2M steps. The result is therefore high-variance: HGR can solve DrawerOpen quickly when it works, but it is substantially less stable than the existing DrawerOpen DDPG SABER/B3 Critic-Only result.

Pushed artifacts intentionally exclude large binary checkpoints and raw high-volume metric streams:

- Excluded: `checkpoint.pt`, `best_checkpoint.pt`
- Excluded: raw `metrics.csv`, `metrics.jsonl`
- Included: `summary.json`, `config.json`, `env_snapshot.json`, `run.log`, `metrics_selected_10k.csv`, `best_checkpoint.json`

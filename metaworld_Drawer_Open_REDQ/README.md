# MetaWorld DrawerOpen REDQ Baselines

This folder contains the independent REDQ+HER strong baseline for `drawer-open-v3`.

The planned REDQ baseline for the SABER comparison is `REDQ-HER-TQB-B0-2M`.
The plain `REDQ-HER-B0-2M.sh` script is kept only as a legacy/reference launcher and is not part of the planned DrawerOpen REDQ result set.

## Planned Experiment

| Variant | Backbone | HER | Reward | Steps | Seeds | Parallelism | Target Q Bounding |
|---|---|---:|---|---:|---:|---:|---|
| REDQ-HER-TQB-B0-2M | REDQ | 0.8 | sparse / v2 | 2,000,000 | 1-5 | 5 CUDA workers | [-50, 0] |

## Completed Result Summary

`REDQ-HER-TQB-B0-2M` completed on 2026-07-12. The comparison row for `DDPG-HER-B3-Critic-Only-2M` is copied from `metaworld_Drawer_Open_DDPG` for direct readability.

| Experiment | Reweighting Signal | Scope | Success Seeds | Final Success | Best Success | AUC Success | Mean Hit90 | Speed |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| REDQ-HER-TQB-B0-2M | REDQ ensemble + TQB | Actor-critic baseline | 5/5 | 1.00 | 1.00 | 0.672 | 644k | 5.00 steps/s |
| DDPG-HER-B3-Critic-Only-2M | SABER support | Critic only | 5/5 | 1.00 | 1.00 | 0.754 | 482k | 23.33 steps/s |

REDQ+HER+TQB is a strong compute-scaled baseline: all 5 seeds reached Hit90, but it used `updates_per_step=4` and a 10-critic ensemble. In this DrawerOpen run, SABER/B3-Critic-Only reached higher AUC and earlier mean Hit90 with roughly 4.7x higher observed environment-step throughput.

### REDQ-HER-TQB-B0-2M Per-Seed Results

The run used 5 parallel CUDA workers, REDQ ensemble size 10, target subset size 2, `updates_per_step=4`, HER ratio 0.8, SAC auto-alpha, and target Q bounding `[-50, 0]`.

| Seed | Final Success | Final Env Success | Best Success | AUC Success | First Hit 90% | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.00 | 0.85 | 1.00 | 0.819 | 370k | 112.49h |
| 2 | 1.00 | 0.75 | 1.00 | 0.897 | 230k | 110.05h |
| 3 | 1.00 | 0.80 | 1.00 | 0.511 | 980k | 109.40h |
| 4 | 1.00 | 0.80 | 1.00 | 0.936 | 150k | 110.78h |
| 5 | 1.00 | 1.00 | 1.00 | 0.196 | 1490k | 112.53h |
| Mean | 1.00 | 0.84 | 1.00 | 0.672 | 644k | 111.05h |

## REDQ Settings

- Actor: SAC-style tanh Gaussian policy
- Critic ensemble size: 10 for `REDQ-HER-TQB-B0-2M`
- REDQ target subset size: 2
- Entropy coefficient: SAC auto-alpha enabled, initialized from `alpha = 0.2`
- Target entropy: `-4.0` for the 4-D MetaWorld drawer action space
- Update-to-data ratio: `UPDATES_PER_STEP=4` for `REDQ-HER-TQB-B0-2M`
- Target backup: min over a random target-critic subset
- Actor objective: ensemble-mean Q with entropy regularization

## Target Q Bounding

The drawer sparse reward convention is `failure = -1` and `success = 0`. With `gamma = 0.98`, the infinite-horizon feasible target range is approximately:

```text
[-1 / (1 - gamma), 0] = [-50, 0]
```

`REDQ-HER-TQB-B0-2M.sh` clips the REDQ target-subset Q backup before the Bellman target is formed.

## Run

```bash
bash REDQ-HER-TQB-B0-2M.sh
```

Optional overrides:

```bash
STEPS=2000000 CUDA_DEVICE=0 UPDATES_PER_STEP=4 bash REDQ-HER-TQB-B0-2M.sh
REDQ_ENSEMBLE_SIZE=10 TARGET_SUBSET_SIZE=2 bash REDQ-HER-TQB-B0-2M.sh
TARGET_Q_MIN=-50 TARGET_Q_MAX=0 SAC_AUTO_ALPHA=true bash REDQ-HER-TQB-B0-2M.sh
```

Results are written under `results_2m/<variant>`, and logs under `logs_2m/<variant>`.

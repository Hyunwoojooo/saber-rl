# MetaWorld DrawerClose REDQ Baselines

This folder contains the independent REDQ+HER strong baseline for `drawer-close-v3`.

The planned REDQ baseline for the SABER comparison is `REDQ-HER-TQB-B0-2M`.
The plain `REDQ-HER-B0-2M.sh` script is kept only as a legacy/reference launcher and is not part of the planned DrawerClose REDQ result set.

## Planned Experiment

| Variant | Backbone | HER | Reward | Steps | Seeds | Parallelism | Target Q Bounding |
|---|---|---:|---|---:|---:|---:|---|
| REDQ-HER-TQB-B0-2M | REDQ | 0.8 | sparse / v2 | 2,000,000 | 1-5 | 5 CUDA workers | [-50, 0] |

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

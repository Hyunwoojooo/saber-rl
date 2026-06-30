# MetaWorld DrawerOpen REDQ Baselines

This folder contains independent REDQ+HER strong baselines for `drawer-open-v3`.

## Experiments

| Variant | Backbone | HER | Reward | Steps | Seeds | Parallelism | Target Q Bounding |
|---|---|---:|---|---:|---:|---:|---|
| REDQ-HER-B0-2M | REDQ | 0.8 | sparse / v2 | 2,000,000 | 1-5 | 5 CUDA workers | off |
| REDQ-HER-TQB-B0-2M | REDQ | 0.8 | sparse / v2 | 2,000,000 | 1-5 | 5 CUDA workers | [-50, 0] |

## REDQ Settings

- Actor: SAC-style tanh Gaussian policy
- Critic ensemble size: 5
- REDQ target subset size: 2
- Entropy coefficient: fixed `alpha = 0.2`
- Update-to-data ratio: `UPDATES_PER_STEP=5` by default in the launchers
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
bash REDQ-HER-B0-2M.sh
bash REDQ-HER-TQB-B0-2M.sh
```

Optional overrides:

```bash
STEPS=2000000 CUDA_DEVICE=0 UPDATES_PER_STEP=5 bash REDQ-HER-TQB-B0-2M.sh
TARGET_Q_MIN=-50 TARGET_Q_MAX=0 bash REDQ-HER-TQB-B0-2M.sh
```

Results are written under `results_2m/<variant>`, and logs under `logs_2m/<variant>`.

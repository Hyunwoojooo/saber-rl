# MetaWorld DrawerClose REDQ Baselines

This folder contains the independent REDQ+HER strong baseline for `drawer-close-v3`.

The planned REDQ baseline for the SABER comparison is `REDQ-HER-TQB-B0-2M`.
The plain `REDQ-HER-B0-2M.sh` script is kept only as a legacy/reference launcher and is not part of the planned DrawerClose REDQ result set.

## Planned Experiment

| Variant | Backbone | HER | Reward | Steps | Seeds | Parallelism | Target Q Bounding |
|---|---|---:|---|---:|---:|---:|---|
| REDQ-HER-TQB-B0-2M | REDQ | 0.8 | sparse / v2 | 2,000,000 | 1-5 | 5 CUDA workers | [-50, 0] |

## Completed Result Summary

`REDQ-HER-TQB-B0-2M` completed on 2026-07-28 from source commit
`0c72f95707e0befb19190a56c778332c1e2bbeb4`. All five seeds completed
2,000,000 environment steps, 7,980,004 updates, and the exact 200-point
10k-to-2M evaluation grid. The SABER row is copied from the verified
`drawer_close_ddpg_saber_context_eligible_count_matched_2m` artifact for
readability.

| Experiment | Scope | Hit90 Seeds | Final Env Success | Best Env Success | Normalized AUC | Median Hit90 | Mean Speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| REDQ-HER-TQB-B0-2M | REDQ actor-critic baseline | 5/5 | 0.10 | 1.00 | 0.188 | 20k | 5.82 steps/s |
| DDPG-HER-B3-Critic-Only-2M | SABER critic-only | 5/5 | 0.28 | 1.00 | 0.324 | 20k | 23.96 steps/s |

### REDQ-HER-TQB-B0-2M Per-Seed Results

Metrics use `eval/env_success_mean` on the exact observed grid, without
smoothing, interpolation, or forward filling. AUC is the trapezoidal integral
over 10k--2M divided by 1,990,000.

| Seed | Final Env Success | Best Env Success | Normalized AUC | First Hit90 | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.25 | 1.00 | 0.163 | 100k | 95.50h |
| 2 | 0.00 | 1.00 | 0.252 | 20k | 95.33h |
| 3 | 0.00 | 1.00 | 0.158 | 10k | 95.61h |
| 4 | 0.05 | 1.00 | 0.202 | 30k | 95.36h |
| 5 | 0.20 | 1.00 | 0.162 | 20k | 95.07h |
| Mean | 0.10 | 1.00 | 0.188 | 36k | 95.38h |

REDQ reached high success early in every seed but did not retain it through
the full run. This comparison is not a one-factor SABER ablation: REDQ uses
four updates per environment step and a 10-critic ensemble, while the SABER
row uses a single-critic DDPG backbone with one update per step.

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

## Paper Artifact

The compact paper artifact is generated from the repository root with:

    python paper/analysis/build_drawerclose_redq_reference.py

It is written to
`metaworld_Drawer_Close_REDQ/paper_artifacts/drawer_close_redq_her_tqb_utd4_2m/`.
The package contains exact-grid seed and aggregate summaries, validation
metadata, and source manifests. Raw run outputs are referenced by
repository-relative path and SHA256 rather than duplicated.

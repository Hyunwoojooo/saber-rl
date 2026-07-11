# MetaWorld DrawerOpen DDPG Experiments

This folder is an independent DrawerOpen DDPG code snapshot. It reuses the existing sparse MetaWorld DrawerOpen-v3 goal wrapper, HER replay path, and SABER/B3 support-weighting implementation from the TD3 DrawerOpen code, but replaces the actor-critic backend with single-critic DDPG.

## Experiments

- `DDPG-HER-B0-2M.sh`: DDPG + HER baseline with unit actor/critic weights.
- `DDPG-HER-B3-Critic-Only-2M.sh`: DDPG + HER + SABER/B3 support weighting applied to critic loss only.
- `DDPG-HER-B3-Actor-Only-2M.sh`: DDPG + HER + SABER/B3 support weighting applied to actor loss only.
- `DDPG-HER-B3-Actor-Critic-2M.sh`: DDPG + HER + SABER/B3 support weighting applied to both actor and critic losses.
- `DDPG-HER-B3-Random-Fixed-Count-2M.sh`: DDPG + HER + fixed-count random critic reweighting with no SABER support lookup.
- `DDPG-HER-B3-Random-Support-2M.sh`: DDPG + HER + density-matched random critic reweighting using SABER support-positive count but random sample assignment.
- `DDPG-HER-B3-Shuffled-Weight-2M.sh`: DDPG + HER + SABER/B3 support weights shuffled across the batch before critic-only assignment.
- `DDPG-HER-PER-Style-2M.sh`: DDPG + HER + TD-error-based PER-style critic loss weighting.
- `run_all_ddpg_drawer_open_2m_5parallel.sh`: convenience runner for B0 followed by B3-Critic-Only.
- `run_ddpg_drawer_open_actor_ablation_2m_5parallel.sh`: convenience runner for Actor-Only followed by Actor+Critic.
- `run_ddpg_drawer_open_ablation_2m_5parallel.sh`: convenience runner for Random-Fixed-Count followed by Shuffled-Weight.

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

## Results

### Completed Result Summary

| Experiment | Reweighting Signal | Scope | Success Seeds | Final Success | Best Success | AUC Success | Mean Hit90 | Speed |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DDPG-HER-B0-2M | None | - | 2/5 | 0.39 | 0.40 | 0.207 | 960k* | 41.05 steps/s |
| DDPG-HER-B3-Critic-Only-2M | SABER support | Critic only | 5/5 | 1.00 | 1.00 | 0.754 | 482k | 23.33 steps/s |
| DDPG-HER-B3-Critic-Only-2M diagnostic seeds 6-10 | SABER support | Critic only | 4/5 | 0.79 | 0.80 | 0.550 | 623k* | 18.45 steps/s |
| DDPG-HER-B3-Random-Fixed-Count-2M | Fixed-count random | Critic only | 4/5 | 0.80 | 0.80 | 0.621 | 423k* | 41.00 steps/s |
| DDPG-HER-B3-Random-Support-2M | SABER density, random assignment | Critic only | 5/5 | 1.00 | 1.00 | 0.706 | 594k | 22.61 steps/s |
| DDPG-HER-B3-Shuffled-Weight-2M | SABER support shuffled | Critic only | 2/5 | 0.40 | 0.42 | 0.307 | 460k* | 22.95 steps/s |
| DDPG-HER-PER-Style-2M | TD-error PER-style | Critic only | 4/5 | 0.80 | 1.00 | 0.715 | 578k* | 39.02 steps/s |
| DDPG-HER-HGR-2M | TD-error hindsight goal ranking | HER goal selection | 2/5 | 0.40 | 0.40 | 0.311 | 450k* | 34.61 steps/s |

`*` Mean Hit90 is computed over successful seeds only. B3-Critic-Only and Random-Support both solved all 5 original seeds, but B3-Critic-Only has the highest AUC and earlier mean Hit90. The B3-Critic-Only diagnostic seed 6-10 run solved 4 of 5 additional seeds while logging support-positive rate, TD-error enrichment, and one-step residual correction diagnostics without changing replay sampling, loss weights, optimizer steps, or policy updates. PER-Style reached the second-highest AUC among the completed DDPG controls, but its final success was less stable because seed 2 regressed to 0.00 by the end. Random-Fixed-Count shows that critic reweighting alone can help but remains weaker than support-derived controls. Shuffled-Weight preserves the SABER-generated support-weight distribution but breaks sample-level assignment, and it remains substantially below B3-Critic-Only. HGR is tracked in `metaworld_Drawer_Open_DDPG_HGR` as a HER goal-prioritization baseline rather than a critic-loss reweighting method; it solved 2 of 5 seeds and remained less stable than B3-Critic-Only.

Runtime note: `DDPG-HER-B3-Shuffled-Weight-2M` was interrupted and resumed from checkpoints. The committed per-seed `summary.json` and `metrics.csv` files have been corrected by combining original and resumed logs, dropping original rows after each resume checkpoint, and recomputing runtime from original-start to resumed-finish timestamps. The corrected mean speed is `22.95 steps/s`.

### DDPG-HER-B0-2M

Completed on 2026-06-23 KST with 5 seeds, 2M steps per seed, sparse DrawerOpen-v3, and 20 evaluation episodes every 10k steps.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.00 | 1.00 | 0.460 | 1090k |
| 2 | 0.00 | 0.00 | 0.000 | - |
| 3 | 0.00 | 0.00 | 0.000 | - |
| 4 | 0.00 | 0.00 | 0.000 | - |
| 5 | 0.95 | 1.00 | 0.576 | 830k |
| Mean | 0.39 | 0.40 | 0.207 | - |

DDPG+HER B0 learned in 2 of 5 seeds and failed to acquire success in seeds 2-4 under the sparse DrawerOpen setting.

### DDPG-HER-B3-Critic-Only-2M

Completed on 2026-06-24 KST with the same 5 seeds, 2M steps per seed, sparse DrawerOpen-v3, and 20 evaluation episodes every 10k steps.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.00 | 1.00 | 0.757 | 510k |
| 2 | 1.00 | 1.00 | 0.787 | 420k |
| 3 | 1.00 | 1.00 | 0.705 | 580k |
| 4 | 1.00 | 1.00 | 0.713 | 500k |
| 5 | 1.00 | 1.00 | 0.806 | 400k |
| Mean | 1.00 | 1.00 | 0.754 | 482k |

DDPG+HER+B3-Critic-Only solved all 5 seeds and reached 90% success between 400k and 580k steps, substantially improving both seed reliability and AUC over DDPG+HER B0.

### DDPG-HER-B3-Critic-Only-2M Diagnostic Seeds 6-10

Completed on 2026-07-11 UTC with additional seeds 6-10, 2M steps per seed, sparse DrawerOpen-v3, 20 evaluation episodes every 10k steps, and online diagnostic logging enabled. The diagnostic logging records support-positive rate, TD-error enrichment, and one-step residual correction metrics; it does not change replay sampling, loss weights, optimizer steps, or policy updates.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
| --- | ---: | ---: | ---: | ---: |
| 6 | 1.00 | 1.00 | 0.766 | 470k |
| 7 | 0.00 | 0.00 | 0.000 | - |
| 8 | 1.00 | 1.00 | 0.576 | 830k |
| 9 | 0.95 | 1.00 | 0.571 | 850k |
| 10 | 1.00 | 1.00 | 0.837 | 340k |
| Mean | 0.79 | 0.80 | 0.550 | 623k* |

`*` The mean first-hit value is computed over the four seeds that reached 90%. The full diagnostic artifact summary is stored at `results_2m/DDPG-HER-B3-Critic-Only-2M/RESULTS_SUMMARY_S6_10_DIAGNOSTIC.md`.

### DDPG-HER-B3-Random-Fixed-Count-2M

Completed on 2026-06-24 KST with 5 seeds, 2M steps per seed, sparse DrawerOpen-v3, and fixed-count random critic reweighting. This control uses `random_fixed_count_k=122`, applies critic-only batch-normalized weights, and does not use SABER support lookup for selecting weighted samples.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.00 | 0.00 | 0.000 | - |
| 2 | 1.00 | 1.00 | 0.818 | 370k |
| 3 | 1.00 | 1.00 | 0.689 | 610k |
| 4 | 1.00 | 1.00 | 0.839 | 280k |
| 5 | 1.00 | 1.00 | 0.760 | 430k |
| Mean | 0.80 | 0.80 | 0.621 | 423k* |

`*` The mean first-hit value is computed over the four successful seeds. Random-Fixed-Count improved over DDPG+HER B0, but remained below SABER/B3-Critic-Only in seed reliability and AUC.

### DDPG-HER-B3-Random-Support-2M

Completed on 2026-06-26 KST with 5 seeds, 2M steps per seed, sparse DrawerOpen-v3, and density-matched random critic reweighting. This control computes SABER support-positive samples to determine the high-weight count, but assigns the high weights to random batch samples instead of the original support-positive samples.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% | Final High-Weight Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.00 | 1.00 | 0.791 | 460k | 0.477 |
| 2 | 1.00 | 1.00 | 0.666 | 680k | 0.496 |
| 3 | 1.00 | 1.00 | 0.513 | 970k | 0.527 |
| 4 | 1.00 | 1.00 | 0.812 | 350k | 0.449 |
| 5 | 1.00 | 1.00 | 0.751 | 510k | 0.355 |
| Mean | 1.00 | 1.00 | 0.706 | 594k | 0.461 |

Random-Support solved all 5 seeds, indicating that SABER-derived support density can provide a useful curriculum-like critic reweighting signal even when exact sample assignment is randomized. It still remained below B3-Critic-Only on AUC and mean Hit90.

### DDPG-HER-B3-Shuffled-Weight-2M

Completed on 2026-06-25 KST with 5 seeds, 2M steps per seed, sparse DrawerOpen-v3, and SABER/B3 support weights shuffled across the batch before critic-only loss weighting. This control preserves the SABER support-weight distribution while breaking the original support-positive sample assignment.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.00 | 0.10 | 0.001 | - |
| 2 | 0.00 | 0.00 | 0.000 | - |
| 3 | 1.00 | 1.00 | 0.781 | 370k |
| 4 | 1.00 | 1.00 | 0.752 | 550k |
| 5 | 0.00 | 0.00 | 0.000 | - |
| Mean | 0.40 | 0.42 | 0.307 | 460k* |

`*` The mean first-hit value is computed over the two successful seeds. The AUC and first-hit values above are recomputed from the combined original and resumed logs, dropping original log entries after each resume checkpoint. Shuffled-Weight solved fewer seeds and had lower mean AUC than B3-Critic-Only, supporting the interpretation that support-weight alignment to the correct samples matters beyond preserving the weight distribution alone. Among the two successful seeds, learning occurred early, but the method failed in the other three seeds.

Runtime correction: this experiment was resumed after interruption. The committed `summary.json` and `metrics.csv` files now store corrected AUC, first-hit, and final runtime values reconstructed from the original plus resumed logs. Pre-correction runtime counters reported 90-218 steps/s because they divided the full 2M step count by resume-only elapsed time; the corrected mean speed is 22.95 steps/s.

### DDPG-HER-PER-Style-2M

Completed on 2026-06-27 KST with 5 seeds, 2M steps per seed, sparse DrawerOpen-v3, and TD-error-based PER-style critic loss weighting.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.00 | 1.00 | 0.760 | 510k |
| 2 | 0.00 | 1.00 | 0.842 | 300k |
| 3 | 1.00 | 1.00 | 0.664 | 670k |
| 4 | 1.00 | 1.00 | 0.607 | 800k |
| 5 | 1.00 | 1.00 | 0.704 | 610k |
| Mean | 0.80 | 1.00 | 0.715 | 578k* |

`*` The mean first-hit value is computed over the five seeds that reached 90% at least once. PER-Style produced strong AUC but weaker final stability than B3-Critic-Only, because seed 2 reached 90% early and later collapsed by the final evaluation.

## Actor-Scope Ablation Settings

`DDPG-HER-B3-Actor-Only-2M.sh` and `DDPG-HER-B3-Actor-Critic-2M.sh` match the TD3 actor-scope ablations:

- Method: `b3_support_runtime_opt`
- `DDPG-HER-B3-Actor-Only-2M.sh` uses `weight_scope=actor_only`
- `DDPG-HER-B3-Actor-Critic-2M.sh` uses `weight_scope=actor_critic`
- Same B3 support lookup, support cache, Numba scanner, and local-search settings as B3-Critic-Only.

## Random/Shuffled Ablation Settings

`DDPG-HER-B3-Random-Fixed-Count-2M.sh` matches the TD3 fixed-count control:

- Method: `b3_fully_random_fixed_count`
- Weight scope: `critic_only`
- `random_fixed_count_k=122`
- `lambda_support=1.0`
- `weighting_mode=batch_norm`
- No SABER support lookup is used for selecting weighted samples.

`DDPG-HER-B3-Random-Support-2M.sh` matches the density-matched random support control:

- Method: `b3_support_random_runtime_opt`
- Weight scope: `critic_only`
- Same B3 support lookup, support cache, Numba scanner, and local-search settings as B3-Critic-Only.
- SABER support-positive count is computed first, then the same number of high weights is randomly assigned across the batch.

`DDPG-HER-B3-Shuffled-Weight-2M.sh` matches the TD3 shuffled-weight control:

- Method: `b3_support_shuffled_runtime_opt`
- Weight scope: `critic_only`
- Same B3 support lookup, support cache, Numba scanner, and local-search settings as B3-Critic-Only.
- SABER raw support weights are computed first, then shuffled across the batch before loss weighting.

## PER-Style Settings

`DDPG-HER-PER-Style-2M.sh` uses TD-error-based critic loss weighting:

- Method: `per_style_loss_weighting`
- Weight scope: `critic_only`
- `per_alpha=0.6`
- `eps_PER=1e-6`
- `U_PER_max=2.0`
- `weighting_mode=batch_norm`

## Run

```bash
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B0-2M.sh
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Critic-Only-2M.sh
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Actor-Only-2M.sh
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Actor-Critic-2M.sh
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Random-Fixed-Count-2M.sh
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Random-Support-2M.sh
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Shuffled-Weight-2M.sh
bash metaworld_Drawer_Open_DDPG/DDPG-HER-PER-Style-2M.sh
```

Run B0 and B3-Critic-Only sequentially:

```bash
bash metaworld_Drawer_Open_DDPG/run_all_ddpg_drawer_open_2m_5parallel.sh
```

Run the actor-scope ablations sequentially:

```bash
bash metaworld_Drawer_Open_DDPG/run_ddpg_drawer_open_actor_ablation_2m_5parallel.sh
```

Run the random/shuffled ablations sequentially:

```bash
bash metaworld_Drawer_Open_DDPG/run_ddpg_drawer_open_ablation_2m_5parallel.sh
```

To override the budget for smoke checks:

```bash
STEPS=1000 STEP_TAG=smoke CUDA_DEVICE=0 bash metaworld_Drawer_Open_DDPG/DDPG-HER-B0-2M.sh
```

Outputs are written under `metaworld_Drawer_Open_DDPG/results_2m/` and logs under `metaworld_Drawer_Open_DDPG/logs_2m/` by default.

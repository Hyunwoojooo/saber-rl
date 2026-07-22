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
- `experiments/DDPG-HER-PER-Full-2M/`: DDPG + HER + slot-level prioritized replay with critic-only importance-sampling correction.
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
| DDPG-HER-PER-Full-2M | TD-error priority + IS | Replay sampling / critic IS | 5/5 | 0.99 | 1.00 | 0.825 | 346k | 33.98 steps/s |
| DDPG-HER-HGR-2M Full | Episode/pair TD-error ranking + IS | HER goal selection | 4/5 | 0.90 | 1.00 | 0.688 | 608k | 13.24 steps/s |

`*` Mean Hit90 is computed over seeds that reached the threshold. Full PER has the highest AUC and earliest mean Hit90 among the completed DrawerOpen DDPG runs in this table. It changes replay sampling and applies critic-only importance-sampling correction, whereas SABER retains uniform replay and changes only critic-loss allocation. B3-Critic-Only and Random-Support both solved all 5 original seeds; B3-Critic-Only remains the strongest support-conditioned critic-loss variant. The B3-Critic-Only diagnostic seed 6-10 run solved 4 of 5 additional seeds while logging support-positive rate, TD-error enrichment, and one-step residual correction diagnostics without changing replay sampling, loss weights, optimizer steps, or policy updates. PER-Style reached AUC 0.715 but its final success was less stable because seed 2 regressed to 0.00 by the end. Random-Fixed-Count shows that critic reweighting alone can help but remains weaker than support-derived controls. Shuffled-Weight preserves the SABER-generated support-weight distribution but breaks sample-level assignment, and it remains substantially below B3-Critic-Only. The HGR row uses the Full HGR completed-episode pair-priority implementation with episode/pair importance correction. Full HGR reached Hit90 on 5/5 seeds and has AUC 0.688; SABER has the higher mean AUC at 0.754, but the paired HGR-minus-SABER AUC interval `[-0.219, 0.047]` includes zero. HGR is therefore a strong matched complete-method baseline, not evidence of the same mechanism as SABER.

Detailed Korean 10-seed diagnosis report: [`DIAGNOSIS_B3_CRITIC_ONLY_10SEED_KO.md`](DIAGNOSIS_B3_CRITIC_ONLY_10SEED_KO.md). It covers support/TD-error enrichment, residual correction, phase-level behavior, the seed 7 failure path, statistical limitations, scanner issues, and follow-up priorities.

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

### DDPG-HER-PER-Full-2M

Completed on 2026-07-21 KST with 5 seeds, 2M steps per seed, sparse DrawerOpen-v3, future-goal HER probability 0.8, and 20 evaluation episodes every 10k steps. Full PER samples replay slots according to TD-error priorities, applies future-goal HER after slot sampling, and uses beta-annealed importance-sampling correction on the critic loss.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.00 | 1.00 | 0.860 | 330k |
| 2 | 1.00 | 1.00 | 0.877 | 270k |
| 3 | 1.00 | 1.00 | 0.814 | 330k |
| 4 | 0.95 | 1.00 | 0.809 | 390k |
| 5 | 1.00 | 1.00 | 0.765 | 410k |
| Mean | 0.99 | 1.00 | 0.825 | 346k |

Full PER reached Hit90 on all five seeds. Relative to critic-only SABER on the same seeds, its paired mean AUC difference is `+0.071`; a 10,000-resample paired seed bootstrap gives an approximate 95% interval of `[+0.014, +0.104]`, with Full PER higher on 4 of 5 seeds. This establishes Full PER as the stronger DrawerOpen DDPG result in mean AUC and Hit90, while the methods intervene at different stages: Full PER changes replay sampling and corrects the critic with IS weights, whereas SABER preserves uniform replay and reallocates critic loss within the sampled HER minibatch.

Committed raw results contain only `metrics.csv` and `summary.json` for each seed under `results_2m/DDPG-HER-PER-Full-2M/`; model checkpoints are excluded.

### DDPG-HER-HGR Pair-Priority + IS 2M

The official DrawerOpen HGR result uses completed-episode transition/future-goal pair priorities,
two-stage episode/pair sampling, beta-annealed importance correction at both stages, and the
matched DDPG+HER backbone. This is the sole DrawerOpen DDPG HGR result reported in this README.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.50 | 1.00 | 0.800 | 410k |
| 2 | 1.00 | 1.00 | 0.705 | 530k |
| 3 | 1.00 | 1.00 | 0.692 | 610k |
| 4 | 1.00 | 1.00 | 0.795 | 430k |
| 5 | 1.00 | 1.00 | 0.449 | 1060k |
| Mean | 0.90 | 1.00 | 0.688 | 608k |

Against matched HER, the paired mean AUC difference is `+0.481` with a 95% bootstrap interval
of `[0.151, 0.738]`. Against critic-only SABER, the HGR-minus-SABER difference is `-0.065` with
an interval of `[-0.219, 0.047]`. Thus Full HGR clearly improves the local HER baseline, while
the five-seed comparison does not establish a difference between Full HGR and SABER. HGR changes
sampling and applies importance correction; SABER keeps the sampled HER minibatch fixed and
changes only critic-loss allocation.

Compact artifact and verification code:
[`drawer_open_ddpg_hgr_pair_priority_is_2m`](../metaworld_Drawer_Open_DDPG_HGR/paper_artifacts/drawer_open_ddpg_hgr_pair_priority_is_2m/README.md).

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

## Full PER Settings

`experiments/DDPG-HER-PER-Full-2M/` implements standard slot-level prioritized replay on the matched DDPG+HER backbone:

- Replay priority exponent: `alpha=0.6`
- IS exponent: `beta=0.4` to `1.0` over 2M steps
- Priority epsilon: `1e-6`
- Replay slots are prioritized before future-goal HER relabeling
- Critic loss receives IS correction; the actor loss is not directly weighted
- Priorities are updated from the single critic's absolute TD error
- SABER support lookup and PER-style loss-only weighting are disabled

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
bash metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-PER-Full-2M/run_5seeds_parallel.sh --steps 2000000
bash metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Context-Count-Only-2M.sh
```

## Context-count-only descriptive control

`DDPG-HER-B3-Context-Count-Only-2M.sh` implements **Context eligibility only (count-unmatched descriptive control)**. For each HER-relabeled query it uses the existing DrawerOpen state-goal-horizon contexts and deterministic recent tail of at most 512 replay transitions. A sample is positive when `n_local >= 4` and `min(n_local, 16) >= 4`. Current-policy/replay action membership, Q values, TD errors, and random assignment are not used.

Raw critic weights are `1 + psi_ctx` and are normalized by their minibatch mean; actor weights stay one. This is not the strict count-matched control: every eligible sample is positive, so its positive count and normalized-weight histogram may differ from SABER Main. The launcher accepts `SEEDS`, `STEPS`, `STEP_TAG`, `CUDA_DEVICE`, `OUTPUT_ROOT`, and `LOG_ROOT`; the shared trainer defaults to `support_query_mode=current_policy` for backward compatibility.

With predicate early exit enabled, the context-only local-count diagnostic is threshold-capped eligibility evidence (it may stop at the fourth exact match), not a full local-population count; do not use it to claim population-size statistics.

For a clean checkout on another machine, set `PYTHON` to the target virtual environment when `.venv/bin/python` is unavailable. Before a full run, execute `code/src/test_context_count_only_drawer_open.py`, `code/src/test_context_count_only_ddpg_loss_scope.py`, `validate_context_count_only_config_parity.py`, and `freeze_context_count_only_source.py --verify` with that interpreter. The source snapshot used here was validated with NumPy 2.3.5, Numba 0.66.0, llvmlite 0.48.0, PyTorch 2.12.1, Gymnasium 1.3.0, Gymnasium-Robotics 1.4.2, and Meta-World 3.1.1.

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


## Independent Experiment Packages

The following self-contained experiment packages are grouped by task and backbone:

- [DDPG-HER-PER-Full-2M](experiments/DDPG-HER-PER-Full-2M/)
- [DDPG-HER-B3-Replay-Action-Query-2M](experiments/DDPG-HER-B3-Replay-Action-Query-2M/)
- [DDPG-HER-B3-Shuffled-Policy-Query-2M](experiments/DDPG-HER-B3-Shuffled-Policy-Query-2M/)

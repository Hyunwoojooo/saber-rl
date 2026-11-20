# MetaWorld Drawer-Close-v3 DDPG Experiments

This folder contains an independent DDPG+HER code snapshot for MetaWorld `drawer-close-v3` with sparse reward, `reward_function_version=v2`, 2M steps, and seeds 1-5. It reuses the Drawer-Close TD3 environment/task/HER/SABER code path, but replaces the actor-critic backend with the single-critic DDPG agent used in `metaworld_Drawer_Open_DDPG`.

## Completed Shuffled-Policy Query Seed Extension

Shuffled-Policy Query has completed **seeds 6-10, 2M steps per seed**.
The [Shuffled-Policy Query result artifact](paper_artifacts/drawer_close_ddpg_shuffled_policy_query_seed_extension_2m/README.md)
includes the five new runs, retained seeds 1-5, and the previously published
SABER runs for paired comparison. Reused runs are not extra independent
observations. Results from other control methods are excluded.

| Cohort | Shuffled-Policy Query mean AUC | SABER mean AUC | SABER - Shuffled-Policy Query | Paired 95% CI |
| --- | ---: | ---: | ---: | --- |
| Historical seeds 1-5 | 0.291508 | 0.324422 | +0.032915 | [-0.062739, +0.119070] |
| Extension seeds 6-10 | 0.281206 | 0.304673 | +0.023467 | [-0.007387, +0.056332] |
| Pooled seeds 1-10 | 0.286357 | 0.314548 | +0.028191 | [-0.018556, +0.073090] |

SABER has higher pooled mean AUC, but the paired interval includes zero.
Pooled FQ-AUC and final-success intervals also include zero. Shuffling
breaks sample-query pairing but can change the number of high weights,
so this is not a count-matched assignment comparison. All seeds, including
zero-final-success outcomes, are retained.

## Completed Replay-Action Query Seed Extension

Replay-Action Query has completed **seeds 6-10, 2M steps per seed**.
The [Replay-Action Query result artifact](paper_artifacts/drawer_close_ddpg_replay_action_query_seed_extension_2m/README.md)
contains the five newly published runs, historical seeds 1-5, and the already
published SABER results for paired comparison. Reused SABER runs are not extra
independent observations. Other control results are not included in this artifact.

| Cohort | Replay-Action Query mean AUC | SABER mean AUC | SABER - Replay-Action Query | Paired 95% CI |
| --- | ---: | ---: | ---: | --- |
| Historical seeds 1-5 | 0.275754 | 0.324422 | +0.048668 | [-0.005553, +0.124824] |
| Extension seeds 6-10 | 0.287412 | 0.304673 | +0.017261 | [-0.017211, +0.051734] |
| Pooled seeds 1-10 | 0.281583 | 0.314548 | +0.032965 | [-0.002136, +0.077193] |

The pooled primary AUC interval includes zero. The secondary FQ-AUC difference
is +0.060200 with an unadjusted 95% interval [+0.001750, +0.117300]; this does
not establish general superiority or necessity of current-policy querying.
All seeds, including zero-final-success outcomes, remain included. Intervals
use 10,000 paired bootstrap resamples; original five-seed exact-enumeration
tables remain unchanged. The artifact includes SDs, curves, configs, source
snapshots, provenance and a read-only verifier.

## Completed Primary Seed Extensions

HER, SABER, HGR and PER have all completed **seeds 6-10, 2M steps per seed**.
The [four-method result artifact](paper_artifacts/drawer_close_ddpg_primary_seed_extension_2m/README.md)
adds HGR/PER results and includes the existing HER/SABER runs for comparison.
It contains 40 distinct runs (four methods x ten seeds), not additional copies
counted as independent observations. Non-primary control results are excluded.

| Method | Seeds 6-10 mean AUC | Seeds 1-10 mean AUC | Seeds 1-10 mean FQ-AUC | Seeds 1-10 mean final success |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.307688 | 0.279347 | 0.241300 | 0.380 |
| SABER (Ours) | 0.304673 | 0.314548 | 0.303300 | 0.340 |
| HGR | 0.291910 | 0.320553 | 0.251050 | 0.165 |
| PER | 0.307563 | 0.307751 | 0.335350 | 0.325 |

All three pooled SABER-minus-comparator AUC intervals include zero. See the
artifact for sample SDs, paired CIs, seed-level outcomes, provenance limits,
and a read-only verifier. Earlier published artifacts remain unchanged.

### Earlier HER-SABER Publication

HER and SABER seeds **6-10** have both completed 2M steps each.
The [HER-SABER comparison artifact](paper_artifacts/drawer_close_ddpg_her_saber_seed_extension_2m/README.md)
contains all seed results, historical/new/pooled tables, paired confidence
intervals, evaluation curves, executed source snapshots, and SHA-256 verification.

| Cohort | HER mean AUC | SABER mean AUC | SABER - HER | Paired 95% CI |
| --- | ---: | ---: | ---: | --- |
| Historical seeds 1-5 | 0.251005 | 0.324422 | +0.073417 | [+0.016181, +0.143191] |
| Extension seeds 6-10 | 0.307688 | 0.304673 | -0.003015 | [-0.055176, +0.053442] |
| Pooled seeds 1-10 | 0.279347 | 0.314548 | +0.035201 | [-0.010289, +0.086697] |

The pooled sample mean favors SABER, but the 10-seed interval includes zero;
superiority and equivalence are not established. All seeds remain included.
The combined artifact reuses the previously published HER runs, not additional
independent executions. Historical execution-environment provenance limitations
are documented in the artifact. Other methods' performance is not included.

### Earlier HER-Only Publication

HER seeds **6-10** completed 2M steps each on November 14, 2026 (UTC).
The [completed HER artifact](paper_artifacts/drawer_close_ddpg_her_seed_extension_2m/README.md)
contains seed-level results, historical/new/pooled cohort tables, evaluation
curves, executed source, and a SHA-256 verifier. Existing seeds 1-5 are reused
as references, not rerun or counted as new executions.

For the five new seeds, AUC is **0.307688 +/- 0.051379**, FQ-AUC is
**0.232400 +/- 0.060551**, and final success is **0.600000 +/- 0.413824**
(mean +/- sample SD). This earlier artifact remains unchanged and contains HER
only. The completed HER-SABER comparison is linked above.
The original five-seed tables below remain unchanged.

## Experiment Set

| Experiment | Script | Method | Weight Scope |
|---|---|---|---|
| DDPG+HER B0 baseline | `DDPG-HER-B0-2M.sh` | `b0_ddpg_her` | `none` |
| DDPG+HER+B3 Critic-Only | `DDPG-HER-B3-Critic-Only-2M.sh` | `b3_support_runtime_opt` | `critic_only` |
| DDPG+HER+B3 Actor-Only | `DDPG-HER-B3-Actor-Only-2M.sh` | `b3_support_runtime_opt` | `actor_only` |
| DDPG+HER+B3 Actor+Critic | `DDPG-HER-B3-Actor-Critic-2M.sh` | `b3_support_runtime_opt` | `actor_critic` |
| DDPG+HER+B3 Random-Fixed-Count | `DDPG-HER-B3-Random-Fixed-Count-2M.sh` | `b3_fully_random_fixed_count` | `critic_only` |
| DDPG+HER+B3 Random-Support | `DDPG-HER-B3-Random-Support-2M.sh` | `b3_support_random_runtime_opt` | `critic_only` |
| DDPG+HER+B3 Shuffled-Weight | `DDPG-HER-B3-Shuffled-Weight-2M.sh` | `b3_support_shuffled_runtime_opt` | `critic_only` |
| DDPG+HER+PER-Style | `DDPG-HER-PER-Style-2M.sh` | `per_style_loss_weighting` | `critic_only` |

## Shared Settings

- MetaWorld env: `drawer-close-v3`
- Reward type: `sparse`
- MetaWorld reward function version: `v2`
- Horizon: 200
- Seeds: 1-5
- Steps: 2,000,000
- HER ratio: 0.8
- Batch size: 256
- Replay capacity: 300000
- CUDA parallelism: 5 processes on `CUDA_DEVICE=0` by default

## B3 Settings

- `M_B=16`
- `M_Q_min=4`
- `N_local_min=4`
- `N_gap_min=128`
- `local_search_max=512`
- `lambda_support=1.0`
- `weighting_mode=batch_norm`
- Numba Drawer support scanner enabled

## Run

From repository root:

```bash
bash metaworld_Drawer_Close_DDPG/DDPG-HER-B0-2M.sh
bash metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Critic-Only-2M.sh
bash metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Actor-Only-2M.sh
bash metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Actor-Critic-2M.sh
bash metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Random-Fixed-Count-2M.sh
bash metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Random-Support-2M.sh
bash metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Shuffled-Weight-2M.sh
bash metaworld_Drawer_Close_DDPG/DDPG-HER-PER-Style-2M.sh
```

Run the full sequence:

```bash
bash metaworld_Drawer_Close_DDPG/run_all_ddpg_drawer_close_2m_5parallel.sh
```

Run only the two actor-scope ablations in sequence:

```bash
bash metaworld_Drawer_Close_DDPG/run_ddpg_drawer_close_actor_ablation_2m_5parallel.sh
```

For a short smoke check:

```bash
STEPS=1000 STEP_TAG=smoke CUDA_DEVICE=0 bash metaworld_Drawer_Close_DDPG/DDPG-HER-B0-2M.sh
```

Outputs are written to `metaworld_Drawer_Close_DDPG/results_2m/`, logs to `metaworld_Drawer_Close_DDPG/logs_2m/`, and orchestrator logs to `metaworld_Drawer_Close_DDPG/orchestrator_logs/`.

## Actor-Scope Ablation Settings

The actor-scope launchers use the same DrawerClose environment, DDPG backbone, HER path, support query, normalized SABER weights, seeds, and training budget as `DDPG-HER-B3-Critic-Only-2M.sh`. The only intentional difference is where the same per-sample weight is applied:

- `DDPG-HER-B3-Actor-Only-2M.sh`: `weight_scope=actor_only`; critic weights remain one.
- `DDPG-HER-B3-Actor-Critic-2M.sh`: `weight_scope=actor_critic`; the same normalized weights are applied to actor and critic losses.

These are scope ablations. They do not change replay sampling, HER relabeling, Bellman targets, support lookup, network architecture, optimizer settings, or UTD ratio.

## Results

### Completed Result Summary

| Experiment | Status | Final Success Seeds | Final Success | Best Success | AUC Success | Mean Hit90 | Speed |
|---|---|---:|---:|---:|---:|---:|---:|
| DDPG-HER-B0-2M | Completed | 0/5 | 0.16 | 1.00 | 0.251 | 74k | 41.22 steps/s |
| DDPG-HER-B3-Critic-Only-2M | Completed | 0/5 | 0.28 | 1.00 | 0.324 | 30k | 23.96 steps/s |
| DDPG-HER-B3-Random-Fixed-Count-2M | Completed | 1/5 | 0.48 | 1.00 | 0.319 | 100k | 41.09 steps/s |
| DDPG-HER-B3-Random-Support-2M | Completed | 1/5 | 0.24 | 1.00 | 0.277 | 68k | 23.84 steps/s |
| DDPG-HER-B3-Shuffled-Weight-2M | Completed | 0/5 | 0.07 | 1.00 | 0.328 | 108k | 24.00 steps/s |
| DDPG-HER-PER-Style-2M | Completed | 0/5 | 0.40 | 1.00 | 0.313 | 108k | 38.94 steps/s |
| DDPG-HER-PER-Full-2M | Completed | 1/5 | 0.46 | 1.00 | 0.308 | 86k | 33.95 steps/s |

### DDPG-HER-B0-2M

Completed on 2026-07-03 KST with 5 seeds, 2M steps per seed, sparse DrawerClose-v3, and 20 evaluation episodes every 10k steps.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 0.30 | 1.00 | 0.282 | 90k |
| 2 | 0.10 | 1.00 | 0.269 | 150k |
| 3 | 0.40 | 1.00 | 0.204 | 40k |
| 4 | 0.00 | 1.00 | 0.249 | 70k |
| 5 | 0.00 | 1.00 | 0.252 | 20k |
| Mean | 0.16 | 1.00 | 0.251 | 74k |

DDPG-HER B0 reached 90% success at least once in every seed, but none of the seeds maintained at least 90% success in the final evaluation. This confirms that DrawerClose-v3 sparse remains unstable for the plain DDPG+HER baseline even though transient successful policies appear early in training.

### DDPG-HER-B3-Critic-Only-2M

Completed on 2026-07-04 KST with 5 seeds, 2M steps per seed, sparse DrawerClose-v3, and SABER/B3 support weighting applied to critic loss only.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 0.65 | 1.00 | 0.288 | 10k |
| 2 | 0.20 | 1.00 | 0.337 | 60k |
| 3 | 0.15 | 1.00 | 0.284 | 50k |
| 4 | 0.00 | 1.00 | 0.461 | 20k |
| 5 | 0.40 | 1.00 | 0.252 | 10k |
| Mean | 0.28 | 1.00 | 0.324 | 30k |

B3-Critic-Only improved over DDPG-HER B0 on final success and AUC, but no seed maintained at least 90% success in the final evaluation. The early hit90 values show that support-weighted critic learning finds successful behavior quickly, while final policy stability remains weak on sparse DrawerClose-v3.

### DDPG-HER-B3-Random-Fixed-Count-2M

Completed on 2026-07-05 KST with 5 seeds, 2M steps per seed, sparse DrawerClose-v3, and critic-only fully random fixed-count weighting. This control does not perform SABER support lookup; it assigns high critic weights to a fixed random subset of 122 samples per batch.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 0.30 | 1.00 | 0.295 | 70k |
| 2 | 0.35 | 1.00 | 0.349 | 170k |
| 3 | 1.00 | 1.00 | 0.330 | 190k |
| 4 | 0.45 | 1.00 | 0.278 | 40k |
| 5 | 0.30 | 1.00 | 0.344 | 30k |
| Mean | 0.48 | 1.00 | 0.319 | 100k |

Random-Fixed-Count produced a strong final snapshot, but its mean AUC remained slightly below B3-Critic-Only. This indicates that generic critic-side stochastic reweighting can be a strong control on sparse DrawerClose-v3, while support-conditioned critic weighting remains at least competitive on the primary AUC metric.

### DDPG-HER-B3-Random-Support-2M

Completed on 2026-07-05 KST with 5 seeds, 2M steps per seed, sparse DrawerClose-v3, and critic-only density-matched random support weighting. This control computes the SABER support-positive count, but assigns the high weights to random batch samples rather than the support-positive samples themselves.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 0.15 | 1.00 | 0.303 | 40k |
| 2 | 1.00 | 1.00 | 0.337 | 150k |
| 3 | 0.05 | 1.00 | 0.213 | 100k |
| 4 | 0.00 | 1.00 | 0.289 | 30k |
| 5 | 0.00 | 1.00 | 0.241 | 20k |
| Mean | 0.24 | 1.00 | 0.277 | 68k |

Random-Support improved over B0 on final success and AUC, but did not match B3-Critic-Only. This suggests that support-derived density alone is not sufficient on DrawerClose DDPG; assigning the critic weights to the aligned support-positive samples still contributes to performance.

### DDPG-HER-B3-Shuffled-Weight-2M

Completed on 2026-07-06 KST with 5 seeds, 2M steps per seed, sparse DrawerClose-v3, and critic-only shuffled SABER weighting. This control computes the SABER support-derived raw weight vector, preserves its weight distribution, and shuffles the assignment across batch samples before applying the weights to the critic loss.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 0.35 | 1.00 | 0.352 | 130k |
| 2 | 0.00 | 1.00 | 0.227 | 240k |
| 3 | 0.00 | 1.00 | 0.389 | 100k |
| 4 | 0.00 | 1.00 | 0.308 | 60k |
| 5 | 0.00 | 1.00 | 0.365 | 10k |
| Mean | 0.07 | 1.00 | 0.328 | 108k |

Shuffled-Weight produced a mean AUC comparable to B3-Critic-Only, but its final success and late-training stability were much weaker. This indicates that support-derived reweighting distribution alone can create transient learning gains, while preserving the support-to-sample assignment is still important for stable final policies on this DrawerClose DDPG setting.

### DDPG-HER-PER-Style-2M

Completed on 2026-07-07 KST with 5 seeds, 2M steps per seed, sparse DrawerClose-v3, and critic-only PER-style TD-error loss weighting.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 0.60 | 1.00 | 0.448 | 50k |
| 2 | 0.55 | 1.00 | 0.321 | 130k |
| 3 | 0.25 | 1.00 | 0.241 | 130k |
| 4 | 0.35 | 1.00 | 0.358 | 130k |
| 5 | 0.25 | 1.00 | 0.199 | 100k |
| Mean | 0.40 | 1.00 | 0.313 | 108k |

PER-Style produced the strongest mean final success among the completed DrawerClose DDPG controls so far, but its mean AUC remained below B3-Critic-Only and Shuffled-Weight. This makes it a strong final-snapshot baseline/control while preserving the need to report AUC and late stability separately.

### DDPG-HER-PER-Full-2M

Completed on 2026-07-23 KST with 5 seeds, 2M steps per seed, sparse
DrawerClose-v3, and full prioritized experience replay. Replay slots are
sampled by priority before future-goal HER relabeling, importance-sampling
weights are applied to the critic loss, and priorities are updated from the
same single-critic absolute TD error used by the optimizer step.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% |
|---:|---:|---:|---:|---:|
| 1 | 0.55 | 1.00 | 0.289 | 100k |
| 2 | 0.00 | 1.00 | 0.245 | 180k |
| 3 | 1.00 | 1.00 | 0.298 | 40k |
| 4 | 0.35 | 1.00 | 0.331 | 50k |
| 5 | 0.40 | 1.00 | 0.377 | 60k |
| Mean | 0.46 | 1.00 | 0.308 | 86k |

Every seed reached 90% success transiently, but only seed 3 retained at least
90% success at the final evaluation. Full PER improved over DDPG+HER B0 in
both AUC and final success. Relative to B3-Critic-Only, it produced higher
final success (0.46 versus 0.28) but lower AUC (0.308 versus 0.324), showing
that the final snapshot and whole-training efficiency favor different methods
in this cohort.


## Independent Experiment Packages

The following self-contained experiment packages are grouped by task and backbone:

- [DDPG-HER-PER-Full-2M](experiments/DDPG-HER-PER-Full-2M/)
- [DDPG-HER-B3-Replay-Action-Query-2M](experiments/DDPG-HER-B3-Replay-Action-Query-2M/)
- [DDPG-HER-B3-Shuffled-Policy-Query-2M](experiments/DDPG-HER-B3-Shuffled-Policy-Query-2M/)

# DrawerOpen DDPG HGR Pair-Priority + IS 2M Artifact

Paper label: **HGR (pair priority + IS)**

> This is a full two-stage HGR transfer under the repository's matched
> DrawerOpen DDPG+HER backbone: episode ranking, future transition-goal-pair
> ranking, and importance correction are all enabled. It is not an exact
> reproduction of the original HGR paper's task or training configuration.

## Scope

This artifact contains the official DrawerOpen DDPG Full HGR result. It covers
seeds 1-5, 2,000,000 environment steps, deterministic evaluation every 10,000
steps with 20 episodes, and the exact recorded evaluation grid. The matched
DDPG+HER and critic-only SABER rows are recomputed from their same-seed curves.

## Aggregate Results

Values after `+/-` are sample standard deviations over five seeds.

| Method | Final success | Best success | AUC | Last-quarter AUC | Hit90 | Hit90 median | Steps/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| HGR (pair priority + IS) | 0.900 +/- 0.224 | 1.000 +/- 0.000 | 0.688 +/- 0.142 | 0.983 +/- 0.025 | 5/5 | 530k | 13.24 |
| DDPG+HER | 0.390 +/- 0.534 | 0.400 +/- 0.548 | 0.207 +/- 0.287 | 0.393 +/- 0.538 | 2/5 | 960k | 41.05 |
| SABER (critic-only DDPG+HER) | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.754 +/- 0.044 | 0.987 +/- 0.019 | 5/5 | 500k | 23.33 |

## Paired Comparisons

Each delta is HGR minus the same-seed reference. The 95% intervals are paired
nonparametric percentile-bootstrap intervals. For five pairs, all 3,125
resamples are enumerated.

| Comparison | Metric | Pairs | Mean delta | 95% CI |
|---|---|---:|---:|---:|
| HGR-minus-HER | final_success | 5 | 0.510 | [-0.090, 1.000] |
| HGR-minus-HER | success_auc | 5 | 0.481 | [0.151, 0.738] |
| HGR-minus-HER | last_quarter_auc | 5 | 0.591 | [0.171, 0.995] |
| HGR-minus-HER | hit90_reached | 5 | 0.600 | [0.200, 1.000] |
| HGR-minus-SABER | final_success | 5 | -0.100 | [-0.300, 0.000] |
| HGR-minus-SABER | success_auc | 5 | -0.065 | [-0.219, 0.047] |
| HGR-minus-SABER | last_quarter_auc | 5 | -0.004 | [-0.036, 0.021] |
| HGR-minus-SABER | hit90_reached | 5 | 0.000 | [0.000, 0.000] |

The full HGR result is substantially stronger than matched HER. SABER has the
higher mean AUC, but the paired HGR-minus-SABER AUC interval includes zero;
these five seeds do not establish superiority of either method. HGR changes
HER sampling and applies sampling-probability correction, whereas SABER keeps
the sampled HER minibatch fixed and changes critic-loss allocation. The result
therefore supports a matched complete-method comparison, not a claim that the
two methods isolate the same mechanism.

## HGR Protocol

- completed-episode pair table over all valid transition/future-goal pairs
- episode and pair priorities with `alpha_episode=alpha_goal=0.6`
- episode and pair IS exponents annealed from 0.4 to 1.0
- HGR relabel fraction 0.8; original-goal uniform branch 0.2
- critic-only IS correction for HGR-relabeled samples; actor weights remain 1
- pre-update absolute TD error updates sampled pair priorities
- no priority clipping, no uniform mixture, UTD 1

## Final HGR Diagnostics

These values are taken from each seed's final 2M evaluation row.

| Metric | Mean | Sample std |
|---|---:|---:|
| `hgr/episode_priority_mean` | 2.913776 | 0.463357 |
| `hgr/episode_priority_std` | 8.347801 | 1.133529 |
| `hgr/episode_priority_max` | 56.477563 | 8.110434 |
| `hgr/pair_table_count` | 1500.000000 | 0.000000 |
| `hgr/pair_priority_count` | 30150000.000000 | 0.000000 |
| `hgr/pair_priority_memory_mb` | 115.013123 | 0.000000 |
| `hgr/goal_priority_entropy_mean` | 9.486030 | 0.026616 |
| `hgr/goal_selected_rank_mean` | 3321.469820 | 426.994305 |
| `hgr/candidate_td_error_mean` | 29.436353 | 5.419489 |
| `hgr/candidate_td_error_max` | 56.477563 | 8.110434 |
| `hgr/relabel_fraction` | 0.788281 | 0.027120 |
| `hgr/is_weight_mean` | 0.102700 | 0.049253 |
| `hgr/is_weight_min` | 0.006617 | 0.004998 |
| `hgr/beta_episode` | 1.000000 | 0.000000 |
| `hgr/beta_goal` | 1.000000 | 0.000000 |
| `hgr/beta_progress` | 0.999999 | 0.000000 |
| `hgr/max_priority` | 56.477564 | 8.110434 |
| `hgr/priority_clip_rate` | 0.000000 | 0.000000 |
| `runtime/hgr_sampling_sec_per_update` | 0.065144 | 0.002375 |

## Metric Definitions

- AUC is trapezoidal normalized success area on exactly `10k, ..., 2M`, divided
  by `2,000,000 - 10,000`; no smoothing, interpolation, or carry-forward is used.
- Last-quarter AUC follows the training summary convention: the final 50
  evaluation points, exactly `1.51M, ..., 2M`, divided by 490,000.
- Hit90 is the first recorded evaluation with success at least 0.90; the median
  is over reached seeds and always accompanies coverage.
- Summary-native final, best, AUC, and Hit90 values are checked against the
  exact-grid recomputation before outputs are written.

## Included Files

- `evaluation_curves_10k.csv`: all 15 exact-grid curves
- `per_seed_results.csv`, `aggregate_results.csv`, `paired_comparisons.csv`
- `runs/`: sanitized HGR config, environment snapshot, summary, and selected
  10k metrics for each seed
- `code/build_artifact.py` and `code/verify_artifact.py`
- `artifact_manifest.json`: SHA256 and size of every payload file except itself

Checkpoints, raw `metrics.csv`, JSONL, and run logs are excluded. Host-specific
absolute output paths in copied HGR config/summary files are replaced by their
repository-relative source paths.

## Rebuild

From the repository root, while the source runs are locally available:

```bash
python3 metaworld_Drawer_Open_DDPG_HGR/paper_artifacts/drawer_open_ddpg_hgr_pair_priority_is_2m/code/build_artifact.py
python3 metaworld_Drawer_Open_DDPG_HGR/paper_artifacts/drawer_open_ddpg_hgr_pair_priority_is_2m/code/verify_artifact.py
```

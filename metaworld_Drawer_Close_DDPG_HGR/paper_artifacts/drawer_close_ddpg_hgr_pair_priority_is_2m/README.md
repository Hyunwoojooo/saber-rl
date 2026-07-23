# DrawerClose DDPG HGR Pair-Priority + IS 2M Artifact

Paper label: **HGR (pair priority + IS)**

> This is a full two-stage HGR transfer under the repository's matched
> DrawerClose DDPG+HER backbone: episode ranking, future transition-goal-pair
> ranking, and importance correction are all enabled. It is not an exact
> reproduction of the original HGR paper's task or training configuration.

## Scope

This artifact contains the official DrawerClose DDPG Full HGR result. It covers
seeds 1-5, 2,000,000 environment steps, deterministic evaluation every 10,000
steps with 20 episodes, and the exact recorded evaluation grid. The matched
DDPG+HER and critic-only SABER rows are recomputed from their same-seed curves.

## Aggregate Results

Values after `+/-` are sample standard deviations over five seeds.

| Method | Final success | Best success | AUC | Last-quarter AUC | Hit90 | Hit90 median | Steps/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| HGR (pair priority + IS) | 0.080 +/- 0.091 | 1.000 +/- 0.000 | 0.349 +/- 0.184 | 0.224 +/- 0.056 | 5/5 | 100k | 13.22 |
| DDPG+HER | 0.160 +/- 0.182 | 1.000 +/- 0.000 | 0.251 +/- 0.030 | 0.252 +/- 0.094 | 5/5 | 70k | 41.22 |
| SABER (critic-only DDPG+HER) | 0.280 +/- 0.251 | 1.000 +/- 0.000 | 0.324 +/- 0.082 | 0.281 +/- 0.102 | 5/5 | 20k | 23.96 |

## Paired Comparisons

Each delta is HGR minus the same-seed reference. The 95% intervals are paired
nonparametric percentile-bootstrap intervals. For five pairs, all 3,125
resamples are enumerated.

| Comparison | Metric | Pairs | Mean delta | 95% CI |
|---|---|---:|---:|---:|
| HGR-minus-HER | final_success | 5 | -0.080 | [-0.240, 0.080] |
| HGR-minus-HER | success_auc | 5 | 0.098 | [0.009, 0.248] |
| HGR-minus-HER | last_quarter_auc | 5 | -0.028 | [-0.105, 0.086] |
| HGR-minus-HER | hit90_reached | 5 | 0.000 | [0.000, 0.000] |
| HGR-minus-SABER | final_success | 5 | -0.200 | [-0.430, -0.010] |
| HGR-minus-SABER | success_auc | 5 | 0.025 | [-0.132, 0.225] |
| HGR-minus-SABER | last_quarter_auc | 5 | -0.057 | [-0.154, 0.041] |
| HGR-minus-SABER | hit90_reached | 5 | 0.000 | [0.000, 0.000] |

Full HGR has the highest mean AUC of the three methods. The paired HGR-minus-SABER AUC interval includes zero, and only two of five seed
differences are positive. All HGR seeds reach Hit90 early, but HGR ends with
lower mean final success, indicating substantial late instability. These five
seeds therefore do not establish superiority of either method. HGR changes
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
| `hgr/episode_priority_mean` | 0.768638 | 0.095568 |
| `hgr/episode_priority_std` | 2.793910 | 0.394378 |
| `hgr/episode_priority_max` | 22.189757 | 4.097375 |
| `hgr/pair_table_count` | 1500.000000 | 0.000000 |
| `hgr/pair_priority_count` | 30150000.000000 | 0.000000 |
| `hgr/pair_priority_memory_mb` | 115.013123 | 0.000000 |
| `hgr/goal_priority_entropy_mean` | 9.445284 | 0.026165 |
| `hgr/goal_selected_rank_mean` | 3359.925079 | 191.778767 |
| `hgr/candidate_td_error_mean` | 11.213221 | 1.871417 |
| `hgr/candidate_td_error_max` | 22.189757 | 4.097375 |
| `hgr/relabel_fraction` | 0.788281 | 0.027120 |
| `hgr/is_weight_mean` | 0.088126 | 0.042095 |
| `hgr/is_weight_min` | 0.003410 | 0.001550 |
| `hgr/beta_episode` | 1.000000 | 0.000000 |
| `hgr/beta_goal` | 1.000000 | 0.000000 |
| `hgr/beta_progress` | 0.999999 | 0.000000 |
| `hgr/max_priority` | 22.189756 | 4.097375 |
| `hgr/priority_clip_rate` | 0.000000 | 0.000000 |
| `runtime/hgr_sampling_sec_per_update` | 0.064953 | 0.002398 |

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
python3 metaworld_Drawer_Close_DDPG_HGR/paper_artifacts/drawer_close_ddpg_hgr_pair_priority_is_2m/code/build_artifact.py
python3 metaworld_Drawer_Close_DDPG_HGR/paper_artifacts/drawer_close_ddpg_hgr_pair_priority_is_2m/code/verify_artifact.py
```

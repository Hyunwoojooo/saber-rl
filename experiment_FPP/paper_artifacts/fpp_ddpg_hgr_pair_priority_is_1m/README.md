# FetchPickAndPlace HGR Pair-Priority + IS 1M Artifact

Paper label: **HGR (matched DDPG+HER)**

> This is an algorithmically faithful HGR pair-priority and importance-correction
> implementation under the paper’s matched DDPG+HER backbone. It is not an exact
> reproduction of the original HGR paper’s environment and training configuration.

## Scope

This artifact contains FetchPickAndPlace-v4 sparse-reward results for seeds 1-5
at 1,000,000 environment steps. HGR is compared with the matched DDPG+HER
baseline and the main critic-only SABER DDPG+HER run set. Backbone settings,
seed streams, evaluation frequency, and evaluation episode count are validated
before any output is written.

## Aggregate Results

All values after `+/-` are sample standard deviations across five seeds.

| Method | Final success | Best success | Success AUC | Best-final drop | Hit90 | Hit90 median among hits |
|---|---:|---:|---:|---:|---:|---:|
| HGR (matched DDPG+HER) | 0.540 +/- 0.365 | 0.810 +/- 0.188 | 0.338 +/- 0.149 | 0.270 +/- 0.186 | 2/5 | 930k |
| DDPG+HER | 0.940 +/- 0.042 | 1.000 +/- 0.000 | 0.539 +/- 0.125 | 0.060 +/- 0.042 | 5/5 | 600k |
| SABER (matched DDPG+HER) | 0.960 +/- 0.042 | 1.000 +/- 0.000 | 0.609 +/- 0.038 | 0.040 +/- 0.042 | 5/5 | 540k |

## Paired Comparisons

Each delta is HGR minus the same-seed reference. Confidence intervals are
two-sided 95% paired nonparametric percentile bootstrap intervals. With five
seeds, all `5^5 = 3,125` resamples are enumerated, so there is no Monte Carlo
seed or sampling error. The full metric set is in `paired_vs_her_saber.csv`.

| Comparison | Metric | Pairs | Mean delta | 95% bootstrap CI |
|---|---|---:|---:|---:|
| HGR-minus-HER | final_success | 5 | -0.400 | [-0.710, -0.110] |
| HGR-minus-HER | success_auc | 5 | -0.201 | [-0.357, -0.031] |
| HGR-minus-HER | hit90_reached | 5 | -0.600 | [-1.000, -0.200] |
| HGR-minus-SABER | final_success | 5 | -0.420 | [-0.730, -0.150] |
| HGR-minus-SABER | success_auc | 5 | -0.271 | [-0.405, -0.197] |
| HGR-minus-SABER | hit90_reached | 5 | -0.600 | [-1.000, -0.200] |

## Metric Definitions

- Final success is `eval/env_success_mean` at exactly 1,000,000 steps.
- Best success is the maximum over exactly `{10k, 20k, ..., 1M}`.
- Success AUC is normalized trapezoidal area on those 100 observed points:
  `trapz(success, step) / (1,000,000 - 10,000)`. Missing points are rejected;
  no values are interpolated or carried forward.
- Best-final drop is `best_success - final_success`.
- Hit90 is the first observed grid step with success at least 0.90. The median
  uses only seeds that hit; the count always reports the denominator.
- Summary-native final, best, AUC, and Hit90 values are audited against the
  recomputed exact-grid values.

## Final HGR Diagnostics

The table uses each seed's 1M evaluation row. Full 10k-grid traces, including
all logged `hgr/*` fields, are stored in each `hgr_metrics_selected.csv`.

| Metric | Mean | Sample std |
|---|---:|---:|
| `hgr/relabel_fraction` | 0.774219 | 0.021467 |
| `hgr/fallback_fraction` | 0.000000 | 0.000000 |
| `hgr/beta_episode` | 0.999999 | 0.000000 |
| `hgr/beta_goal` | 0.999999 | 0.000000 |
| `hgr/is_weight_mean` | 0.129677 | 0.044543 |
| `hgr/is_weight_min` | 0.002848 | 0.001142 |
| `hgr/episode_priority_mean` | 45.145904 | 98.996080 |
| `hgr/pair_priority_count` | 7650000.000000 | 0.000000 |
| `hgr/priority_safety_replacement_rate` | 0.000000 | 0.000000 |
| `runtime/hgr_sampling_sec_per_update` | 0.024291 | 0.008421 |

## Included Files

Each HGR seed directory contains only:

- `summary.json`
- `config.json`
- `env_snapshot.json`
- `metrics_selected_10k.csv` (200 source rows: train and evaluation rows at each grid step)
- `hgr_metrics_selected.csv` (100 evaluation rows)

Root analysis outputs are `aggregate_results.csv`, `per_seed_results.csv`, and
`paired_vs_her_saber.csv`. `code/build_artifact.py` rebuilds the artifact and
`code/verify_artifact.py` performs an independent integrity check.

Checkpoints, raw `metrics.csv`, raw JSONL, and logs are intentionally excluded.
`artifact_manifest.json` records SHA256 and byte size for every payload file.
The manifest excludes itself because a file cannot contain its own final hash;
this exclusion is explicit in the manifest schema.

## Source Runs

- HGR: `experiment_FPP/runs/ddpg_hgr_tdpriority_original_pair_is_1m/`
- DDPG+HER: `experiment_FPP/runs/B0/`
- DDPG+HER+SABER: `experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512/`
- HGR implementation source commit: `79d3347b0bff0890f6d65c28a04c5c0ca515c846`

## Rebuild

From the repository root, with the three run roots available:

```bash
python3 experiment_FPP/paper_artifacts/fpp_ddpg_hgr_pair_priority_is_1m/code/build_artifact.py
python3 experiment_FPP/paper_artifacts/fpp_ddpg_hgr_pair_priority_is_1m/code/verify_artifact.py
```

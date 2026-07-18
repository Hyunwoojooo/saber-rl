# FetchPickAndPlace TD3+HER+HGR Pair-Priority + IS 1M Artifact

Paper label: **HGR (matched TD3+HER)**

> This is an algorithmically faithful HGR pair-priority and importance-correction
> implementation transferred to a matched TD3+HER backbone. It is not an exact
> reproduction of the original HGR paper’s environment and training configuration.

## Scope

This artifact contains FetchPickAndPlace-v4 sparse-reward results for seeds 1-5
at 1,000,000 environment steps. HGR is compared with the matched TD3+HER
baseline and the main critic-only SABER TD3+HER run set. Backbone settings,
seed streams, evaluation frequency, and evaluation episode count are validated
before any output is written.

## Aggregate Results

All values after `+/-` are sample standard deviations across five seeds.

| Method | Final success | Best success | Success AUC | Best-final drop | Hit90 | Hit90 median among hits |
|---|---:|---:|---:|---:|---:|---:|
| HGR (matched TD3+HER) | 0.610 +/- 0.198 | 0.830 +/- 0.045 | 0.367 +/- 0.034 | 0.220 +/- 0.208 | 1/5 | 630k |
| TD3+HER | 0.840 +/- 0.188 | 0.880 +/- 0.144 | 0.372 +/- 0.032 | 0.040 +/- 0.065 | 3/5 | 980k |
| SABER (matched TD3+HER) | 0.970 +/- 0.045 | 0.990 +/- 0.022 | 0.442 +/- 0.035 | 0.020 +/- 0.027 | 5/5 | 840k |

## Paired Comparisons

Each delta is HGR minus the same-seed reference. Confidence intervals are
two-sided 95% paired nonparametric percentile bootstrap intervals. With five
seeds, all `5^5 = 3,125` resamples are enumerated, so there is no Monte Carlo
seed or sampling error. The full metric set is in `paired_vs_her_saber.csv`.

| Comparison | Metric | Pairs | Mean delta | 95% bootstrap CI |
|---|---|---:|---:|---:|
| HGR-minus-HER | final_success | 5 | -0.230 | [-0.380, -0.110] |
| HGR-minus-HER | success_auc | 5 | -0.004 | [-0.045, 0.023] |
| HGR-minus-HER | hit90_reached | 5 | -0.400 | [-0.800, 0.000] |
| HGR-minus-SABER | final_success | 5 | -0.360 | [-0.510, -0.200] |
| HGR-minus-SABER | success_auc | 5 | -0.075 | [-0.085, -0.065] |
| HGR-minus-SABER | hit90_reached | 5 | -0.800 | [-1.000, -0.400] |

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
| `hgr/is_weight_mean` | 0.089068 | 0.040495 |
| `hgr/is_weight_min` | 0.001077 | 0.000387 |
| `hgr/episode_priority_mean` | 0.454175 | 0.068358 |
| `hgr/pair_priority_count` | 7650000.000000 | 0.000000 |
| `hgr/priority_safety_replacement_rate` | 0.000000 | 0.000000 |
| `runtime/hgr_sampling_sec_per_update` | 0.024806 | 0.007181 |

## Included Files

Each HGR seed directory contains only:

- `summary.json`
- `config.json`
- `env_snapshot.json`
- `metrics_selected_10k.csv` (200 source rows: train and evaluation rows at each grid step)
- `hgr_metrics_selected.csv` (100 evaluation rows)

Root analysis outputs are `aggregate_results.csv`, `per_seed_results.csv`, and
`paired_vs_her_saber.csv`. `completion_validation.json` records the strict 1M
completion gate, `input_manifest.json` hashes every input, and
`provenance/frozen_source.sha256` preserves the launch-time source manifest.
`code/build_artifact.py` rebuilds the artifact and `code/verify_artifact.py`
performs an independent integrity check.

Checkpoints, raw `metrics.csv`, raw JSONL, and logs are intentionally excluded.
`artifact_manifest.json` records SHA256 and byte size for every payload file.
The manifest excludes itself because a file cannot contain its own final hash;
this exclusion is explicit in the manifest schema.

## Source Runs

- HGR: `experiment_FPP/runs/td3_hgr_tdpriority_1m/`
- TD3+HER: `experiment_FPP/runs/td3_her_baseline_ls512/`
- TD3+HER+SABER: `experiment_FPP/runs/td3_saber_critic_only_v4_ls512/`
- Exact launch-time HGR source commit: `e36059dda646cfbaf60896df07bfc1e6d722f71e`
- The result commit may add a repository-root portability fix to `run.py`;
  the algorithm, trainer, and frozen configuration remain unchanged.

## Rebuild

From the repository root, with the three run roots available:

```bash
python3 experiment_FPP/paper_artifacts/fpp_td3_hgr_pair_priority_is_1m/code/build_artifact.py
python3 experiment_FPP/paper_artifacts/fpp_td3_hgr_pair_priority_is_1m/code/verify_artifact.py
```

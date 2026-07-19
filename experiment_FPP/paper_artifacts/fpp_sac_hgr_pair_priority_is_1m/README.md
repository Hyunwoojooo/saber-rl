# FetchPickAndPlace SAC+HER+HGR Pair-Priority + IS 1M Artifact

Paper label: **HGR (matched SAC+HER)**

> This is an algorithmically faithful HGR pair-priority and importance-correction
> implementation transferred to a matched SAC+HER backbone. It is not an exact
> reproduction of the original HGR paper’s environment and training configuration.

## Scope

This artifact contains FetchPickAndPlace-v4 sparse-reward results for seeds 1-5
at 1,000,000 environment steps. HGR is compared with the matched SAC+HER
baseline and the main critic-only SABER SAC+HER run set. Backbone settings,
seed streams, configured evaluation frequency, and evaluation episode count are
validated before any output is written.

## Aggregate Results

All values after `+/-` are sample standard deviations across five seeds
(`statistics.stdev`, denominator `n-1`). HGR rows use the complete observed
25k grid (40 evaluations). The two reference artifacts preserve evaluations on
the common 50k grid (20 evaluations), so their aggregate rows use that grid.

| Method | Final success | Best success | Success AUC | Best-final drop | Hit90 | Hit90 median among hits |
|---|---:|---:|---:|---:|---:|---:|
| HGR (matched SAC+HER) | 0.760 +/- 0.251 | 0.880 +/- 0.110 | 0.475 +/- 0.071 | 0.120 +/- 0.164 | 2/5 | 800k |
| SAC+HER | 0.790 +/- 0.288 | 0.900 +/- 0.141 | 0.471 +/- 0.079 | 0.110 +/- 0.160 | 3/5 | 800k |
| SABER (matched SAC+HER) | 0.990 +/- 0.022 | 1.000 +/- 0.000 | 0.583 +/- 0.030 | 0.010 +/- 0.022 | 5/5 | 750k |

## Paired Comparisons

Each delta is HGR minus the same-seed reference. Confidence intervals are
two-sided 95% paired nonparametric percentile bootstrap intervals. With five
seeds, all `5^5 = 3,125` resamples are enumerated, so there is no Monte Carlo
seed or sampling error. Every paired metric is recomputed on the exact common
`{50k, 100k, ..., 1M}` grid (20 points); HGR's intervening 25k points are not
used in a paired delta. The full metric set is in `paired_vs_her_saber.csv`.

| Comparison | Metric | Pairs | Mean delta | 95% bootstrap CI |
|---|---|---:|---:|---:|
| HGR-minus-HER | final_success | 5 | -0.030 | [-0.300, 0.270] |
| HGR-minus-HER | success_auc | 5 | 0.013 | [-0.072, 0.098] |
| HGR-minus-HER | hit90_reached | 5 | -0.200 | [-0.800, 0.400] |
| HGR-minus-SABER | final_success | 5 | -0.230 | [-0.430, -0.030] |
| HGR-minus-SABER | success_auc | 5 | -0.100 | [-0.159, -0.029] |
| HGR-minus-SABER | hit90_reached | 5 | -0.600 | [-1.000, -0.200] |

## Metric Definitions

- Final success is `eval/env_success_mean` at exactly 1,000,000 steps.
- HGR primary best success is the maximum over exactly
  `{25k, 50k, ..., 1M}`; HGR primary Success AUC is normalized trapezoidal
  area on those 40 observed points: `trapz(success, step) / (1,000,000 - 25,000)`.
- Reference aggregates and every paired comparison use exactly
  `{50k, 100k, ..., 1M}`; their AUC denominator is `1,000,000 - 50,000`.
  Missing points are rejected; no values are interpolated or carried forward.
- Best-final drop is `best_success - final_success`.
- Hit90 is the first observed grid step with success at least 0.90. The median
  uses only seeds that hit; the count always reports the denominator.
- HGR summary-native final, best, AUC, and Hit90 values are audited against the
  recomputed complete 25k grid. For the compact reference artifacts, final is
  audited against the summary while best, AUC, and Hit90 are recomputed on the
  preserved common 50k grid.

## Final HGR Diagnostics

The table uses each seed's 1M evaluation row. Full 25k-grid traces, including
all logged `hgr/*` fields, are stored in each `hgr_metrics_selected.csv`.

| Metric | Mean | Sample std |
|---|---:|---:|
| `hgr/relabel_fraction` | 0.774219 | 0.021467 |
| `hgr/fallback_fraction` | 0.000000 | 0.000000 |
| `hgr/beta_episode` | 0.999999 | 0.000000 |
| `hgr/beta_goal` | 0.999999 | 0.000000 |
| `hgr/is_weight_mean` | 0.094136 | 0.040009 |
| `hgr/is_weight_min` | 0.000989 | 0.000501 |
| `hgr/episode_priority_mean` | 0.363998 | 0.053795 |
| `hgr/pair_priority_count` | 7650000.000000 | 0.000000 |
| `hgr/priority_safety_replacement_rate` | 0.000000 | 0.000000 |
| `runtime/hgr_sampling_sec_per_update` | 0.026270 | 0.007298 |

## Included Files

Each HGR seed directory contains only:

- `summary.json`
- `config.json`
- `env_snapshot.json`
- `metrics_selected_25k.csv` (80 source rows: train and evaluation rows at each grid step)
- `hgr_metrics_selected.csv` (40 evaluation rows)

Root analysis outputs are `aggregate_results.csv`, `per_seed_results.csv`, and
`paired_vs_her_saber.csv`. `completion_validation.json` records the strict 1M
completion gate, `input_manifest.json` hashes every input, and
`provenance/frozen_source.sha256` preserves the launch-time source manifest.
`code/build_artifact.py` rebuilds the artifact and `code/verify_artifact.py`
performs an independent integrity check.

The launch-time source manifest captures the exact frozen experiment files,
but an exact Python/package dependency lock was not captured at launch. The
saved `env_snapshot.json` records the runtime environment metadata that was
available; exact dependency-level reproduction therefore has this provenance
limitation.

Checkpoints, raw `metrics.csv`, raw JSONL, and logs are intentionally excluded.
`artifact_manifest.json` records SHA256 and byte size for every payload file.
The manifest excludes itself because a file cannot contain its own final hash;
this exclusion is explicit in the manifest schema.

## Source Runs

- HGR: `experiment_FPP/runs/sac_hgr_tdpriority_1m/`
- SAC+HER reference artifact: `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512/`
- SAC+HER+SABER reference artifact: `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_saber_critic_only_v4_ls512/`
- Exact launch-time HGR source commit: `c2be4e85a451af61dabba8e54517e9d6d4c1cbf2`
- The result commit may add a repository-root portability fix to `run.py`;
  the algorithm, trainer, and frozen configuration remain unchanged.

## Rebuild

From the repository root, with the HGR raw run root and compact SAC reference
artifacts available:

```bash
python3 experiment_FPP/paper_artifacts/fpp_sac_hgr_pair_priority_is_1m/code/build_artifact.py
python3 experiment_FPP/paper_artifacts/fpp_sac_hgr_pair_priority_is_1m/code/verify_artifact.py
```

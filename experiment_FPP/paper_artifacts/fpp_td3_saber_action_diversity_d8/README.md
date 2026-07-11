# FPP TD3 SABER Action-Diversity D8 Ablation

## Purpose

This artifact evaluates whether SABER's exact-context support predicate benefits
from requiring multiple distinct replay-action bins. The production variant keeps
the legacy policy-bin membership condition and additionally requires `Dmin=8` unique
replay-action bins. Random-Diversity-Count assigns the same number of high weights
uniformly at random within each batch as a cardinality-matched mechanism control.

No training is run by the packaging or analysis commands in this artifact.

## Threshold Selection

The seed-1 Dmin=1 probe ran for 50k steps. Dmin was selected without using success
or AUC, from counterfactual support-positive counters over the exact 20k-50k window.
The pre-registered target was `r_D/r_1` in `[0.4, 0.7]` with `r_D >= 0.03`; when no
candidate entered that interval, the pre-registered nearest-interval fallback chose
the closest candidate, then the closest midpoint, then the smallest Dmin.

| Dmin | Positive rate | Ratio to Dmin=1 |
|---:|---:|---:|
| 1 | 0.192647 | 1.000000 |
| 2 | 0.192546 | 0.999473 |
| 4 | 0.191736 | 0.995269 |
| 8 | 0.187357 | 0.972537 |

Selected threshold: `Dmin=8`. The packaged probe input is
`data/probe/fetch_pick_and_place_td3_saber_diversity_probe_d1_50k_s1_50000_ls512/metrics_selected_10k.csv`.

## Evaluation

Methods are TD3+HER, TD3+HER+SABER (legacy Dmin=1),
TD3+HER+SABER-Diverse (Dmin=8), and Random-Diversity-Count (Dmin=8), each with
the preserved seed mapping 1-5. Evaluation uses exact checkpoints
`10k, 20k, ..., 1M` (100 points). It does not interpolate, smooth, extrapolate, or
forward-fill. Success uses `eval/env_success_mean` and falls back to
`eval/success_mean` only when the primary value is absent or non-finite.

AUC is trapezoidal area divided by `1,000,000 - 10,000 = 990,000`. Hit90 is the
first exact 10k-grid checkpoint with success at least 0.90. Means are arithmetic
means and standard deviations are sample standard deviations (`ddof=1`). Paired
AUC intervals use 10,000 paired percentile bootstrap resamples with RNG seed 27027;
paired seed differences, not methods independently, are resampled.

| Method | AUC mean +/- std | Final mean +/- std | Hit90 reached | Median Hit90 |
|---|---:|---:|---:|---:|
| TD3+HER | 0.371616 +/- 0.032057 | 0.840000 +/- 0.188414 | 3/5 | 980000.0 |
| TD3+HER+SABER | 0.442374 +/- 0.035036 | 0.970000 +/- 0.044721 | 5/5 | 840000.0 |
| TD3+HER+SABER-Diverse | 0.357273 +/- 0.045122 | 0.800000 +/- 0.183712 | 3/5 | 990000.0 |
| Random-Diversity-Count | 0.361263 +/- 0.052388 | 0.790000 +/- 0.265518 | 3/5 | 920000.0 |

AUC ranking is TD3+HER+SABER > TD3+HER > Random-Diversity-Count >
TD3+HER+SABER-Diverse.

| Method | Native AUC mean | Recomputed 10k AUC mean | Native - recomputed |
|---|---:|---:|---:|
| TD3+HER | 0.371616 | 0.371616 | +0.000000 |
| TD3+HER+SABER | 0.442374 | 0.442374 | +0.000000 |
| TD3+HER+SABER-Diverse | 0.357273 | 0.357273 | +0.000000 |
| Random-Diversity-Count | 0.361263 | 0.361263 | +0.000000 |

| Paired comparison | Mean AUC delta | 95% percentile CI |
|---|---:|---:|
| Diverse - main SABER | -0.085101 | [-0.112222, -0.059444] |
| Diverse - HER | -0.014343 | [-0.040354, 0.011667] |
| Diverse - Random-Diversity-Count | -0.003990 | [-0.051414, 0.065303] |

These intervals are uncertainty summaries; this artifact makes no additional
statistical-significance claim. Diverse D8 is lower than main SABER in this run set,
while its intervals against HER and the cardinality-matched random control include
zero. The random control therefore does not support attributing a difference to
action diversity alone.

## Contents

- `data/`: sanitized configs, environment snapshots, summaries, and compact exact-10k CSVs.
- `code/`: training implementation snapshot, scanner tests, threshold selector, analyzer, and tests.
- `results/`: recomputed tables, paired intervals, report, and validation record.
- `input_manifest.json`: source artifact paths, provenance, sizes, and SHA256 digests.
- `artifact_manifest.json`: packaged-file hashes, coverage, dependencies, and generation metadata.

Checkpoints, full per-update metrics, JSONL logs, and run logs are intentionally
excluded. No PDF or PNG is included because the original analysis produced tabular
outputs only.

## Reproduction

From this artifact directory, with NumPy installed from `requirements.txt`:

```bash
python code/test_analyze_results.py
python code/test_select_diversity_threshold.py
python code/select_diversity_threshold.py \
  --metrics data/probe/fetch_pick_and_place_td3_saber_diversity_probe_d1_50k_s1_50000_ls512/metrics_selected_10k.csv \
  --output /tmp/selected_threshold.json
python code/analyze_results.py \
  --repo-root . \
  --her-root data/her \
  --saber-root data/saber \
  --diverse-root data/diverse_d8 \
  --random-count-root data/random_count_d8 \
  --output-dir /tmp/td3_saber_action_diversity_d8
```

Numba scanner tests can be run with the training dependencies in
`code/requirements-numba.txt`:

```bash
python code/src/test_numba_support_scan_fpp.py
python code/src/test_numba_runtime_opt_fpp.py
```

## Provenance

- Source repository ID: `ARTIFACT_REPO`
- Source branch: `main`
- Source repository commit at packaging: `55e620046b44fcce03f8eccb8fb125462a580247`
- Evaluation source commit for tracked HER and main-SABER inputs is recorded per method in `input_manifest.json`.
- Diverse D8, Random-Diversity-Count D8, and probe files were generated from an uncommitted experiment implementation snapshot; their source commit is `null`, with exact source and packaged hashes retained in the manifests.

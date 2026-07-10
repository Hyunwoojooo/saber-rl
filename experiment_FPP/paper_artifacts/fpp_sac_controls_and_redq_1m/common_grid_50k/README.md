# FPP SAC Common-Grid 50k Reanalysis

## Purpose

This package removes the evaluation-frequency mismatch between legacy
FetchPickAndPlace SAC HER/SABER runs and newer control runs. It recomputes all
reported learning-curve metrics on the exact shared grid
"50k, 100k, ..., 1M" (20 checkpoints).

No interpolation, smoothing, forward filling, or extrapolation is used. No new
RL training is performed.

## Evaluation Values

For each CSV row, "eval/env_success_mean" is used when finite. The analyzer
falls back to "eval/success_mean" only when the preferred value is absent or
non-finite. Training-only rows are ignored. Duplicate finite evaluation rows at
the same step must agree within an absolute tolerance of 1e-12.

## Metrics

For grid steps s_i and success values y_i, common-grid AUC is

~~~text
sum_i ((s_(i+1) - s_i) * (y_i + y_(i+1)) / 2)
-------------------------------------------------
                 1,000,000 - 50,000
~~~

The normalization span is therefore 950,000 environment steps.
"common_grid_hit90" is the first exact common-grid checkpoint whose success is
at least 0.90. If no checkpoint reaches 0.90, Hit90 is unreached.

All seven methods use seeds 1-5. Aggregate means are arithmetic means and
standard deviations are sample standard deviations (ddof=1). Paired AUC
confidence intervals use 10,000 paired percentile-bootstrap resamples with RNG
seed 27027; seed differences are resampled, not methods independently.

## Native Versus Common-Grid Results

| Method | Native AUC mean | Common AUC mean +/- std | Common - native | Final success mean | Hit90 reached | Median Hit90 |
|---|---:|---:|---:|---:|---:|---:|
| HER | 0.462949 | 0.471053 +/- 0.079494 | +0.008104 | 0.790000 | 3/5 | 800000 |
| HER+SABER | 0.565385 | 0.583421 +/- 0.030186 | +0.018036 | 0.990000 | 5/5 | 750000 |
| Random-Support | 0.463788 | 0.482105 +/- 0.068967 | +0.018317 | 0.690000 | 2/5 | 725000 |
| Random-Fixed-Count | 0.505051 | 0.525526 +/- 0.059032 | +0.020476 | 1.000000 | 5/5 | 900000 |
| HER high-RR x4 | 0.739141 | 0.767895 +/- 0.097224 | +0.028753 | 0.960000 | 5/5 | 450000 |
| REDQ-HER TQB | 0.838283 | 0.866579 +/- 0.051983 | +0.028296 | 1.000000 | 5/5 | 300000 |
| SABER diagnostic | 0.558636 | 0.583421 +/- 0.030186 | +0.024785 | 0.990000 | 5/5 | 750000 |

The primary-method ranking is maintained: **yes**.

Native and common-grid primary ranking:

~~~text
REDQ-HER TQB > HER high-RR x4 > HER+SABER > Random-Fixed-Count > Random-Support > HER
~~~

The optional SABER diagnostic has the same common-grid AUC curve as the main
HER+SABER condition and is retained as a separate audit row.

## Paired Common-Grid AUC Differences

| Comparison | Mean paired difference | 95% paired bootstrap CI |
|---|---:|---:|
| SABER - HER | +0.112368 | [0.064211, 0.160526] |
| SABER - Random-Support | +0.101316 | [0.034211, 0.163684] |
| SABER - Random-Fixed-Count | +0.057895 | [0.002632, 0.113158] |
| SABER - high-RR x4 | -0.184474 | [-0.235263, -0.122895] |
| REDQ-HER TQB - SABER | +0.283158 | [0.212632, 0.329474] |

The intervals are reported directly; this package makes no broader statistical
significance claim.

## Reproduction

From the repository root:

~~~bash
python -m pip install -r experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/common_grid_50k/requirements.txt

python experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/common_grid_50k/code/test_analyze_sac_common_grid.py -v

python experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/common_grid_50k/code/analyze_sac_common_grid.py \
  --repo-root . \
  --artifact-root experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m \
  --include-diagnostic \
  --output-dir experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/common_grid_50k/results
~~~

The package was generated in blind repository ID "ARTIFACT_REPO", branch
"main", commit "71325ee96336a507b4967f992d3ad90fe10cc8e3". Input provenance uses
blind IDs "SOURCE_REPO_A" and "SOURCE_REPO_B"; exact source commits and
repo-relative paths are recorded in "input_manifest.json".

## Package Contents

- "code/": analyzer and seven CPU-only tests.
- "results/": four CSV tables, grid validation JSON, and Markdown report.
- "requirements.txt": pinned NumPy dependency used for the packaged analysis.
- "input_manifest.json": source paths and SHA256 for all 70 input files.
- "artifact_manifest.json": package settings, coverage, and packaged-file hashes.

"sac_common_grid_curve.pdf" and "sac_common_grid_curve.png" are intentionally
omitted because matplotlib was not installed in the packaging environment.
The figures are optional and their omission does not affect any numerical
result.

The package contains no checkpoints, model weights, replay buffers, raw
large-scale metrics, run logs, usernames, home-directory paths, hostnames, or
non-blind repository URLs.

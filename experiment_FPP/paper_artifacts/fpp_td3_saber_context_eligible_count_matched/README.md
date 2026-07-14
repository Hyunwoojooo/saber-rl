# FPP TD3 SABER Context-Eligible Count-Matched Control

## Purpose

This artifact isolates membership-aligned sample assignment from context-gated,
count-preserving critic reweighting in FetchPickAndPlace-v4 TD3:

- E is the local-count/context-eligible mask immediately before action membership.
- M is E intersected with current-policy action-bin membership (Main SABER).
- k is sum(M).
- Z uniformly samples exactly k indices without replacement from the true indices
  of E (the control assignment).

The control recomputes E, counterfactual M, and k from the current trajectory at
every update. Z uses only E, k, and a dedicated assignment RNG; it does not use
M locations, policy actions, or action bins. Natural overlap between M and Z is
allowed. k=0 yields an all-false Z, k=|E| yields Z=E, and k>|E| fails immediately.

Both Main and control retain support_query_mode=current_policy. Positive count,
context eligibility, and raw/normalized weight histograms are matched per update;
only membership-aligned assignment differs. Context-count-only is included as a
count-unmatched descriptive reference. Random-support is excluded because its
generation provenance is not established here. Packaging and analysis run no
training.

## Evaluation

The package contains Main SABER, the context-eligible count-matched control, and
the count-unmatched context-only reference, each with seeds 1-5. Evaluation uses
the exact grid 10k, 20k, ..., 1M (100 points).
No interpolation, smoothing, extrapolation, or forward filling is used. Success
uses 'eval/env_success_mean' and falls back to
'eval/success_mean' only when the primary value is absent or non-finite.

AUC is trapezoidal area divided by 1,000,000 - 10,000 = 990,000. Hit90 is the
first exact-grid checkpoint with success at least 0.90. Means are arithmetic means
and standard deviations are sample standard deviations (ddof=1).

| Method | AUC mean +/- std | Final mean +/- std | Best mean +/- std | Hit90 reached | Median Hit90 |
|---|---:|---:|---:|---:|---:|
| TD3+HER+SABER (Main) | 0.442374 +/- 0.035036 | 0.970000 +/- 0.044721 | 0.990000 +/- 0.022361 | 5/5 | 840000 |
| Context-eligible count-matched | 0.390758 +/- 0.069952 | 0.910000 +/- 0.096177 | 0.930000 +/- 0.057009 | 4/5 | 955000 |
| Context-count-only (count-unmatched reference) | 0.395657 +/- 0.089501 | 0.800000 +/- 0.242384 | 0.900000 +/- 0.106066 | 3/5 | 850000 |

AUC ranking: TD3+HER+SABER (Main) > Context-count-only (count-unmatched reference) > Context-eligible count-matched.

| Method | Native AUC mean | Recomputed AUC mean | Native - recomputed |
|---|---:|---:|---:|
| TD3+HER+SABER (Main) | 0.442374 | 0.442374 | +0.000000 |
| Context-eligible count-matched | 0.390758 | 0.390758 | +0.000000 |
| Context-count-only (count-unmatched reference) | 0.395657 | 0.395657 | +0.000000 |

## Primary Paired Comparison

Intervals are 95% paired percentile bootstrap CIs from 10,000 resamples using RNG
seed 27027. Paired seed differences are resampled directly. Exact sign-flip
p-values are two-sided over all 32 sign assignments for five paired seeds.

| Comparison | Mean AUC delta | 95% percentile CI | Exact sign-flip p |
|---|---:|---:|---:|
| Main SABER - Context-eligible count-matched | +0.051616 | [-0.016465, 0.099192] | 0.250000 |

The intervals, exact sign-flip values, and leave-one-seed-out summaries are
reported as generated. This artifact makes no additional statistical-significance
claim.

The CI includes zero; an independent membership-assignment contribution is not established.
This is an FPP TD3 result only. It does not establish a universal mechanism or
task-agnostic causality.

## Assignment Diagnostics

Values aggregate non-overlapping 1,000-update windows over each control seed.

| Seed | E rate | Main rate | Assigned rate | Overlap/k | Jaccard |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.959056 | 0.210955 | 0.210955 | 0.219929 | 0.126282 |
| 2 | 0.957385 | 0.209422 | 0.209422 | 0.218603 | 0.124403 |
| 3 | 0.956419 | 0.211973 | 0.211973 | 0.221471 | 0.126079 |
| 4 | 0.955538 | 0.239731 | 0.239731 | 0.250916 | 0.145805 |
| 5 | 0.951159 | 0.231479 | 0.231479 | 0.243106 | 0.140572 |

## Contents

- 'data/': sanitized configs, environment snapshots, summaries, and compact
  exact-10k metrics for all 15 runs plus control assignment diagnostics.
- 'code/': implementation snapshot, analyzer, unit tests, and standalone package
  parity test.
- 'results/': recomputed result tables, report, and validation record.
- 'input_manifest.json': source paths, provenance, sizes, and SHA256 digests.
- 'artifact_manifest.json': packaged-file hashes, coverage, dependencies, and
  generation metadata.

Checkpoints, full per-update metrics, JSONL files, and logs are intentionally
excluded. No PDF or PNG is included because the analysis is tabular.

## Reproduction

Run from this artifact directory:

~~~bash
python code/test_analyze_results.py
python code/test_packaged_config_parity.py
python code/analyze_results.py \
  --repo-root . \
  --main-root data/main \
  --control-root data/control \
  --context-only-root data/context_only \
  --source-parity-evidence code/source_parity_evidence.json \
  --output-dir /tmp/fpp_td3_saber_context_eligible_count_matched
~~~

Assignment and scanner tests require the dependencies in
'code/requirements-numba.txt':

~~~bash
python code/src/test_context_eligible_count_matched.py
python code/src/test_support_query_modes.py
python code/src/test_numba_support_scan_fpp.py
python code/src/test_numba_runtime_opt_fpp.py
~~~

The current-policy legacy parity test requires the full repository and is not
packaged. Its captured passing evidence is included in
'code/source_parity_evidence.json'.

## Provenance

- Source repository ID: ARTIFACT_REPO
- Source branch: main
- Source repository commit at packaging: 19233d5d7af82f6ed95efe31919b1816e918b986
- Per-method source commits and exact input SHA256 values are recorded in
  'input_manifest.json'. Unknown run provenance is recorded as null, not inferred.

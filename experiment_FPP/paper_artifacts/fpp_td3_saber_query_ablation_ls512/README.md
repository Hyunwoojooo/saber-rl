# FPP TD3 SABER Support-Query Ablation

## Purpose

This artifact tests whether SABER needs the current policy action as the query in
its exact-context support predicate. All training settings and seed mappings are
fixed. Only the query mode changes:

- current_policy: the original main SABER query; existing runs are reused.
- replay_action: the sampled replay action is queried and remains in the pool.
- shuffled_policy: a batch derangement preserves the policy-action distribution
  while removing sample alignment.
- context_count_only: action membership is removed and only context counts remain.

Replay-query can self-support because the sampled transition is deliberately not
removed from the local pool. Context-count-only is not a strict one-factor control
because removing membership changes the positive count. No training is run by the
packaging or analysis commands.

## Evaluation

The package contains TD3+HER, main TD3+HER+SABER, and the three query variants,
each with seeds 1-5. Evaluation uses the exact grid 10k, 20k, ..., 1M (100 points).
No interpolation, smoothing, extrapolation, or forward filling is used. Success
uses 'eval/env_success_mean' and falls back to
'eval/success_mean' only when the primary value is absent or non-finite.

AUC is trapezoidal area divided by 1,000,000 - 10,000 = 990,000. Hit90 is the
first exact-grid checkpoint with success at least 0.90. Means are arithmetic means
and standard deviations are sample standard deviations (ddof=1).

| Method | AUC mean +/- std | Final mean +/- std | Hit90 reached | Median Hit90 |
|---|---:|---:|---:|---:|
| TD3+HER | 0.371616 +/- 0.032057 | 0.840000 +/- 0.188414 | 3/5 | 980000 |
| TD3+HER+SABER (Main) | 0.442374 +/- 0.035036 | 0.970000 +/- 0.044721 | 5/5 | 840000 |
| Replay-query | 0.367778 +/- 0.042310 | 0.660000 +/- 0.219089 | 1/5 | 920000 |
| Shuffled-policy-query | 0.346616 +/- 0.051300 | 0.550000 +/- 0.226385 | 1/5 | 890000 |
| Context-count-only | 0.395657 +/- 0.089501 | 0.800000 +/- 0.242384 | 3/5 | 850000 |

AUC ranking: TD3+HER+SABER (Main) > Context-count-only > TD3+HER > Replay-query > Shuffled-policy-query.

| Method | Native AUC mean | Recomputed AUC mean | Native - recomputed |
|---|---:|---:|---:|
| TD3+HER | 0.371616 | 0.371616 | +0.000000 |
| TD3+HER+SABER (Main) | 0.442374 | 0.442374 | +0.000000 |
| Replay-query | 0.367778 | 0.367778 | +0.000000 |
| Shuffled-policy-query | 0.346616 | 0.346616 | +0.000000 |
| Context-count-only | 0.395657 | 0.395657 | +0.000000 |

## Paired Comparisons

Intervals are 95% paired percentile bootstrap CIs from 10,000 resamples using RNG
seed 27027. Paired seed differences are resampled directly. Exact sign-flip
p-values are two-sided over all 32 sign assignments for five paired seeds.

| Comparison | Mean AUC delta | 95% percentile CI | Exact sign-flip p |
|---|---:|---:|---:|
| Main SABER - Replay-query | +0.074596 | [0.059343, 0.092273] | 0.062500 |
| Main SABER - Shuffled-policy | +0.095758 | [0.063131, 0.127576] | 0.062500 |
| Main SABER - Context-count-only | +0.046717 | [-0.040909, 0.121919] | 0.312500 |

The intervals, exact sign-flip values, and leave-one-seed-out summaries are
reported as generated. This artifact makes no additional statistical-significance
claim.

## Mechanism Snapshots

Values are means over exact 10k evaluation snapshots and then arithmetic means
across available seeds. Blank legacy fields are not imputed.

| Method | Context eligible | Support positive | Query present given context |
|---|---:|---:|---:|
| TD3+HER+SABER (Main) |  | 0.219063 |  |
| Replay-query | 0.958000 | 0.425859 | 0.444305 |
| Shuffled-policy-query | 0.955125 | 0.153805 | 0.160971 |
| Context-count-only | 0.956297 | 0.956297 |  |

## Contents

- 'data/': sanitized configs, environment snapshots, summaries, and compact
  exact-10k metrics for all 25 runs.
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
  --her-root data/her \
  --saber-root data/saber \
  --replay-root data/replay \
  --shuffled-root data/shuffled \
  --context-only-root data/context_only \
  --output-dir /tmp/fpp_td3_saber_query_ablation
~~~

Scanner and query-mode tests require the dependencies in
'code/requirements-numba.txt':

~~~bash
python code/src/test_support_query_modes.py
python code/src/test_numba_support_scan_fpp.py
python code/src/test_numba_runtime_opt_fpp.py
~~~

The full-repository-only current-policy legacy parity test is intentionally not
packaged because it references the sibling baseline source tree. It passed before
the existing main SABER results were reused.

## Provenance

- Source repository ID: ARTIFACT_REPO
- Source branch: main
- Source repository commit at packaging: 0e571769bd23366687edfb4967aa655120c74ef2
- Per-method source commits and exact input SHA256 values are recorded in
  'input_manifest.json'. Unknown run provenance is recorded as null, not inferred.

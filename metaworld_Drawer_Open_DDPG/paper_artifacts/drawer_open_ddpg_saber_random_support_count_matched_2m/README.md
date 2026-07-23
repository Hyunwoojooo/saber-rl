# DrawerOpen DDPG Random-Support (Batch-Wide Count-Matched), 2M

This compact paper artifact compares the matched DrawerOpen DDPG HER baseline,
SABER Main, and the Random-Support count-matched control for seeds 1–5.

Random-Support computes the counterfactual Main support mask on its own
minibatch and replay state, preserves that update's number of positive weights,
and assigns the same number of positive weights uniformly across the minibatch.
It therefore preserves the per-update positive count and weight histogram while
breaking sample-level support membership.

## Results

| Method | Normalized success AUC (mean ± sample SD) | Final success (mean ± sample SD) | Hit90 |
|---|---:|---:|---:|
| HER | 0.207312 ± 0.286818 | 0.390000 ± 0.534322 | 960k (2/5) |
| Random-Support (count matched) | 0.706482 ± 0.121876 | 1.000000 ± 0.000000 | 510k (5/5) |
| SABER Main | 0.753668 ± 0.044374 | 1.000000 ± 0.000000 | 500k (5/5) |

The paired Main-minus-Random AUC difference is **+0.047186**. Its ordered
exhaustive paired-bootstrap 95% interval is **[-0.042010, 0.136382]**, with
positive differences in 3/5 seeds and an exact two-sided sign-flip
`p = 0.4375`. The interval includes zero, so this result does not isolate an
independent benefit of exact action-bin membership.

All AUC values use only the 200 observed `eval/env_success_mean` evaluations on
the exact 10k–2M grid. There is no fallback metric, smoothing, interpolation,
forward filling, or extrapolation.

## Positive-count invariant

The package retains every finite logged pair of
`reference_high_weight_rate` and `assigned_high_weight_rate`. Across 50,880
rows (10,176 per seed), the two values match exactly: zero mismatches and
maximum absolute difference 0. This verifies the intended batch-wide
count-matching invariant from packaged evidence rather than from a summary
claim alone.

## Provenance caveat

The Random-Support raw result commit is
`86912701b2ca0e26586c43748f2a984c0f301960`. That commit preserves raw metrics,
summaries, and the launcher, but not the original per-run config or environment
snapshot. Random-Support config files in this artifact are therefore explicitly
labelled reconstructions from the frozen launcher, and environment files are
explicitly labelled same-seed proxies. They are not represented as original
run files. The recorded `weighting_mode=batch_norm` is launcher-explicit;
`support_query_mode=current_policy` is the effective value derived from the
frozen method source/default and was not originally serialized.

HER and Main compact data are reused, without retraining, from the independently
verified replay-action artifact at commit
`1a184f69588c0d02ece942bc106845a4516e81cb`; its comparator configs and
environment records already carry the same reconstructed/proxy labels.

The `code/src` tree is unchanged between ancestor commit
`7c870d78b6b064d45432c84a8e9b4f1c624a2357` and the Random-Support result
commit. This is source-tree consistency evidence, not proof that the recorded
runs executed a particular commit. The frozen result-commit source and launcher
are included under `source_snapshot/`.

## Verify

From this directory:

```bash
python3 verify_artifact.py --artifact-dir .
```

The verifier checks the complete package manifest, all 15 exact evaluation
curves, AUC/final/Hit90 statistics, all 3,125 ordered paired-bootstrap
resamples, all 32 sign assignments, the 50,880-row count invariant, completion
records, frozen source hashes, and reconstructed/proxy provenance labels.

Checkpoints, full raw metrics, JSONL, logs, and temporary outputs are excluded.

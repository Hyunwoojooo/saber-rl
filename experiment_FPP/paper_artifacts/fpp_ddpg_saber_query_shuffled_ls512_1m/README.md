# FPP DDPG SABER Shuffled-Policy Query Ablation

## Purpose

This artifact tests whether sample-aligned current-policy querying is important to
FPP DDPG SABER. In the shuffled arm, a fixed-point-free minibatch derangement
preserves the complete current-policy-action multiset but assigns each action to a
different sample's unchanged context. Only `support_query_mode` changes from
`current_policy` to `shuffled_policy`.

This is **shuffled-policy query**, not shuffled-weight and not random-support.
The shuffled action is used only by support membership. Critic Bellman regression
still uses the sample's replay action, the resulting critic weight stays on that
sample's residual, and the actor remains unweighted DDPG.

## Evaluation protocol

- Seeds: 1–5, paired by seed.
- Exact evaluation grid: 10k, 20k, ..., 1M (100 points per seed).
- Primary success field only: `eval/env_success_mean`.
- No smoothing, interpolation, forward filling, or extrapolation.
- Normalized AUC: trapezoidal integral over 10k–1M divided by 990,000.
- Standard deviations are sample SD (`ddof=1`).
- Paired bootstrap: 10,000 percentile resamples, RNG seed 27027.
- Exact two-sided sign flip: all 32 assignments.

| Arm | AUC mean ± SD | Final mean ± SD | Best mean ± SD | Hit90 reached | Median Hit90 |
|---|---:|---:|---:|---:|---:|
| Main SABER | 0.609091 ± 0.037778 | 0.960000 ± 0.041833 | 1.000000 ± 0.000000 | 5/5 | 540000.0 |
| Shuffled-policy query | 0.566313 ± 0.035876 | 0.950000 ± 0.070711 | 1.000000 ± 0.000000 | 5/5 | 630000.0 |

Main − Shuffled paired AUC is +0.042778
(sample SD 0.052774, 95% paired-bootstrap CI
[-0.001515, +0.085152],
exact sign-flip p=0.187500).
All five seeds and all paired statistics are retained regardless of direction.
If the interval includes zero, this artifact does not claim that an independent
alignment effect has been established.

## Contents and provenance

- `data/main/` and `data/shuffled/`: sanitized config, environment snapshot,
  summary, and one-row-per-checkpoint exact-grid metrics for seeds 1–5.
- `results/`: deterministic analysis tables and validation record.
- `code/`: frozen execution/analysis source snapshot.
- `tests/`: packaged analysis/artifact tests.
- `input_manifest.json`: original input paths, sizes, SHA256, and source commits.
- `artifact_manifest.json`: every packaged file's size and SHA256.
- `frozen_source.sha256`: aggregate SHA256 of the frozen `code/` tree.

Execution source commit: `7bfb233bb6b3f08d704aa79d24a2ac263e01dbaf`.

Checkpoints, full per-update metrics, JSONL, large logs, smoke outputs, caches, and
temporary files are intentionally excluded.

## Verification

From this directory:

```bash
python verify_artifact.py --artifact-dir .
```

# FPP DDPG SABER Replay-Action Query LS512 (1M)

Compact matched seeds 1-5 paper artifact.

- Execution source commit: `b4c29b1339ecb5ead598b101438e061703293306`
- Canonical Main reference source commit: `c43d40502b84ae713924ebf995be623e7acc7ca8`
- Scientific factor: `support_query_mode: current_policy -> replay_action`
- Backbone/scope: DDPG, HER=0.8, SABER critic-only, LS512

The primary endpoint is normalized trapezoidal environment-success AUC on the
exact observed 10k, 20k, ..., 1M grid (100 points; denominator 990,000). The
fallback success column is used only when the environment-success value is
absent/non-finite. There is no interpolation, smoothing, extrapolation, or
forward filling.

Main minus Replay mean AUC delta is `+0.011414` with paired
percentile-bootstrap 95% CI `[-0.028081,
0.058802]` (10,000 resamples, RNG seed 27027).
The exact two-sided sign-flip test enumerates all 32 assignments.

Replay-action querying retains the sampled transition in the local replay pool and can therefore self-support.

The Replay execution commit was captured by the production launcher. The
historical Main run files do not capture an execution commit SHA; the Main Git
commit reported here is therefore provenance for the frozen canonical reference
source, not proof that this exact commit launched those historical runs.

## Contents

- `code/execution_source/`: Replay source read byte-for-byte from its captured
  execution commit.
- `code/main_source/`: canonical Main reference source read byte-for-byte from
  its recorded reference commit.
- `code/source_provenance.json`: self-contained Git commit/tree/blob proof
  objects for both frozen source trees.
- remaining `code/` files: analysis/build/verification code.
- `data/main/`, `data/replay/`: sanitized per-seed config, environment snapshot,
  summary, and exact-grid compact metrics only.
- `results/`: exact analysis tables, query diagnostics, validation, and report.
- `tests/`: analysis/artifact tests and captured prepackage test evidence.
- `input_manifest.json`: input provenance, source hashes, and sanitization record.
- `artifact_manifest.json` and `SHA256SUMS`: complete packaged integrity records.

Large checkpoints, full per-update metrics, JSONL, raw logs, and smoke outputs are
excluded.

Verify from a copied artifact, without the repository:

```bash
python3 code/verify_artifact.py --artifact-root .
```

Regenerate analysis from packaged inputs:

```bash
python3 code/analyze_results.py --main-root data/main --replay-root data/replay --output-dir /tmp/query-results
```

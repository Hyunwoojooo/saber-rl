# FPP SABER DDPG/TD3/SAC Mechanism Diagnostics

This artifact is the paper-facing source of truth for the FetchPickAndPlace
(FPP) SABER mechanism diagnostics across DDPG, TD3, and SAC. It consolidates
the frozen per-seed inputs, a common analysis contract, derived tables,
provenance records, and deterministic rebuild and verification programs.
It does not launch training.

The package covers 15 completed runs:

- backbones: DDPG, TD3, and SAC;
- seeds: 1, 2, 3, 4, and 5 for every backbone;
- training: 1,000,000 environment steps and 995,001 updates per run;
- diagnostics: one full 256-transition minibatch every 10,000 steps from
  10,000 through 1,000,000;
- coverage: 100 diagnostic batches and 25,600 full-batch observations per
  run, or 1,500 batches and 384,000 observations in total.

## Main descriptive results

The table reports the mean ± sample standard deviation across five seeds.
Support rate is sample-weighted within seed. Top-k enrichment and residual
reduction are equal-update means within seed. Definitions are frozen in
[METRICS.md](METRICS.md).

| Backbone | Diagnostic support-positive rate | Top-10% TD enrichment | Top-20% TD enrichment | Support-positive residual reduction | Support-negative residual reduction |
|---|---:|---:|---:|---:|---:|
| DDPG | 0.366781 ± 0.011071 | 1.408986 ± 0.076658 | 1.326273 ± 0.051499 | 0.019173 ± 0.003150 | 0.008011 ± 0.002060 |
| TD3 | 0.219063 ± 0.009077 | 1.670044 ± 0.074692 | 1.581926 ± 0.085786 | 0.021341 ± 0.001493 | 0.006469 ± 0.003373 |
| SAC | 0.196992 ± 0.012526 | 1.828055 ± 0.088809 | 1.747591 ± 0.049297 | 0.022260 ± 0.004713 | 0.005656 ± 0.001433 |

These are mechanism-consistent observations: in each backbone, the
support-positive subset is enriched among high absolute TD-error samples and
has a larger immediate residual reduction than the support-negative subset.
They are not a causal demonstration that SABER support weighting improves
policy performance.
This is not causal proof.

Residuals are measured on the same minibatch immediately before and after one
critic update while holding the pre-update TD target fixed. They are
same-minibatch, in-sample measurements, not held-out measurements and not
evidence of generalization. Cross-backbone pooling is forbidden and no
cross-backbone grand mean is produced.

## Inputs and compact-data policy

The primary paper-facing input is each run's `diagnostic_metrics.jsonl`. Every
row summarizes the complete 256-transition training minibatch. The copied
inputs for each algorithm and seed are:

```text
data/<ddpg|td3|sac>/seed_<1..5>/
  config.json
  env_snapshot.json
  summary.json
  metrics_selected_10k.csv
  diagnostic_metrics.jsonl
```

The original `diagnostic_samples.jsonl` files contain only an isolated-RNG
128-of-256 sample subset from each diagnostic minibatch. They are secondary
exploratory logs, are excluded from headline calculations, and are not copied
into this compact artifact. All 15 excluded files remain recoverable from
their immutable data snapshots; their source paths, retained source file IDs,
sizes, and SHA256 values are recorded in
`provenance/external_raw_samples.json`. Their combined size is 228,710,866
bytes.

The final-checkpoint or training-curve support rate in `summary.json` or
`metrics_selected_10k.csv` is not the diagnostic-window support-positive rate
reported here. The two quantities have different grains and must not be
substituted for each other.

Checkpoints, full per-update `metrics.csv`/`metrics.jsonl`, raw training logs,
and the secondary `diagnostic_samples.jsonl` files are deliberately excluded.

## Layout

```text
README.md
METRICS.md
code/
  rebuild_results.py
  build_manifests.py
  verify_artifact.py
data/
  ddpg/seed_1 ... seed_5/
  td3/seed_1 ... seed_5/
  sac/seed_1 ... seed_5/
results/
  diagnostic_by_step.csv
  per_seed_diagnostics.csv
  aggregate_by_backbone.csv
  support_rate.csv
  td_error_enrichment.csv
  residual_reduction.csv
  config_summary.csv
  data_coverage.csv
  schema_compatibility.csv
  analysis_results.json
provenance/
  source_snapshots/
  source_manifest.json
  external_raw_samples.json
input_manifest.json
artifact_manifest.json
validation.json
```

`diagnostic_by_step.csv` is the normalized 1,500-row batch table.
`per_seed_diagnostics.csv` is the statistical analysis unit used to construct
`aggregate_by_backbone.csv`. The three focused CSV files expose support rate,
TD-error enrichment, and residual reduction without requiring readers to
filter the wider tables.

## Rebuild and verify

From this artifact directory, run:

```bash
python3 -B code/rebuild_results.py --artifact-root .
python3 -B code/build_manifests.py --artifact-root .
python3 -B code/verify_artifact.py --artifact-root .
```

The first command regenerates all derived result tables and `validation.json`
from the 15 copied run inputs. The second regenerates the input, source, and
artifact SHA256 manifests after the result files are rebuilt. The third is a
fail-closed verifier for file hashes, exact algorithm/seed coverage, the
100-point 10k grid, 1M completion, schemas, finite values, metric identities,
and the documented provenance boundary. None of these commands runs training.

## Provenance

The artifact distinguishes a data snapshot from a verified
execution-source snapshot:

DDPG and SAC execution-source snapshots are unverified. The TD3 execution-source snapshot is verified.

| Backbone | Repository | data snapshot | Execution-source status |
|---|---|---|---|
| DDPG | `anonymous_repository/source_a` | `snapshot_reference_016` | Unverified: no execution snapshot ID was recorded |
| TD3 | `anonymous_repository/source_a` | `snapshot_fpp_ddpg_high_rr_v1` | Verified at the same anonymous snapshot |
| SAC | `anonymous_repository/source_b` | `snapshot_reference_017` | Unverified: no execution snapshot ID was recorded |

For DDPG and SAC, the package preserves candidate launcher, configuration, and
trainer snapshots with SHA256 hashes under `provenance/source_snapshots/`.
Those snapshots are useful for auditing the logged schema, but this artifact
does not relabel them as proven execution source. Consequently,
`validation.json` reports partial execution-source provenance even when every
data and analysis invariant passes. TD3 is the only backbone here with a
verified execution-source snapshot.

The preserved `legacy_analyzer.py` files are historical formula references;
only `code/rebuild_results.py` is canonical for this integrated artifact.

## Interpretation boundary

DDPG uses one critic. TD3 and SAC use two critics, and their reported
per-sample absolute TD residual is the arithmetic mean of the two critics'
absolute residuals. Absolute residual magnitudes are therefore
algorithm- and critic-family-dependent. The artifact supports within-backbone
support-positive versus support-negative diagnostics; it does not treat the
three backbones as exchangeable observations, pool their samples, or report a
single grand mean.

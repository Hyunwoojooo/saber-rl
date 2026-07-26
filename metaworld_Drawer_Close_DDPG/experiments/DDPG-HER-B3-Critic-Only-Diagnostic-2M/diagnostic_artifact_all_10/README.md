# DrawerClose DDPG SABER Diagnostic: Seeds 1--10

This summary-only artifact combines the two validated DrawerClose DDPG
HER+SABER diagnostic cohorts. Seeds are the independent aggregation units.

## Source artifacts

- `seeds_1_5`: seeds 1--5, `metaworld_Drawer_Close_DDPG/experiments/DDPG-HER-B3-Critic-Only-Diagnostic-2M/diagnostic_artifact` at `d2038c7c964e73d0a8862ff3a5d361c21ae2f046`
- `seeds_6_10`: seeds 6--10, `metaworld_Drawer_Close_DDPG/experiments/DDPG-HER-B3-Critic-Only-Diagnostic-2M/diagnostic_artifact_s6_10` at `e990acdd30923bc8249f08b53d3abb7f7ae52ce9`

The source artifacts retain all per-seed compressed diagnostic snapshots and
sanitized metadata. This directory deliberately does not duplicate those raw
files. `provenance.json` records each source commit, repository path, source
manifest hash, training-configuration hash, and seed-to-source mapping.

## Contents

- `results/diagnostic_by_seed.csv`: the ten source seed summaries.
- `results/diagnostic_aggregate.csv`: seed-level means, sample standard
  deviations, and 10,000-resample bootstrap confidence intervals.
- `results/diagnostic_by_phase_window.csv`: seed-level temporal summaries.
- `results/paper_summary.json`: manuscript-facing support, TD-error, residual,
  and phase aggregates.
- `validation/validation_summary.json`: source integrity, coverage, and config
  parity checks.
- `artifact_manifest.json`: byte counts and SHA-256 hashes for this artifact.

## Rebuild

From the experiment directory, run:

```bash
python analysis/build_combined_diagnostic_artifact.py
```

The builder verifies both source manifests before recomputing the aggregate.
It rejects missing seeds, duplicate seeds, incomplete diagnostics, schema
changes, source hash mismatches, and training-setting differences beyond run
names and seed values.

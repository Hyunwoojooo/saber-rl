# DrawerOpen SAC: HER Seed Extension

Published snapshot: **2026-11-11T03:54:19.265462+00:00**.
HER seeds 6-10 completed on **2026-11-10T00:13:35.004982+00:00**.

**Only HER is included in this publication.** Five new seeds (6-10), each at 2M steps,
are combined descriptively with five retained historical HER runs (1-5).
No new SABER, HGR, PER or CPU-pilot results are exported. This is not a completed four-method comparison.

## Results

Values are mean +/- sample SD (ddof=1).

| HER cohort | Seeds | AUC | FQ-AUC | Final success | Hit90 | Steps/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| historical_1_5 | 1-5 | 0.447035 +/- 0.410848 | 0.600800 +/- 0.542523 | 0.600000 +/- 0.547723 | 3/5 | 37.924 +/- 4.007 |
| extension_6_10 | 6-10 | 0.526332 +/- 0.383685 | 0.777200 +/- 0.437263 | 0.800000 +/- 0.447214 | 4/5 | 30.580 +/- 0.031 |
| pooled_1_10 | 1-10 | 0.486683 +/- 0.377089 | 0.689000 +/- 0.473746 | 0.700000 +/- 0.483046 | 7/10 | 34.252 +/- 4.703 |

### New Seed-Level Results

| Seed | AUC | FQ-AUC | Final success | Hit90 step | Runtime (h) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.773869 | 1.000000 | 1.000 | 500000 | 18.191 |
| 7 | 0.757035 | 1.000000 | 1.000 | 560000 | 18.182 |
| 8 | 0.000000 | 0.000000 | 0.000 | Not reached | 18.153 |
| 9 | 0.863568 | 1.000000 | 1.000 | 270000 | 18.163 |
| 10 | 0.237186 | 0.886000 | 1.000 | 1500000 | 18.148 |

## Interpretation and Scope

Four of five new seeds reach final success 1.0; seed 8 has zero success throughout and is retained.
The pooled result is descriptive. No SABER-minus-HER interval is computed by this artifact:
comparing ten HER seeds against five historical or selectively completed SABER seeds would not be the planned paired analysis.
The historical runs overlap the [original five-seed artifact](../drawer_open_sac_her_saber_hgr_per_2m/README.md); do not count them twice.

## Protocol and Provenance

- FPP-reference SAC with automatic entropy temperature, initial alpha=0.2, target entropy=-4, twin critics and stochastic actor. This is not the older fixed-alpha prototype.
- DrawerOpen-v3 sparse reward v2, 2M steps, UTD=1, HER ratio=0.8, horizon=200, gamma=0.98, batch=256, replay capacity=300000.
- Each run has 1,995,001 optimizer updates and 20 evaluation episodes at each of 200 checkpoints.
- AUC uses the exact 10k..2M grid divided by 1,990,000. FQ-AUC uses 1.5M..2M divided by 500,000. Curves are unsmoothed and retain all checkpoints.
- Success is eval/env_success_mean (ever-success within each episode). Hit90 is the first crossing, not sustained success.
- New seeds started from scratch; no checkpoint continuation, replacement or splicing. Historical HER seeds 1-3 were retained complete pre-outage runs; seeds 4-5 were complete fresh recovery runs.
- Exact recorded scientific configurations match across all ten HER runs after removing documented seed/output metadata. Frozen sources are byte-identical to the linked previously published source.
- The inherited original-goal observation components plus separately relabeled HER goal are unchanged. Their causal effect is not established.
- Recorded core package versions were matched in the prelaunch audit; this does not assert byte-for-byte identity of every historical machine component.
- Whole-run runtime includes evaluation, checkpoints and concurrent workloads. It is not a controlled wall-clock comparison.
- The predeclared four-method protocol is retained for provenance, but its other methods are not results in this snapshot. The fixed-size extension followed inspection of the historical outcomes.

## Files

- [Seed metrics](analysis/seed_metrics.csv), [aggregate metrics](analysis/aggregate_metrics.csv), [full evaluation curves](analysis/learning_curves.csv), [report](analysis/report.json).
- [Provenance](provenance.json), [HER completion](provenance/her_completion.json), [protocol](protocol.json), [environment audit](provenance/environment.json).
- [Executed scientific source](../../../ICML_27/experiments/draweropen_drawerclose_sac_her_vs_saber/source/) is already archived. Source hashes are checked against the extension snapshot; no generic trainer is substituted.
- data/ retains configs, summaries, evaluation rows, environment records and completion records. New runs also retain launch records and the full-log finite-loss check counts.
- Absolute workspace prefixes are redacted in JSON; unavailable optional diagnostics become null. Evaluation strings are not rounded.
- Model weights, replay buffers, full training logs and other methods' result directories are excluded. This compact artifact is not a standalone training bundle.

## Verification

From the repository root using Python 3.12:

```bash
python3 ICML_27/tools/sac_seed_extension_results.py verify
```

Verification is read-only and CPU-only. It recomputes all 2,000 evaluation points, checks source/config/completion records and manifest hashes, and never imports a trainer or starts a run.

# DrawerClose SAC: HER Seed Extension

Published snapshot: **2026-11-26T08:26:38.267093+00:00**.
HER seeds 6-10 completed on **2026-11-25T23:39:43.302516+00:00**.

**Only HER is included in this publication.** Five new seeds (6-10), each at 2M steps,
are combined descriptively with five retained historical HER runs (1-5).
No new SABER, HGR, PER or CPU-pilot results are exported. This is not a completed four-method comparison.

## Results

Values are mean +/- sample SD (ddof=1).

| HER cohort | Seeds | AUC | FQ-AUC | Final success | Hit90 | Steps/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| historical_1_5 | 1-5 | 0.260628 +/- 0.284573 | 0.247500 +/- 0.395872 | 0.200000 +/- 0.447214 | 5/5 | 37.748 +/- 4.714 |
| extension_6_10 | 6-10 | 0.096633 +/- 0.026787 | 0.031000 +/- 0.027459 | 0.000000 +/- 0.000000 | 5/5 | 30.526 +/- 0.039 |
| pooled_1_10 | 1-10 | 0.178631 +/- 0.209240 | 0.139250 +/- 0.288108 | 0.100000 +/- 0.316228 | 10/10 | 34.137 +/- 4.936 |

### New Seed-Level Results

| Seed | AUC | FQ-AUC | Final success | Hit90 step | Runtime (h) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.113819 | 0.023000 | 0.000 | 30000 | 18.216 |
| 7 | 0.070352 | 0.001000 | 0.000 | 40000 | 18.177 |
| 8 | 0.079146 | 0.015000 | 0.000 | 90000 | 18.225 |
| 9 | 0.134673 | 0.045000 | 0.000 | 30000 | 18.207 |
| 10 | 0.085176 | 0.071000 | 0.000 | 130000 | 18.174 |

## Interpretation and Scope

5/5 new seeds reach Hit90 at least once and 5/5 finish with zero success at 2M. All runs are retained.
The pooled result is descriptive. No SABER-minus-HER interval is computed by this artifact:
comparing ten HER seeds against five historical or selectively completed SABER seeds would not be the planned paired analysis.
The historical runs overlap the [original five-seed artifact](../drawer_close_sac_her_saber_hgr_per_2m/README.md); do not count them twice.

## Protocol and Provenance

- FPP-reference SAC with automatic entropy temperature, initial alpha=0.2, target entropy=-4, twin critics and stochastic actor. This is not the older fixed-alpha prototype.
- DrawerClose-v3 sparse reward v2, 2M steps, UTD=1, HER ratio=0.8, horizon=200, gamma=0.98, batch=256, replay capacity=300000.
- Each run has 1,995,001 optimizer updates and 20 evaluation episodes at each of 200 checkpoints.
- AUC uses the exact 10k..2M grid divided by 1,990,000. FQ-AUC uses 1.5M..2M divided by 500,000. Curves are unsmoothed and retain all checkpoints.
- Success is eval/env_success_mean (ever-success within each episode). Hit90 is the first crossing, not sustained success.
- New seeds started from scratch; no checkpoint continuation, replacement or splicing. Historical HER references are complete fresh recovery runs.
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
python3 ICML_27/tools/sac_seed_extension_results.py verify --task drawer_close
```

Verification is read-only and CPU-only. It recomputes all 2,000 evaluation points, checks source/config/completion records and manifest hashes, and never imports a trainer or starts a run.

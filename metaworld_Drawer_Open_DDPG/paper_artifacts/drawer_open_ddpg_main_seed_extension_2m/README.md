# DrawerOpen DDPG: Completed Main-Paper Seed Extensions

Snapshot: **2026-10-29T06:03:39.383343+00:00**. This is not a live dashboard.

This artifact contains **24 completed new runs** and 25 historical seed-level references. Each new run has 2M steps.
The earlier HER/SABER artifact's 10 new runs are included again for comparison, not counted as new independent observations.
Only completed production outputs are exported. A partially completed method has seed-level data only, not a final aggregate or paired CI.

## Queue Coverage

| Method | Completed new seeds | State at capture |
| --- | --- | --- |
| HER | 6, 7, 8, 9, 10 | completed |
| SABER (Ours) | 6, 7, 8, 9, 10 | completed |
| HGR | 6, 7, 8, 9, 10 | completed |
| PER | 6, 7, 8, 9, 10 | completed |
| Replay-Action Query | 6, 7, 8, 9 | running |
| Shuffled-Policy Query | None | queued |
| Random-Support | None | queued |
| Context-Eligible Count-Matched | None | queued |

## historical_1_5

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.207312 +/- 0.286818 | 0.392700 +/- 0.537791 | 0.390000 +/- 0.534322 | 2/5 |
| SABER (Ours) | 0.753668 +/- 0.044374 | 0.987400 +/- 0.018270 | 1.000000 +/- 0.000000 | 5/5 |
| HGR | 0.688191 +/- 0.142395 | 0.982100 +/- 0.026768 | 0.900000 +/- 0.223607 | 5/5 |
| PER | 0.825101 +/- 0.044487 | 0.981700 +/- 0.020873 | 0.990000 +/- 0.022361 | 5/5 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_her | +0.546357 | [+0.339874, +0.741005] | 5/5 |
| saber_minus_hgr | +0.065477 | [-0.047186, +0.219045] | 3/5 |
| saber_minus_per | -0.071432 | [-0.103869, -0.014020] | 1/5 |

## extension_6_10

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.594824 +/- 0.334349 | 0.789500 +/- 0.441622 | 0.800000 +/- 0.447214 | 4/5 |
| SABER (Ours) | 0.550075 +/- 0.328911 | 0.792900 +/- 0.443334 | 0.790000 +/- 0.442154 | 4/5 |
| HGR | 0.441608 +/- 0.407851 | 0.589500 +/- 0.538200 | 0.600000 +/- 0.547723 | 3/5 |
| PER | 0.475779 +/- 0.435331 | 0.594800 +/- 0.543037 | 0.600000 +/- 0.547723 | 3/5 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_her | -0.044749 | [-0.478116, +0.456080] | 1/5 |
| saber_minus_hgr | +0.108467 | [-0.390779, +0.589196] | 3/5 |
| saber_minus_per | +0.074296 | [-0.133166, +0.345528] | 2/5 |

## pooled_1_10

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.401068 +/- 0.357713 | 0.591100 +/- 0.508879 | 0.595000 +/- 0.512320 | 6/10 |
| SABER (Ours) | 0.651872 +/- 0.245907 | 0.890150 +/- 0.313065 | 0.895000 +/- 0.314863 | 9/10 |
| HGR | 0.564899 +/- 0.315961 | 0.785800 +/- 0.414574 | 0.750000 +/- 0.424918 | 8/10 |
| PER | 0.650440 +/- 0.344969 | 0.788250 +/- 0.415736 | 0.795000 +/- 0.419292 | 8/10 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_her | +0.250804 | [-0.067643, +0.543933] | 6/10 |
| saber_minus_hgr | +0.086972 | [-0.181385, +0.344536] | 6/10 |
| saber_minus_per | +0.001432 | [-0.104976, +0.148699] | 3/10 |

## Completed New Seed Results

No row below is an intermediate checkpoint. Missing seeds are intentionally not imputed or dropped to form a smaller final cohort.

| Method | Seed | AUC | FQ-AUC | Final success | Hit90 step |
| --- | ---: | ---: | ---: | ---: | ---: |
| HER | 6 | 0.782663 | 0.993000 | 1.00 | 450000 |
| HER | 7 | 0.752513 | 0.960000 | 1.00 | 470000 |
| HER | 8 | 0.752010 | 0.999000 | 1.00 | 510000 |
| HER | 9 | 0.686935 | 0.995500 | 1.00 | 660000 |
| HER | 10 | 0.000000 | 0.000000 | 0.00 | Not reached |
| SABER (Ours) | 6 | 0.765578 | 0.989000 | 1.00 | 470000 |
| SABER (Ours) | 7 | 0.000000 | 0.000000 | 0.00 | Not reached |
| SABER (Ours) | 8 | 0.576382 | 0.998000 | 1.00 | 830000 |
| SABER (Ours) | 9 | 0.570980 | 0.977500 | 0.95 | 850000 |
| SABER (Ours) | 10 | 0.837437 | 1.000000 | 1.00 | 340000 |
| HGR | 6 | 0.635427 | 0.986000 | 1.00 | 740000 |
| HGR | 7 | 0.764573 | 0.992000 | 1.00 | 490000 |
| HGR | 8 | 0.807538 | 0.969500 | 1.00 | 400000 |
| HGR | 9 | 0.000000 | 0.000000 | 0.00 | Not reached |
| HGR | 10 | 0.000503 | 0.000000 | 0.00 | Not reached |
| PER | 6 | 0.828643 | 0.998000 | 1.00 | 390000 |
| PER | 7 | 0.000000 | 0.000000 | 0.00 | Not reached |
| PER | 8 | 0.000503 | 0.000000 | 0.00 | Not reached |
| PER | 9 | 0.809296 | 0.998000 | 1.00 | 400000 |
| PER | 10 | 0.740452 | 0.978000 | 1.00 | 450000 |
| Replay-Action Query | 6 | 0.657789 | 0.990000 | 1.00 | 710000 |
| Replay-Action Query | 7 | 0.746231 | 1.000000 | 1.00 | 540000 |
| Replay-Action Query | 8 | 0.655276 | 1.000000 | 1.00 | 750000 |
| Replay-Action Query | 9 | 0.680653 | 0.996000 | 1.00 | 670000 |

## Interpretation and Protocol

- Higher sample means with a CI including zero do not establish superiority or equivalence. Pooled SABER/PER AUC means are very close; no general superiority claim follows.
- The historical, added-seed and pooled estimates are reported separately. Their differing directions and failure-to-learn outcomes must remain visible.
- Only methods with all five new seeds complete enter aggregate/paired tables. Partial methods remain explicitly incomplete even though finished seeds can be inspected.
- DrawerOpen-v3 / DDPG / sparse reward v2; 2M steps, UTD=1, 1,995,001 optimizer updates, HER ratio 0.8, horizon 200, 20 evaluation episodes every 10k steps.
- AUC uses exactly 200 evaluation points from 10k to 2M, divided by 1,990,000. FQ-AUC covers 1.5M..2M, divided by 500,000. Success is eval/env_success_mean. No smoothing, interpolation or silently shortened horizons.
- Final success is evaluated at 2M. Hit90 is the first evaluation >=0.90, not sustained task mastery. SD is across seeds with ddof=1.
- Paired percentile bootstrap resamples seed rows 10,000 times using NumPy default_rng(2027); endpoints are 2.5% and 97.5%. These are approximate, pointwise, unadjusted intervals, not simultaneous guarantees. AUC is primary; other metrics are secondary.
- Every completed zero/low-success run is retained. New runs began from scratch; no checkpoint continuation or outcome-based exclusions.
- Historical exact 3,125-assignment mechanism intervals are not overwritten: any later pooled control comparison will use the declared Monte Carlo procedure.
- Original HER/SABER/Random full configs and exact historical environment locks were not retained. New configs follow the audited canonical-source/launcher reconstruction. Retained configs were checked where available; bitwise historical environment identity is not asserted.
- Earlier diagnostic and assignment executions are not extra independent seeds in these cohorts. Source/result overlap across artifacts must not be counted twice.
- Runtime records include instrumentation and are not evidence of a matched-cost algorithm ranking.

## Included and Omitted

- [Seed metrics](analysis/seed_metrics.csv), [aggregates](analysis/aggregate_metrics.csv), [paired intervals](analysis/paired_differences.csv), [curves](analysis/learning_curves.csv), [report](analysis/report.json).
- data/historical_1_5 contains retained evaluation curves, not reconstructed historical execution configs.
- data/extension_6_10 contains every completed run's evaluation curve, config, summary, simulator metadata and actual launch record.
- source_snapshot contains byte-identical executed trainer sources for included methods. Original launchers are historical presets; launch_record.json records the actual extension command. This is not a standalone training package.
- [Provenance](provenance.json) records original file hashes and export transformations. [Queue snapshot](provenance/queue_snapshot.json) records status only for unfinished runs; their result directories are omitted.
- Checkpoints, replay buffers, raw training logs, caches, locks and unfinished output files are excluded. Workspace prefixes are redacted in exported JSON; unavailable optional diagnostics become null, never valid performance values.

## Verification

From the repository root with Python 3.12 and NumPy:

```bash
python3 ICML_27/tools/ddpg_completed_extension_results.py verify
```

Read-only CPU verification recomputes all seed metrics and complete-cohort intervals, checks configs/completion/source hashes, and verifies every manifest entry. It does not import trainers or modify the running queue.

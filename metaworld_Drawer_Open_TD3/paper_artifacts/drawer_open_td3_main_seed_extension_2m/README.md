# DrawerOpen TD3: Completed Main-Paper Seed Extensions

Snapshot: **2026-11-07T21:12:17.656999+00:00**. This is not a live dashboard.

This artifact contains **20 completed new runs** and 20 historical seed-level references.
Each new run has 2M environment steps. Only completed runs are exported.
Unfinished methods are listed as status only; their intermediate curves and results are omitted.

## Queue Coverage

| Method | Completed new seeds | State at capture |
| --- | --- | --- |
| HER | 6, 7, 8, 9, 10 | completed |
| SABER (Ours) | 6, 7, 8, 9, 10 | completed |
| HGR | 6, 7, 8, 9, 10 | completed |
| PER-full | 6, 7, 8, 9, 10 | completed |

## historical_1_5

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.126508 +/- 0.282739 | 0.199900 +/- 0.446990 | 0.190000 +/- 0.424853 | 1/5 |
| SABER (Ours) | 0.361834 +/- 0.338296 | 0.598100 +/- 0.545988 | 0.590000 +/- 0.538981 | 3/5 |
| HGR | 0.370528 +/- 0.343037 | 0.597000 +/- 0.544987 | 0.580000 +/- 0.531037 | 3/5 |
| PER-full | 0.406432 +/- 0.374116 | 0.595800 +/- 0.543908 | 0.600000 +/- 0.547723 | 3/5 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_her | +0.235327 | [-0.058191, +0.529129] | 3/5 |
| saber_minus_hgr | -0.008693 | [-0.488769, +0.471382] | 3/5 |
| saber_minus_per | -0.044598 | [-0.100628, -0.002211] | 1/5 |

## extension_6_10

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.148291 +/- 0.331590 | 0.199000 +/- 0.444978 | 0.200000 +/- 0.447214 | 1/5 |
| SABER (Ours) | 0.554623 +/- 0.312782 | 0.796200 +/- 0.445109 | 0.800000 +/- 0.447214 | 4/5 |
| HGR | 0.117035 +/- 0.261558 | 0.198800 +/- 0.444530 | 0.200000 +/- 0.447214 | 1/5 |
| PER-full | 0.383568 +/- 0.356180 | 0.586800 +/- 0.536152 | 0.600000 +/- 0.547723 | 3/5 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_her | +0.406332 | [-0.176231, +0.726131] | 4/5 |
| saber_minus_hgr | +0.437588 | [+0.170101, +0.670603] | 4/5 |
| saber_minus_per | +0.171055 | [-0.246482, +0.588593] | 3/5 |

## pooled_1_10

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.137399 +/- 0.290738 | 0.199450 +/- 0.420479 | 0.195000 +/- 0.411265 | 2/10 |
| SABER (Ours) | 0.458229 +/- 0.323527 | 0.697150 +/- 0.481088 | 0.695000 +/- 0.479844 | 7/10 |
| HGR | 0.243781 +/- 0.317104 | 0.397900 +/- 0.513688 | 0.390000 +/- 0.504315 | 4/10 |
| PER-full | 0.395000 +/- 0.344580 | 0.591300 +/- 0.509180 | 0.600000 +/- 0.516398 | 6/10 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_her | +0.320829 | [+0.016682, +0.602965] | 7/10 |
| saber_minus_hgr | +0.214447 | [-0.097636, +0.501464] | 7/10 |
| saber_minus_per | +0.063229 | [-0.149962, +0.300094] | 4/10 |

## Completed New Seed Results

| Method | Seed | AUC | FQ-AUC | Final success | Hit90 step |
| --- | ---: | ---: | ---: | ---: | ---: |
| HER | 6 | 0.000000 | 0.000000 | 0.00 | Not reached |
| HER | 7 | 0.000000 | 0.000000 | 0.00 | Not reached |
| HER | 8 | 0.741457 | 0.995000 | 1.00 | 500000 |
| HER | 9 | 0.000000 | 0.000000 | 0.00 | Not reached |
| HER | 10 | 0.000000 | 0.000000 | 0.00 | Not reached |
| SABER (Ours) | 6 | 0.764573 | 1.000000 | 1.00 | 490000 |
| SABER (Ours) | 7 | 0.671608 | 0.989000 | 1.00 | 620000 |
| SABER (Ours) | 8 | 0.000000 | 0.000000 | 0.00 | Not reached |
| SABER (Ours) | 9 | 0.664070 | 0.994000 | 1.00 | 660000 |
| SABER (Ours) | 10 | 0.672864 | 0.998000 | 1.00 | 650000 |
| HGR | 6 | 0.584925 | 0.994000 | 1.00 | 900000 |
| HGR | 7 | 0.000000 | 0.000000 | 0.00 | Not reached |
| HGR | 8 | 0.000251 | 0.000000 | 0.00 | Not reached |
| HGR | 9 | 0.000000 | 0.000000 | 0.00 | Not reached |
| HGR | 10 | 0.000000 | 0.000000 | 0.00 | Not reached |
| PER-full | 6 | 0.000000 | 0.000000 | 0.00 | Not reached |
| PER-full | 7 | 0.000754 | 0.000000 | 0.00 | Not reached |
| PER-full | 8 | 0.576884 | 0.997000 | 1.00 | 910000 |
| PER-full | 9 | 0.591960 | 0.941000 | 1.00 | 810000 |
| PER-full | 10 | 0.748241 | 0.996000 | 1.00 | 530000 |

## Protocol and Interpretation

- DrawerOpen-v3 / TD3 / sparse reward v2; UTD=1, HER ratio 0.8, horizon 200, 20 evaluation episodes every 10k steps.
- Every completed run has 2M environment steps and 1,995,001 recorded update iterations. TD3 retains its delayed actor updates.
- Success is eval/env_success_mean. AUC uses all 200 checkpoints from 10k to 2M, divided by 1,990,000; FQ-AUC covers 1.5M..2M, divided by 500,000. No smoothing, interpolation or silently shortened horizon.
- Final success is evaluated at 2M. Hit90 means the first evaluation >=0.90, not sustained success. SD is across seeds with ddof=1.
- Historical seeds 1-5, new seeds 6-10 and pooled seeds 1-10 are separate. Pooling does not establish historical environment identity.
- All completed zero/low-success runs are retained. Zero success is a learning outcome, not an omitted or crashed run.
- No checkpoint continuation or outcome-based exclusion is used. Only methods with all five new seeds complete receive aggregate results.
- Paired contrasts, when available, use 10,000 seed-row bootstrap resamples with default_rng(2027), endpoints 2.5% and 97.5%. These are approximate, pointwise, unadjusted intervals, not simultaneous guarantees.
- Retained source/preset and historical metric parity were audited. Original per-run full configs, execution commits and exact historical dependency locks were not retained; byte-for-byte historical environment identity is not asserted.
- Original internal B0 names may contain ddpg, but the retained agent backend is TD3. Actual launch arguments are recorded per run.
- Throughput records reflect five-worker concurrency, not a dedicated compute-matched comparison.

## Included and Omitted

- [Seed metrics](analysis/seed_metrics.csv), [aggregates](analysis/aggregate_metrics.csv), [paired intervals](analysis/paired_differences.csv), [curves](analysis/learning_curves.csv), [report](analysis/report.json).
- data/historical_1_5 contains original retained evaluation curves only, not reconstructed historical execution configs.
- data/extension_6_10 contains complete evaluation curves, configs, summaries, simulator metadata and actual launch records.
- source_snapshot contains byte-identical executed trainer sources for included methods only. Original launchers are historical presets, not the actual extension commands.
- [Provenance](provenance.json), [source origins](provenance/source_origins.json), [queue snapshot](provenance/queue_snapshot.json) and [protocol](protocol.json) record scope and omissions.
- Workspace prefixes are redacted in JSON; unavailable optional diagnostics become null. Required loss and evaluation values are validated as finite before export.
- Checkpoints, replay buffers, raw training logs, caches and unfinished result directories are omitted. This is a compact result artifact, not a standalone training bundle.

## Verification

From the repository root with Python 3.12 and NumPy:

```bash
python3 ICML_27/tools/td3_seed_extension_results.py verify
```

This read-only CPU check regenerates metrics from every included evaluation point, checks completion/config/source hashes and verifies the manifest. It does not import trainers or modify the live queue.

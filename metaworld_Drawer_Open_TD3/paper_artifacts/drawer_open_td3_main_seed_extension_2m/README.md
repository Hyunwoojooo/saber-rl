# DrawerOpen TD3: Completed Main-Paper Seed Extensions

Snapshot: **2026-11-05T05:09:44.289423+00:00**. This is not a live dashboard.

This artifact contains **5 completed new runs** and 5 historical seed-level references.
Each new run has 2M environment steps. Only completed runs are exported.
Unfinished methods are listed as status only; their intermediate curves and results are omitted.

## Queue Coverage

| Method | Completed new seeds | State at capture |
| --- | --- | --- |
| HER | 6, 7, 8, 9, 10 | completed |
| SABER (Ours) | None | running |
| HGR | None | queued |
| PER-full | None | queued |

## historical_1_5

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.126508 +/- 0.282739 | 0.199900 +/- 0.446990 | 0.190000 +/- 0.424853 | 1/5 |

## extension_6_10

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.148291 +/- 0.331590 | 0.199000 +/- 0.444978 | 0.200000 +/- 0.447214 | 1/5 |

## pooled_1_10

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.137399 +/- 0.290738 | 0.199450 +/- 0.420479 | 0.195000 +/- 0.411265 | 2/10 |

**No new paired comparison is available.** SABER and a comparator must both have all five
extension seeds complete before paired contrasts are reported. The paired CSV contains a header only.
Do not compare ten HER seeds against five historical SABER seeds as a paired ten-seed result.

## Completed New Seed Results

| Method | Seed | AUC | FQ-AUC | Final success | Hit90 step |
| --- | ---: | ---: | ---: | ---: | ---: |
| HER | 6 | 0.000000 | 0.000000 | 0.00 | Not reached |
| HER | 7 | 0.000000 | 0.000000 | 0.00 | Not reached |
| HER | 8 | 0.741457 | 0.995000 | 1.00 | 500000 |
| HER | 9 | 0.000000 | 0.000000 | 0.00 | Not reached |
| HER | 10 | 0.000000 | 0.000000 | 0.00 | Not reached |

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

# DrawerClose TD3: Completed Main-Paper Seed Extensions

Snapshot: **2026-11-25T04:32:17.381043+00:00**. This is not a live dashboard.

This artifact contains **20 completed new runs** and 20 historical seed-level references.
Each new run has 2M environment steps. Only completed runs are exported.
HGR and PER-full seeds 6-10 are newly published. The ten HER/SABER extension runs from the earlier comparison at fb4e55a are reused, not new independent runs.
Seeds 1-5 are the retained main-paper references. There are 40 distinct runs across the four methods and ten matched seeds.
The complete extension includes zero-final-success runs.
[Runtime records](analysis/runtime_metrics.csv) cover all twenty extension runs.

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
| HER | 0.328744 +/- 0.347775 | 0.283800 +/- 0.364561 | 0.350000 +/- 0.479583 | 5/5 |
| SABER (Ours) | 0.548769 +/- 0.421505 | 0.533700 +/- 0.484514 | 0.590000 +/- 0.538981 | 5/5 |
| HGR | 0.601608 +/- 0.305387 | 0.460100 +/- 0.380981 | 0.500000 +/- 0.456892 | 5/5 |
| PER-full | 0.523769 +/- 0.379026 | 0.582600 +/- 0.509836 | 0.600000 +/- 0.547723 | 5/5 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_her | +0.220025 | [-0.215553, +0.655603] | 2/5 |
| saber_minus_hgr | -0.052839 | [-0.469824, +0.254975] | 2/5 |
| saber_minus_per | +0.025000 | [-0.376030, +0.455050] | 2/5 |

## extension_6_10

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.410829 +/- 0.421561 | 0.392800 +/- 0.524506 | 0.350000 +/- 0.487340 | 5/5 |
| SABER (Ours) | 0.245653 +/- 0.373636 | 0.183800 +/- 0.387172 | 0.200000 +/- 0.447214 | 5/5 |
| HGR | 0.526357 +/- 0.352294 | 0.546700 +/- 0.398554 | 0.520000 +/- 0.454973 | 5/5 |
| PER-full | 0.280477 +/- 0.241259 | 0.233300 +/- 0.342449 | 0.370000 +/- 0.411704 | 5/5 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_her | -0.165176 | [-0.515251, +0.030704] | 2/5 |
| saber_minus_hgr | -0.280704 | [-0.708442, +0.218191] | 1/5 |
| saber_minus_per | -0.034824 | [-0.326080, +0.283794] | 1/5 |

## pooled_1_10

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.369786 +/- 0.366892 | 0.338300 +/- 0.429696 | 0.350000 +/- 0.455826 | 10/10 |
| SABER (Ours) | 0.397211 +/- 0.408082 | 0.358750 +/- 0.452733 | 0.395000 +/- 0.510147 | 10/10 |
| HGR | 0.563982 +/- 0.313342 | 0.503400 +/- 0.370393 | 0.510000 +/- 0.429987 | 10/10 |
| PER-full | 0.402123 +/- 0.325823 | 0.407950 +/- 0.448930 | 0.485000 +/- 0.472611 | 10/10 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_her | +0.027425 | [-0.260776, +0.325419] | 4/10 |
| saber_minus_hgr | -0.166771 | [-0.472595, +0.140829] | 3/10 |
| saber_minus_per | -0.004912 | [-0.246408, +0.251466] | 3/10 |

## Completed New Seed Results

| Method | Seed | AUC | FQ-AUC | Final success | Hit90 step |
| --- | ---: | ---: | ---: | ---: | ---: |
| HER | 6 | 0.079899 | 0.000000 | 0.00 | 110000 |
| HER | 7 | 0.078141 | 0.005000 | 0.00 | 90000 |
| HER | 8 | 0.890578 | 0.944000 | 0.75 | 90000 |
| HER | 9 | 0.851759 | 0.990000 | 1.00 | 90000 |
| HER | 10 | 0.153769 | 0.025000 | 0.00 | 110000 |
| SABER (Ours) | 6 | 0.035553 | 0.000000 | 0.00 | 310000 |
| SABER (Ours) | 7 | 0.099749 | 0.000000 | 0.00 | 140000 |
| SABER (Ours) | 8 | 0.033668 | 0.011000 | 0.00 | 80000 |
| SABER (Ours) | 9 | 0.908291 | 0.876000 | 1.00 | 90000 |
| SABER (Ours) | 10 | 0.151005 | 0.032000 | 0.00 | 70000 |
| HGR | 6 | 0.076256 | 0.145500 | 0.00 | 1140000 |
| HGR | 7 | 0.580151 | 0.401500 | 0.30 | 30000 |
| HGR | 8 | 0.847487 | 0.946500 | 1.00 | 30000 |
| HGR | 9 | 0.259799 | 0.244000 | 0.30 | 160000 |
| HGR | 10 | 0.868090 | 0.996000 | 1.00 | 190000 |
| PER-full | 6 | 0.064824 | 0.000000 | 0.00 | 300000 |
| PER-full | 7 | 0.154774 | 0.120000 | 0.40 | 160000 |
| PER-full | 8 | 0.145226 | 0.029000 | 0.00 | 110000 |
| PER-full | 9 | 0.379774 | 0.186000 | 0.45 | 90000 |
| PER-full | 10 | 0.657789 | 0.831500 | 1.00 | 220000 |

## Protocol and Interpretation

- DrawerClose-v3 / TD3 / sparse reward v2; UTD=1, HER ratio 0.8, horizon 200, 20 evaluation episodes every 10k steps.
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
python3 ICML_27/tools/drawerclose_td3_main_results.py verify
```

This read-only CPU check regenerates metrics from every included evaluation point, checks completion/config/source hashes and verifies the manifest. It does not import trainers or modify the live queue.

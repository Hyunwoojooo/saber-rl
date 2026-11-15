# DrawerClose DDPG: Completed HER Seed Extension

Queue snapshot: **2026-11-15T07:28:47.266667+00:00** (not live monitoring).

HER seeds 6-10 completed 2M steps each. This artifact includes these **five new runs**
and five retained HER seed 1-5 evaluation curves. Historical runs are references, not new executions.
**No SABER or other method's training results or paired confidence intervals are included.**

## HER Results

| Cohort | n | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| historical_1_5 | 5 | 0.251005 +/- 0.029646 | 0.250200 +/- 0.093102 | 0.160000 +/- 0.181659 | 5/5 |
| extension_6_10 | 5 | 0.307688 +/- 0.051379 | 0.232400 +/- 0.060551 | 0.600000 +/- 0.413824 | 5/5 |
| pooled_1_10 | 10 | 0.279347 +/- 0.049562 | 0.241300 +/- 0.074633 | 0.380000 +/- 0.380205 | 10/10 |

## New Seeds 6-10

| Seed | AUC | FQ-AUC | Final success | First Hit90 step |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 0.216709 | 0.155500 | 0.05 | 150000 |
| 7 | 0.328894 | 0.196000 | 0.30 | 80000 |
| 8 | 0.341960 | 0.258000 | 0.70 | 60000 |
| 9 | 0.321734 | 0.314500 | 0.95 | 80000 |
| 10 | 0.329146 | 0.238000 | 1.00 | 60000 |

## Scope and Interpretation

These are final 2M HER results, not intermediate checkpoint estimates. They cannot establish a HER-SABER ranking while the matching SABER extension is incomplete.
Hit90 records the first evaluation reaching 90%, not sustained success. Low-final-success runs remain included, without seed exclusions.
The three cohort summaries separate historical 1-5, new 6-10, and pooled 1-10. Cross-period pooling does not prove identical historical execution environments.

## Queue Status at Capture

| Method | Status | Completed new seeds |
| --- | --- | ---: |
| her | completed | 5/5 |
| saber | running | 0/5 |
| hgr | queued | 0/5 |
| per | queued | 0/5 |
| replay_query | queued | 0/5 |
| shuffled_query | queued | 0/5 |
| random_support | queued | 0/5 |
| strict_random | queued | 0/5 |

Other methods appear only as timestamped queue status, not performance data.

## Protocol and Reproducibility

- DrawerClose-v3, DDPG+HER, sparse reward v2, 2M steps, UTD=1, HER ratio 0.8, horizon 200.
- Each new run completed 1,995,001 updates from scratch. No checkpoint continuation or outcome-dependent restarts.
- Environment success is evaluated with 20 episodes every 10k steps. Every curve contains all 200 checkpoints from 10k to 2M.
- Normalized trapezoidal AUC uses denominator 1,990,000. FQ-AUC covers 1.5M-2M with denominator 500,000. Final success is measured at 2M; SD uses ddof=1.
- No smoothing, interpolation, shortened horizons, or incomplete-run aggregation is used.
- Original historical HER execution configs and dependency locks were not retained. New runs use the audited earliest retained canonical source and launcher contract. This is not bitwise verification of historical executions.
- Recorded runtime is descriptive for these executions, not a matched-cost comparison with other algorithms.

## Included Files

- [Seed metrics](analysis/seed_metrics.csv), [cohort aggregates](analysis/aggregate_metrics.csv), [learning curves](analysis/learning_curves.csv), [runtime](analysis/runtime_metrics.csv), [report](analysis/report.json).
- `data/extension_6_10/her/`: evaluations, config, summary, simulator metadata, and actual launch record for each completed run.
- `data/historical_1_5/her/`: original evaluation curves only; no reconstructed config is represented as an original execution record.
- `source_snapshot/her/`: byte-identical executed scientific source. Its original launcher targets historical seeds; actual new-run arguments are in each `launch_record.json`.
- [Provenance](provenance.json), [source origins](provenance/source_origins.json), [environment](provenance/environment.json), [queue snapshot](provenance/queue_snapshot.json), [protocol](protocol.json), and [SHA-256 manifest](MANIFEST.sha256).
- Checkpoints, replay buffers, raw training logs, caches, and unfinished run outputs are omitted. JSON workspace paths are redacted; unavailable optional nonfinite diagnostics become null. Required performance/loss values must remain finite.

## Verification

From the repository root with Python 3.12 and NumPy:

```bash
python3 ICML_27/tools/drawerclose_ddpg_seed_extension_results.py verify
```

Read-only CPU verification checks completed-run contracts, regenerates all tables from the included curves, verifies source hashes and the manifest, and does not import a trainer.

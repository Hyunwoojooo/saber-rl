# DrawerClose DDPG: HER vs SABER Seed Extension

Queue snapshot: **2026-11-18T05:03:23.569242+00:00** (not live monitoring).

Both methods completed seeds 6-10, 2M steps per run: ten extension executions.
This publication adds five SABER runs and reuses the five already-published HER runs.
Historical seeds 1-5 are retained references, not new executions. Across both methods there are 20 distinct runs, not 25.

## Results

| Cohort | Metric | HER mean +/- SD | SABER mean +/- SD | SABER - HER | Paired 95% CI | Positive pairs |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| historical_1_5 | auc | 0.251005 +/- 0.029646 | 0.324422 +/- 0.082453 | +0.073417 | [+0.016181, +0.143191] | 4/5 |
| historical_1_5 | final_quarter_auc | 0.250200 +/- 0.093102 | 0.277400 +/- 0.099717 | +0.027200 | [-0.080600, +0.102300] | 4/5 |
| historical_1_5 | final_success | 0.160000 +/- 0.181659 | 0.280000 +/- 0.251496 | +0.120000 | [-0.080000, +0.320000] | 3/5 |
| extension_6_10 | auc | 0.307688 +/- 0.051379 | 0.304673 +/- 0.034399 | -0.003015 | [-0.055176, +0.053442] | 2/5 |
| extension_6_10 | final_quarter_auc | 0.232400 +/- 0.060551 | 0.329200 +/- 0.047155 | +0.096800 | [+0.040300, +0.157700] | 5/5 |
| extension_6_10 | final_success | 0.600000 +/- 0.413824 | 0.400000 +/- 0.329773 | -0.200000 | [-0.710000, +0.310000] | 2/5 |
| pooled_1_10 | auc | 0.279347 +/- 0.049562 | 0.314548 +/- 0.060463 | +0.035201 | [-0.010289, +0.086697] | 6/10 |
| pooled_1_10 | final_quarter_auc | 0.241300 +/- 0.074633 | 0.303300 +/- 0.078441 | +0.062000 | [-0.001801, +0.115802] | 9/10 |
| pooled_1_10 | final_success | 0.380000 +/- 0.380205 | 0.340000 +/- 0.283627 | -0.040000 | [-0.355000, +0.255000] | 5/10 |

The extension-only mean AUC difference is close to zero. The pooled mean favors SABER, but its paired AUC interval includes zero.
These data do not establish superiority or equivalence. The original five-seed interval must not be presented as the ten-seed result.
FQ-AUC and final success are secondary, unadjusted comparisons. Extension FQ-AUC favors SABER, but the pooled FQ-AUC interval includes zero; final success does not show a clear advantage.

## New Seeds 6-10

| Seed | HER AUC | SABER AUC | SABER - HER | HER final success | SABER final success |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.216709 | 0.311055 | +0.094347 | 0.05 | 0.60 |
| 7 | 0.328894 | 0.323995 | -0.004899 | 0.30 | 0.85 |
| 8 | 0.341960 | 0.253266 | -0.088693 | 0.70 | 0.00 |
| 9 | 0.321734 | 0.343467 | +0.021734 | 0.95 | 0.30 |
| 10 | 0.329146 | 0.291583 | -0.037563 | 1.00 | 0.25 |

All seeds are retained, including SABER seed 8 with zero final success. Hit90 is first attainment, not sustained success; complete seed metrics include it.

## Queue Status at Capture

| Method | Status | Completed new seeds |
| --- | --- | ---: |
| her | completed | 5/5 |
| saber | completed | 5/5 |
| hgr | completed | 5/5 |
| per | completed | 5/5 |
| replay_query | running | 0/5 |
| shuffled_query | queued | 0/5 |
| random_support | queued | 0/5 |
| strict_random | queued | 0/5 |

Only HER and SABER performance is included. Other methods appear as timestamped status only, never performance aggregates.

## Protocol and Limitations

- DrawerClose-v3, DDPG+HER, sparse reward v2, 2M steps, UTD=1, HER ratio 0.8, horizon 200.
- SABER is critic-only, with K=5, N_local_min=4, M_Q_min=4, M_B=16, scan limit 512, lambda_support=1 and batch-mean normalization.
- All ten extension runs started from scratch and completed 1,995,001 updates. No checkpoint continuation or outcome-dependent exclusions.
- Environment success uses 20 episodes every 10k steps. Each curve includes all 200 checkpoints from 10k to 2M.
- Normalized trapezoidal AUC divides by 1,990,000; FQ-AUC covers 1.5M-2M and divides by 500,000. Final success is at 2M; SD uses ddof=1. No smoothing, interpolation or shortened horizons.
- Pair by seed number. Bootstrap paired seed rows 10,000 times with NumPy default_rng(2027); use 2.5%/97.5% percentile endpoints. These approximate intervals are not simultaneous coverage guarantees.
- AUC is primary. Historical 1-5, extension 6-10 and pooled 1-10 are reported separately to expose cohort variation.
- Historical HER/SABER full configs, exact execution commits and dependency locks were not retained. Audited canonical source and launcher contracts are used, not a claim of bitwise historical execution identity.
- This is the primary seed extension, not a diagnostic rerun. Previously published HER data are included for self-contained verification, not additional independent observations.
- Runtime is descriptive for these executions, not proof of matched hardware load or a compute-normalized performance comparison.

## Included Files

- [Seed metrics](analysis/seed_metrics.csv), [aggregates](analysis/aggregate_metrics.csv), [paired intervals](analysis/paired_differences.csv), [curves](analysis/learning_curves.csv), [runtime](analysis/runtime_metrics.csv), [report](analysis/report.json).
- `data/extension_6_10/`: evaluations, configs, summaries, simulator metadata and actual launch records.
- `data/historical_1_5/`: retained evaluation CSVs only; no reconstructed configs presented as original execution records.
- `source_snapshot/`: byte-identical executed scientific source. Original launchers target historical seeds; actual extension commands are in per-run `launch_record.json`.
- [Provenance](provenance.json), [source origins](provenance/source_origins.json), [environment](provenance/environment.json), [queue](provenance/queue_snapshot.json), [protocol](protocol.json), [manifest](MANIFEST.sha256).
- Checkpoints, replay buffers, raw logs, caches and unfinished outputs are omitted. JSON workspace prefixes are redacted; unavailable optional diagnostics become null. Required performance/loss values must be finite.

## Verification

From the repository root with Python 3.12 and NumPy:

```bash
python3 ICML_27/tools/drawerclose_ddpg_her_saber_results.py verify
```

Read-only CPU verification regenerates metrics and nine intervals, checks completion/config/source contracts and the manifest, and never imports a trainer.

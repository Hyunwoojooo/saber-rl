# DrawerOpen DDPG: HER vs SABER Seed Extension

HER and SABER seeds 6-10 are complete: 10 new executions, 2M steps per run, UTD=1.
They are combined with the 10 original main-paper executions (two methods x seeds 1-5).
All outcomes, including HER seed 10 and SABER seed 7 with zero success, are retained.

## Results

| Cohort | HER mean AUC | SABER mean AUC | SABER - HER | Paired 95% CI | Positive pairs |
| --- | ---: | ---: | ---: | --- | ---: |
| historical_1_5 | 0.207312 | 0.753668 | +0.546357 | [+0.339874, +0.741005] | 5/5 |
| extension_6_10 | 0.594824 | 0.550075 | -0.044749 | [-0.478116, +0.456080] | 1/5 |
| pooled_1_10 | 0.401068 | 0.651872 | +0.250804 | [-0.067643, +0.543933] | 6/10 |

The positive pooled sample mean is smaller than the original five-seed estimate.
The extension-only and pooled AUC intervals include zero: superiority, equivalence, and consistent benefit are not established.
The added seeds favor SABER in only one of five pairs. Do not report the original five-seed CI as the ten-seed result.

| New seed | HER AUC | SABER AUC | HER final success | SABER final success |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 0.782663 | 0.765578 | 1.00 | 1.00 |
| 7 | 0.752513 | 0.000000 | 1.00 | 0.00 |
| 8 | 0.752010 | 0.576382 | 1.00 | 1.00 |
| 9 | 0.686935 | 0.570980 | 1.00 | 0.95 |
| 10 | 0.000000 | 0.837437 | 0.00 | 1.00 |

## Remaining Queue Is Not Complete

Timestamped snapshot: 2026-10-26T05:29:00.731792+00:00. This is not live monitoring.

| Condition | Status | Completed seeds | Minimum / maximum step |
| --- | --- | ---: | --- |
| her | completed | 5/5 | 2,000,000 / 2,000,000 |
| saber | completed | 5/5 | 2,000,000 / 2,000,000 |
| hgr | running | 0/5 | 340,200 / 341,000 |
| per | queued | 0/5 | 0 / 0 |
| replay_query | queued | 0/5 | 0 / 0 |
| shuffled_query | queued | 0/5 | 0 / 0 |
| random_support | queued | 0/5 | 0 / 0 |
| strict_random | queued | 0/5 | 0 / 0 |

The full planned extension is eight conditions x five new seeds = 40 new runs.
HGR and the remaining controls are not included in final performance aggregates or comparisons in this artifact.
Their status appears only in [the queue snapshot](provenance/queue_snapshot.json).

## Protocol and Cohort Identity

- DrawerOpen-v3, DDPG, sparse reward v2, 2M steps, HER ratio=0.8, horizon=200, 20 evaluation episodes per checkpoint.
- Every final curve contains exactly 200 evaluations from 10k to 2M. Full AUC is normalized by 1,990,000; FQ-AUC covers 1.5M..2M, divided by 500,000. No smoothing, interpolation or truncated horizon.
- Success is `eval/env_success_mean`; final success is that metric at 2M. Hit90 is the first evaluation >=0.90, not sustained success.
- Each extension run completed 1,995,001 updates. Same-seed pairing is used; bootstrap seed rows are resampled 10,000 times with NumPy default_rng(2027). The 95% percentile endpoints are 2.5% and 97.5%.
- AUC is the primary endpoint. FQ-AUC and final-success intervals are secondary, unadjusted intervals, not simultaneous guarantees.
- Historical 1-5, extension 6-10 and pooled 1-10 are reported separately. Cross-period pooling is not proof of identical historical execution environments.
- Original HER/SABER full configs, exact execution commits and dependency locks were not retained. New runs use the earliest retained canonical trainer and launcher-contract reconstruction, not bitwise-verified historical execution.
- These are the October extension executions, not the post-review assignment experiment or earlier diagnostic executions. Same-number diagnostic reruns are not additional independent seeds and must not be counted again.
- All completed failures-to-learn remain included; no checkpoint continuation, outcome-based restart, or significance-based stopping was used.

## Included Files

- [Seed metrics](analysis/seed_metrics.csv), [aggregate metrics](analysis/aggregate_metrics.csv), [paired intervals](analysis/paired_differences.csv), [full curves](analysis/learning_curves.csv), [report](analysis/report.json).
- `data/historical_1_5/`: retained evaluation CSVs only. No reconstructed configuration is presented as an original execution config.
- `data/extension_6_10/`: evaluation CSVs plus recorded configs, summaries, simulator descriptions and actual launch arguments.
- `source_snapshot/`: byte-identical trainers for the two completed conditions. `original_launcher.sh` is the historical launcher, not the executed seed-extension command; use per-run `launch_record.json` to inspect actual arguments.
- [Provenance](provenance.json), [source origins](provenance/source_origins.json), [recorded new environment](provenance/environment.json), [full queue protocol](protocol.json).
- JSON workspace prefixes are redacted; nonfinite unavailable diagnostics become null. Required performance/loss values must remain finite. Original hashes and export transformations are recorded.
- Checkpoints, replay buffers, raw training logs, caches, process locks and unfinished training outputs are omitted. This is a performance-result artifact, not a standalone training bundle.

## Verification

From the repository root with Python 3.12 and NumPy (recorded version 2.3.5):

```bash
python3 ICML_27/tools/ddpg_seed_extension_results.py verify
```

Verification is read-only and CPU-only; no trainer is imported. It recomputes all metrics and nine intervals from the included curves, checks completion/configs/source hashes, and verifies the manifest.

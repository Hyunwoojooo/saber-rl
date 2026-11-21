# DrawerClose DDPG: Random-Support Seed Extension

Snapshot: **2026-11-21T05:53:04.177211+00:00**. This is not a live dashboard.

This artifact contains **10 completed seed-extension runs** and 10 historical seed-level references. Each new run has 2M steps.
Five Random-Support runs are newly published. Five previously published SABER extension runs are reused for comparison, not additional independent observations.
There are 20 distinct runs: two methods x ten seeds. Historical seeds 1-5 are reused, not rerun. All earlier artifacts remain unchanged.
Only Random-Support and SABER results are included. Other methods appear as timestamped queue status only.
Random-Support computes the SABER support-positive count on its own batch and replay state, then chooses that many positions uniformly without replacement across the full minibatch. This matches the weight distribution to its own counterfactual, not eligibility or counts across separate online runs.

## Queue Coverage

| Method | Included extension seeds | State at capture |
| --- | --- | --- |
| HER | None | completed |
| SABER (Ours) | 6, 7, 8, 9, 10 | completed |
| HGR | None | completed |
| PER | None | completed |
| Replay-Action Query | None | completed |
| Shuffled-Policy Query | None | completed |
| Random-Support | 6, 7, 8, 9, 10 | completed |
| Context-Eligible Count-Matched | None | running |

## historical_1_5

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| SABER (Ours) | 0.324422 +/- 0.082453 | 0.277400 +/- 0.099717 | 0.280000 +/- 0.251496 | 5/5 |
| Random-Support | 0.276683 +/- 0.049642 | 0.326000 +/- 0.134135 | 0.240000 +/- 0.429244 | 5/5 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_random_support | +0.047739 | [-0.004070, +0.114724] | 3/5 |

## extension_6_10

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| SABER (Ours) | 0.304673 +/- 0.034399 | 0.329200 +/- 0.047155 | 0.400000 +/- 0.329773 | 5/5 |
| Random-Support | 0.287337 +/- 0.057902 | 0.277300 +/- 0.064302 | 0.040000 +/- 0.054772 | 5/5 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_random_support | +0.017337 | [-0.027035, +0.051357] | 4/5 |

## pooled_1_10

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| SABER (Ours) | 0.314548 +/- 0.060463 | 0.303300 +/- 0.078441 | 0.340000 +/- 0.283627 | 10/10 |
| Random-Support | 0.282010 +/- 0.051155 | 0.301650 +/- 0.102435 | 0.140000 +/- 0.307137 | 10/10 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_random_support | +0.032538 | [-0.003040, +0.072075] | 7/10 |

## Completed New Seed Results

Both included methods have all seeds 6-10 at 2M; these are not intermediate checkpoints.

| Method | Seed | AUC | FQ-AUC | Final success | Hit90 step |
| --- | ---: | ---: | ---: | ---: | ---: |
| SABER (Ours) | 6 | 0.311055 | 0.320000 | 0.60 | 70000 |
| SABER (Ours) | 7 | 0.323995 | 0.388500 | 0.85 | 60000 |
| SABER (Ours) | 8 | 0.253266 | 0.286000 | 0.00 | 90000 |
| SABER (Ours) | 9 | 0.343467 | 0.367000 | 0.30 | 370000 |
| SABER (Ours) | 10 | 0.291583 | 0.284500 | 0.25 | 40000 |
| Random-Support | 6 | 0.304774 | 0.319500 | 0.00 | 120000 |
| Random-Support | 7 | 0.265327 | 0.275000 | 0.10 | 40000 |
| Random-Support | 8 | 0.202261 | 0.169000 | 0.10 | 40000 |
| Random-Support | 9 | 0.306030 | 0.293000 | 0.00 | 110000 |
| Random-Support | 10 | 0.358291 | 0.330000 | 0.00 | 50000 |

## Pooled Paired Metrics (Seeds 1-10)

All differences are SABER minus Random-Support.

| Metric | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| auc | +0.032538 | [-0.003040, +0.072075] | 7/10 |
| final_quarter_auc | +0.001650 | [-0.072500, +0.064500] | 6/10 |
| final_success | +0.200000 | [-0.080000, +0.435000] | 7/10 |

## Interpretation and Protocol

- SABER has higher mean AUC in both the added-seed and pooled cohorts, but both AUC intervals include zero. Superiority and equivalence are not established.
- Both pooled secondary intervals include zero. The larger added-seed final-success difference is secondary and does not override the primary AUC uncertainty.
- The historical, added-seed and pooled estimates are reported separately. Their differing directions and failure-to-learn outcomes must remain visible.
- Both included methods have five complete extension runs. Other methods are excluded even if their queue status later becomes completed.
- DrawerClose-v3 / DDPG / sparse reward v2; 2M steps, UTD=1, 1,995,001 optimizer updates, HER ratio 0.8, horizon 200, 20 evaluation episodes every 10k steps.
- AUC uses exactly 200 evaluation points from 10k to 2M, divided by 1,990,000. FQ-AUC covers 1.5M..2M, divided by 500,000. Success is eval/env_success_mean. No smoothing, interpolation or silently shortened horizons.
- Final success is evaluated at 2M. Hit90 is the first evaluation >=0.90, not sustained task mastery. SD is across seeds with ddof=1.
- Paired percentile bootstrap resamples seed rows 10,000 times using NumPy default_rng(2027); endpoints are 2.5% and 97.5%. These are approximate, pointwise, unadjusted intervals, not simultaneous guarantees. AUC is primary; other metrics are secondary.
- Every completed zero/low-success run is retained. New runs began from scratch; no checkpoint continuation or outcome-based exclusions.
- Random-Support uses method=b3_support_random_runtime_opt with the current-policy support query. Both methods retain critic-only weighting, lambda_support=1, action bins=5, minimum local count=4 and deterministic tail-512 scanning, with unchanged uniform replay and HER relabeling.
- Historical exact 3,125-assignment mechanism-control bootstrap intervals are not overwritten. All intervals in this artifact, including historical-only estimates, use the declared 10,000-resample Monte Carlo procedure.
- The original full configs for both methods and exact historical environment locks were not retained. The retained launchers and source contracts were audited. New configs follow frozen-source/launcher contracts; bitwise historical environment identity is not asserted.
- Earlier diagnostic and assignment executions are not extra independent seeds in these cohorts. Source/result overlap across artifacts must not be counted twice.
- Runtime records include instrumentation and are not evidence of a matched-cost algorithm ranking.

## Included and Omitted

- [Seed metrics](analysis/seed_metrics.csv), [aggregates](analysis/aggregate_metrics.csv), [paired intervals](analysis/paired_differences.csv), [curves](analysis/learning_curves.csv), [runtime](analysis/runtime_metrics.csv), [report](analysis/report.json).
- data/historical_1_5 contains retained evaluation curves, not reconstructed historical execution configs.
- data/extension_6_10 contains every completed run's evaluation curve, config, summary, simulator metadata and actual launch record.
- source_snapshot contains byte-identical executed trainer sources for included methods. Original launchers are historical presets; launch_record.json records the actual extension command. This is not a standalone training package.
- [Provenance](provenance.json) records original file hashes and transformations. [Queue snapshot](provenance/queue_snapshot.json) records status only for non-included methods; their result directories are omitted. [Manifest](MANIFEST.sha256) covers every artifact file.
- Checkpoints, replay buffers, raw training logs, caches, locks and unfinished output files are excluded. Workspace prefixes are redacted in exported JSON; unavailable optional diagnostics become null, never valid performance values.

## Verification

From the repository root with Python 3.12 and NumPy:

```bash
python3 ICML_27/tools/drawerclose_ddpg_random_support_results.py verify
```

Read-only CPU verification recomputes all seed metrics and complete-cohort intervals, checks configs/completion/source hashes, and verifies every manifest entry. It does not import trainers or modify the training queue.

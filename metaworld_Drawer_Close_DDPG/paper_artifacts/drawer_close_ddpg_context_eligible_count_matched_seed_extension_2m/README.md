# DrawerClose DDPG: Context-Eligible Count-Matched Seed Extension

Snapshot: **2026-11-21T09:59:37.307814+00:00**. This is not a live dashboard.

This artifact contains **10 completed seed-extension runs** and 10 historical seed-level references. Each new run has 2M steps.
Five Context-Eligible Count-Matched runs are newly published. Five previously published SABER extension runs are reused for comparison, not additional independent observations.
There are 20 distinct runs: two methods x ten seeds. Historical seeds 1-5 are reused, not rerun. All earlier artifacts remain unchanged.
Only Context-Eligible Count-Matched and SABER results are included. Other methods appear as timestamped queue status only.
This control derives the high-weight count from the current-policy SABER mask on its own batch and buffer, then assigns that many high weights uniformly without replacement within the context-eligible positions. It preserves context eligibility and weight distribution within that batch, not buffer contents or counts across separate online runs.

## Queue Coverage

| Method | Included extension seeds | State at capture |
| --- | --- | --- |
| HER | None | completed |
| SABER (Ours) | 6, 7, 8, 9, 10 | completed |
| HGR | None | completed |
| PER | None | completed |
| Replay-Action Query | None | completed |
| Shuffled-Policy Query | None | completed |
| Random-Support | None | completed |
| Context-Eligible Count-Matched | 6, 7, 8, 9, 10 | completed |

## historical_1_5

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| SABER (Ours) | 0.324422 +/- 0.082453 | 0.277400 +/- 0.099717 | 0.280000 +/- 0.251496 | 5/5 |
| Context-Eligible Count-Matched | 0.294799 +/- 0.063108 | 0.324700 +/- 0.110924 | 0.330000 +/- 0.460435 | 5/5 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_strict_random | +0.029623 | [-0.033191, +0.092437] | 3/5 |

## extension_6_10

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| SABER (Ours) | 0.304673 +/- 0.034399 | 0.329200 +/- 0.047155 | 0.400000 +/- 0.329773 | 5/5 |
| Context-Eligible Count-Matched | 0.297010 +/- 0.039716 | 0.306800 +/- 0.072484 | 0.450000 +/- 0.331662 | 5/5 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_strict_random | +0.007663 | [-0.022764, +0.037362] | 3/5 |

## pooled_1_10

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| SABER (Ours) | 0.314548 +/- 0.060463 | 0.303300 +/- 0.078441 | 0.340000 +/- 0.283627 | 10/10 |
| Context-Eligible Count-Matched | 0.295905 +/- 0.049723 | 0.315750 +/- 0.088840 | 0.390000 +/- 0.383551 | 10/10 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_strict_random | +0.018643 | [-0.014874, +0.056295] | 6/10 |

## Completed New Seed Results

Both included methods have all seeds 6-10 at 2M; these are not intermediate checkpoints.

| Method | Seed | AUC | FQ-AUC | Final success | Hit90 step |
| --- | ---: | ---: | ---: | ---: | ---: |
| SABER (Ours) | 6 | 0.311055 | 0.320000 | 0.60 | 70000 |
| SABER (Ours) | 7 | 0.323995 | 0.388500 | 0.85 | 60000 |
| SABER (Ours) | 8 | 0.253266 | 0.286000 | 0.00 | 90000 |
| SABER (Ours) | 9 | 0.343467 | 0.367000 | 0.30 | 370000 |
| SABER (Ours) | 10 | 0.291583 | 0.284500 | 0.25 | 40000 |
| Context-Eligible Count-Matched | 6 | 0.287060 | 0.219500 | 0.35 | 20000 |
| Context-Eligible Count-Matched | 7 | 0.266583 | 0.410500 | 0.20 | 90000 |
| Context-Eligible Count-Matched | 8 | 0.254271 | 0.261500 | 0.20 | 50000 |
| Context-Eligible Count-Matched | 9 | 0.339950 | 0.312000 | 1.00 | 10000 |
| Context-Eligible Count-Matched | 10 | 0.337186 | 0.330500 | 0.50 | 100000 |

## Pooled Paired Metrics (Seeds 1-10)

All differences are SABER minus Context-Eligible Count-Matched.

| Metric | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| auc | +0.018643 | [-0.014874, +0.056295] | 6/10 |
| final_quarter_auc | -0.012450 | [-0.078700, +0.052509] | 4/10 |
| final_success | -0.050000 | [-0.385000, +0.280000] | 5/10 |

## Interpretation and Protocol

- SABER has higher mean AUC in both the added-seed and pooled cohorts, but both AUC intervals include zero. Superiority and equivalence are not established.
- The pooled FQ-AUC and final-success means favor the control, while added-seed FQ-AUC favors SABER. All secondary paired intervals include zero. These results do not isolate an advantage from assigning high weights to the exact action-bin matches.
- The historical, added-seed and pooled estimates are reported separately. Their differing directions and failure-to-learn outcomes must remain visible.
- Both included methods have five complete extension runs. Other methods are excluded even if their queue status later becomes completed.
- DrawerClose-v3 / DDPG / sparse reward v2; 2M steps, UTD=1, 1,995,001 optimizer updates, HER ratio 0.8, horizon 200, 20 evaluation episodes every 10k steps.
- AUC uses exactly 200 evaluation points from 10k to 2M, divided by 1,990,000. FQ-AUC covers 1.5M..2M, divided by 500,000. Success is eval/env_success_mean. No smoothing, interpolation or silently shortened horizons.
- Final success is evaluated at 2M. Hit90 is the first evaluation >=0.90, not sustained task mastery. SD is across seeds with ddof=1.
- Paired percentile bootstrap resamples seed rows 10,000 times using NumPy default_rng(2027); endpoints are 2.5% and 97.5%. These are approximate, pointwise, unadjusted intervals, not simultaneous guarantees. AUC is primary; other metrics are secondary.
- Every completed zero/low-success run is retained. New runs began from scratch; no checkpoint continuation or outcome-based exclusions.
- Context-Eligible Count-Matched uses support_query_mode=current_policy. Its support_assignment_mode is context_eligible_count_matched. Both methods retain critic-only weighting, lambda_support=1, action bins=5, minimum local count=4 and deterministic tail-512 scanning, with unchanged uniform replay and HER relabeling.
- Historical exact 3,125-assignment mechanism-control bootstrap intervals are not overwritten. All intervals in this artifact, including historical-only estimates, use the declared 10,000-resample Monte Carlo procedure.
- The original SABER full configs and exact historical environment locks were not retained. The original Context-Eligible Count-Matched full configs were retained and audited. New configs follow frozen-source/launcher contracts; bitwise historical environment identity is not asserted.
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
python3 ICML_27/tools/drawerclose_ddpg_strict_random_results.py verify
```

Read-only CPU verification recomputes all seed metrics and complete-cohort intervals, checks configs/completion/source hashes, and verifies every manifest entry. It does not import trainers or modify the training queue.

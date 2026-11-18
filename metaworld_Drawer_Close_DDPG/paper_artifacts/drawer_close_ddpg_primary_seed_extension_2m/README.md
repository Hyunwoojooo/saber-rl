# DrawerClose DDPG: HER, SABER, HGR and PER Seed Extensions

Snapshot: **2026-11-18T07:03:54.891845+00:00**. This is not a live dashboard.

This artifact contains **20 completed seed-extension runs** and 20 historical seed-level references. Each new run has 2M steps.
Ten HGR/PER runs are newly published. The earlier HER/SABER artifact's ten extension runs are included again for comparison, not additional independent observations.
There are 40 distinct runs: four methods x ten seeds. The earlier HER-only and HER-SABER artifacts remain unchanged.
Only HER, SABER, HGR and PER results are included. Other methods appear as timestamped queue status only.

## Queue Coverage

| Method | Included extension seeds | State at capture |
| --- | --- | --- |
| HER | 6, 7, 8, 9, 10 | completed |
| SABER (Ours) | 6, 7, 8, 9, 10 | completed |
| HGR | 6, 7, 8, 9, 10 | completed |
| PER | 6, 7, 8, 9, 10 | completed |
| Replay-Action Query | None | running |
| Shuffled-Policy Query | None | queued |
| Random-Support | None | queued |
| Context-Eligible Count-Matched | None | queued |

## historical_1_5

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.251005 +/- 0.029646 | 0.250200 +/- 0.093102 | 0.160000 +/- 0.181659 | 5/5 |
| SABER (Ours) | 0.324422 +/- 0.082453 | 0.277400 +/- 0.099717 | 0.280000 +/- 0.251496 | 5/5 |
| HGR | 0.349196 +/- 0.184181 | 0.225500 +/- 0.051615 | 0.080000 +/- 0.090830 | 5/5 |
| PER | 0.307940 +/- 0.049338 | 0.339600 +/- 0.063478 | 0.460000 +/- 0.362974 | 5/5 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_her | +0.073417 | [+0.016181, +0.143191] | 4/5 |
| saber_minus_hgr | -0.024774 | [-0.225452, +0.131985] | 3/5 |
| saber_minus_per | +0.016482 | [-0.059523, +0.093719] | 2/5 |

## extension_6_10

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.307688 +/- 0.051379 | 0.232400 +/- 0.060551 | 0.600000 +/- 0.413824 | 5/5 |
| SABER (Ours) | 0.304673 +/- 0.034399 | 0.329200 +/- 0.047155 | 0.400000 +/- 0.329773 | 5/5 |
| HGR | 0.291910 +/- 0.038519 | 0.276600 +/- 0.104700 | 0.250000 +/- 0.259808 | 5/5 |
| PER | 0.307563 +/- 0.044337 | 0.331100 +/- 0.086975 | 0.190000 +/- 0.270185 | 5/5 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_her | -0.003015 | [-0.055176, +0.053442] | 2/5 |
| saber_minus_hgr | +0.012764 | [-0.027286, +0.052814] | 3/5 |
| saber_minus_per | -0.002889 | [-0.038090, +0.032312] | 2/5 |

## pooled_1_10

| Method | AUC mean +/- SD | FQ-AUC mean +/- SD | Final success mean +/- SD | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.279347 +/- 0.049562 | 0.241300 +/- 0.074633 | 0.380000 +/- 0.380205 | 10/10 |
| SABER (Ours) | 0.314548 +/- 0.060463 | 0.303300 +/- 0.078441 | 0.340000 +/- 0.283627 | 10/10 |
| HGR | 0.320553 +/- 0.129026 | 0.251050 +/- 0.082350 | 0.165000 +/- 0.204192 | 10/10 |
| PER | 0.307751 +/- 0.044222 | 0.335350 +/- 0.071924 | 0.325000 +/- 0.333542 | 10/10 |

| AUC contrast | Mean difference | Paired 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber_minus_her | +0.035201 | [-0.010289, +0.086697] | 6/10 |
| saber_minus_hgr | -0.006005 | [-0.106470, +0.074774] | 6/10 |
| saber_minus_per | +0.006796 | [-0.036244, +0.049686] | 4/10 |

## Completed New Seed Results

Every primary method includes all seeds 6-10 at 2M; these are not intermediate checkpoints.

| Method | Seed | AUC | FQ-AUC | Final success | Hit90 step |
| --- | ---: | ---: | ---: | ---: | ---: |
| HER | 6 | 0.216709 | 0.155500 | 0.05 | 150000 |
| HER | 7 | 0.328894 | 0.196000 | 0.30 | 80000 |
| HER | 8 | 0.341960 | 0.258000 | 0.70 | 60000 |
| HER | 9 | 0.321734 | 0.314500 | 0.95 | 80000 |
| HER | 10 | 0.329146 | 0.238000 | 1.00 | 60000 |
| SABER (Ours) | 6 | 0.311055 | 0.320000 | 0.60 | 70000 |
| SABER (Ours) | 7 | 0.323995 | 0.388500 | 0.85 | 60000 |
| SABER (Ours) | 8 | 0.253266 | 0.286000 | 0.00 | 90000 |
| SABER (Ours) | 9 | 0.343467 | 0.367000 | 0.30 | 370000 |
| SABER (Ours) | 10 | 0.291583 | 0.284500 | 0.25 | 40000 |
| HGR | 6 | 0.239698 | 0.231500 | 0.00 | 20000 |
| HGR | 7 | 0.305402 | 0.148000 | 0.40 | 20000 |
| HGR | 8 | 0.277387 | 0.236000 | 0.00 | 30000 |
| HGR | 9 | 0.292085 | 0.365500 | 0.25 | 80000 |
| HGR | 10 | 0.344975 | 0.402000 | 0.60 | 110000 |
| PER | 6 | 0.261809 | 0.199000 | 0.20 | 60000 |
| PER | 7 | 0.371859 | 0.421500 | 0.00 | 20000 |
| PER | 8 | 0.273618 | 0.293500 | 0.10 | 100000 |
| PER | 9 | 0.301759 | 0.364500 | 0.00 | 80000 |
| PER | 10 | 0.328769 | 0.377000 | 0.65 | 40000 |

## Interpretation and Protocol

- All three pooled SABER-minus-comparator AUC intervals include zero. Higher sample means do not establish superiority, and overlapping uncertainty does not establish equivalence.
- The historical, added-seed and pooled estimates are reported separately. Their differing directions and failure-to-learn outcomes must remain visible.
- All four primary methods have five complete extension runs. Non-primary control results are excluded even if their queue status later becomes completed.
- DrawerClose-v3 / DDPG / sparse reward v2; 2M steps, UTD=1, 1,995,001 optimizer updates, HER ratio 0.8, horizon 200, 20 evaluation episodes every 10k steps.
- AUC uses exactly 200 evaluation points from 10k to 2M, divided by 1,990,000. FQ-AUC covers 1.5M..2M, divided by 500,000. Success is eval/env_success_mean. No smoothing, interpolation or silently shortened horizons.
- Final success is evaluated at 2M. Hit90 is the first evaluation >=0.90, not sustained task mastery. SD is across seeds with ddof=1.
- Paired percentile bootstrap resamples seed rows 10,000 times using NumPy default_rng(2027); endpoints are 2.5% and 97.5%. These are approximate, pointwise, unadjusted intervals, not simultaneous guarantees. AUC is primary; other metrics are secondary.
- Every completed zero/low-success run is retained. New runs began from scratch; no checkpoint continuation or outcome-based exclusions.
- HGR is the retained episode/transition-future-goal pair-priority implementation with importance weighting. PER is full prioritized replay (ddpg_her_per_full), not the uniform-replay TD-error loss-weighting control. Each method retains its own audited preset.
- Original HER/SABER full configs and exact historical environment locks were not retained. New configs follow audited canonical-source/launcher contracts. Retained baseline configs were checked where available; bitwise historical environment identity is not asserted.
- Earlier diagnostic and assignment executions are not extra independent seeds in these cohorts. Source/result overlap across artifacts must not be counted twice.
- Runtime records include instrumentation and are not evidence of a matched-cost algorithm ranking.

## Included and Omitted

- [Seed metrics](analysis/seed_metrics.csv), [aggregates](analysis/aggregate_metrics.csv), [paired intervals](analysis/paired_differences.csv), [curves](analysis/learning_curves.csv), [runtime](analysis/runtime_metrics.csv), [report](analysis/report.json).
- data/historical_1_5 contains retained evaluation curves, not reconstructed historical execution configs.
- data/extension_6_10 contains every completed run's evaluation curve, config, summary, simulator metadata and actual launch record.
- source_snapshot contains byte-identical executed trainer sources for included methods. Original launchers are historical presets; launch_record.json records the actual extension command. This is not a standalone training package.
- [Provenance](provenance.json) records original file hashes and transformations. [Queue snapshot](provenance/queue_snapshot.json) records status only for non-primary methods; their result directories are omitted. [Manifest](MANIFEST.sha256) covers every artifact file.
- Checkpoints, replay buffers, raw training logs, caches, locks and unfinished output files are excluded. Workspace prefixes are redacted in exported JSON; unavailable optional diagnostics become null, never valid performance values.

## Verification

From the repository root with Python 3.12 and NumPy:

```bash
python3 ICML_27/tools/drawerclose_ddpg_completed_extension_results.py verify
```

Read-only CPU verification recomputes all seed metrics and complete-cohort intervals, checks configs/completion/source hashes, and verifies every manifest entry. It does not import trainers or modify the training queue.

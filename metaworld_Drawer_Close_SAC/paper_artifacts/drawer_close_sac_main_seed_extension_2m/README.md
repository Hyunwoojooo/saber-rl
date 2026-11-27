# DrawerClose SAC: HER and SABER Seed Extensions

Published snapshot: **2026-11-27T04:14:54.245163+00:00**.
SABER seeds 6-10 completed on **2026-11-27T03:09:27.496479+00:00**.

**HER and SABER only:** ten completed new runs (five per method, seeds 6-10),
each at 2M steps, plus ten historical seed-level references (1-5).
New HGR, PER and CPU-pilot results are not included. The original HER run files are unchanged.

## Results

Values are mean +/- sample SD (ddof=1).

| Cohort | Method | Seeds | AUC | FQ-AUC | Final success | Hit90 | Steps/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| historical_1_5 | HER | 1-5 | 0.260628 +/- 0.284573 | 0.247500 +/- 0.395872 | 0.200000 +/- 0.447214 | 5/5 | 37.748 +/- 4.714 |
| historical_1_5 | SABER (Ours) | 1-5 | 0.103317 +/- 0.056453 | 0.017600 +/- 0.009839 | 0.000000 +/- 0.000000 | 5/5 | 21.194 +/- 1.636 |
| extension_6_10 | HER | 6-10 | 0.096633 +/- 0.026787 | 0.031000 +/- 0.027459 | 0.000000 +/- 0.000000 | 5/5 | 30.526 +/- 0.039 |
| extension_6_10 | SABER (Ours) | 6-10 | 0.187990 +/- 0.250501 | 0.218200 +/- 0.423296 | 0.200000 +/- 0.447214 | 5/5 | 21.345 +/- 0.847 |
| pooled_1_10 | HER | 1-10 | 0.178631 +/- 0.209240 | 0.139250 +/- 0.288108 | 0.100000 +/- 0.316228 | 10/10 | 34.137 +/- 4.936 |
| pooled_1_10 | SABER (Ours) | 1-10 | 0.145653 +/- 0.176910 | 0.117900 +/- 0.301424 | 0.100000 +/- 0.316228 | 10/10 | 21.269 +/- 1.231 |

### Paired AUC: SABER Minus HER

| Cohort | Mean difference | Pointwise 95% CI | Nominal 98.333333% CI | Positive pairs |
| --- | ---: | --- | --- | ---: |
| historical_1_5 | -0.157312 | [-0.419070, +0.025025] | [-0.440276, +0.046231] | 1/5 |
| extension_6_10 | +0.091357 | [-0.032312, +0.300201] | [-0.036231, +0.306030] | 2/5 |
| pooled_1_10 | -0.032977 | [-0.212508, +0.130491] | [-0.242155, +0.162193] | 3/10 |

### New Seed-Level Results

| Method | Seed | AUC | FQ-AUC | Final success | Hit90 step | Runtime (h) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HER | 6 | 0.113819 | 0.023000 | 0.000 | 30000 | 18.216 |
| HER | 7 | 0.070352 | 0.001000 | 0.000 | 40000 | 18.177 |
| HER | 8 | 0.079146 | 0.015000 | 0.000 | 90000 | 18.225 |
| HER | 9 | 0.134673 | 0.045000 | 0.000 | 30000 | 18.207 |
| HER | 10 | 0.085176 | 0.071000 | 0.000 | 130000 | 18.174 |
| SABER (Ours) | 6 | 0.071859 | 0.017500 | 0.000 | 60000 | 26.464 |
| SABER (Ours) | 7 | 0.114824 | 0.063000 | 0.000 | 90000 | 25.269 |
| SABER (Ours) | 8 | 0.066332 | 0.028000 | 0.000 | 120000 | 26.232 |
| SABER (Ours) | 9 | 0.634171 | 0.974500 | 1.000 | 50000 | 24.851 |
| SABER (Ours) | 10 | 0.052764 | 0.008000 | 0.000 | 110000 | 27.486 |

## Interpretation and Scope

Across ten paired seeds, SABER AUC is higher in 3, lower in 7, and equal in 0.
SABER-minus-HER mean AUC is +0.091357 in seeds 6-10 and -0.032977 in seeds 1-10.
SABER has higher AUC in 2/5 new pairs. The largest new paired difference is seed 9 (+0.499497), with SABER AUC 0.634171 and final success 1.000.
Both pointwise 95% intervals include zero, so neither establishes an AUC advantage or equivalence.
All zero/low-success seeds are retained. Do not pool the historical, extension and combined summaries as independent cohorts.
The ten historical references overlap the [original five-seed artifact](../drawer_close_sac_her_saber_hgr_per_2m/README.md).
Only five new SABER executions are added by this update; the previously published HER runs are not new executions.

## Protocol, Statistics and Caveats

- FPP-reference automatic-temperature SAC: twin critics, stochastic actor, initial alpha=0.2, target entropy=-4. This is not the legacy fixed-alpha prototype.
- DrawerClose-v3 sparse reward v2, 2M steps, UTD=1, HER ratio=0.8, horizon=200, gamma=0.98, batch=256, replay capacity=300000.
- SABER reweights critic losses only and queries support at the squashed policy mean: K=5, minimum local count=4, tail-512. Actor objectives and Bellman targets remain the inherited SAC definitions.
- Each run has 1,995,001 optimizer updates and 20 evaluation episodes at each of 200 checkpoints.
- Success is eval/env_success_mean, ever-success within an evaluation episode. AUC integrates the exact 10k..2M grid divided by 1,990,000; FQ-AUC integrates 1.5M..2M divided by 500,000. No missing-grid truncation or smoothing.
- Hit90 is the first crossing, not sustained success. Missing Hit90 stays missing.
- Pair by seed, not row order. NumPy default_rng(2027), 10,000 seed-row bootstrap resamples; all metrics share the same sampled indices within each cohort. Percentile 95% endpoints are 0.025 and 0.975.
- Adjusted endpoints are 0.05/6 and 1-0.05/6, retaining the predeclared three-AUC-comparison family (SABER versus HER/HGR/PER), although only the HER contrast is published here. Small-sample coverage is approximate, not an exact simultaneous guarantee.
- FQ-AUC/final-success intervals are secondary descriptive analyses. The three-contrast adjustment does not cover every metric and all three overlapping cohort summaries jointly.
- All new seeds began from scratch without continuation, splicing, replacement or outcome-based exclusion. Historical HER/SABER references are complete fresh recovery runs.
- Recorded scientific configs match across historical and new seeds within each method after documented seed/output metadata is removed. Source files match the already published executed snapshot byte-for-byte.
- The inherited original-goal observation components plus the separately HER-relabeled goal are unchanged in both methods; their causal effect is unresolved.
- Core recorded package versions matched the prelaunch audit. This is not a claim of byte-for-byte historical host identity. Runtime includes evaluation, checkpoints and concurrent workloads, not a controlled wall-clock comparison.
- The fixed-size extension was selected after historical outcomes were observed. The four-method protocol is retained, but HGR/PER extension results are not included here.

## Files and Verification

- [Seed metrics](analysis/seed_metrics.csv), [aggregates](analysis/aggregate_metrics.csv), [evaluation curves](analysis/learning_curves.csv), [report](analysis/report.json).
- [Paired statistics](analysis/paired_statistics.csv) and [seed-level differences](analysis/paired_seed_differences.csv).
- [Provenance](provenance.json), [HER completion](provenance/her_completion.json), [SABER completion](provenance/saber_completion.json), [protocol](protocol.json), [environment](provenance/environment.json).
- [Executed source](../../../ICML_27/experiments/draweropen_drawerclose_sac_her_vs_saber/source/) is unchanged and hash-verified. Configs, summaries, evaluation rows, completion and launch records are included.
- Full-log finite-loss counts are retained for new runs. Workspace prefixes are redacted; optional unavailable JSON diagnostics become null. Evaluation strings are not rounded.
- Checkpoints, replay buffers, full raw logs, new HGR/PER results and CPU-pilot results are omitted. This compact result artifact is not a standalone training bundle.

From the repository root with Python 3.12 and NumPy:

```bash
python3 ICML_27/tools/sac_seed_extension_results.py verify --task drawer_close
```

This read-only CPU check regenerates metrics and paired intervals from 4,000 evaluation points, checks completion/config/source hashes and verifies the manifest. It does not import trainers or change the live queue.

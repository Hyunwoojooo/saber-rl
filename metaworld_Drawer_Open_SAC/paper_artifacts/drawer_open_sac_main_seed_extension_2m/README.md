# DrawerOpen SAC: Completed HER, SABER, HGR and PER Seed Extensions

Published snapshot: **2026-11-14T07:50:37.262406+00:00**.
PER seeds 6-10 and the four-method queue completed on **2026-11-13T22:25:22.945512+00:00**.

**All twenty new runs are complete:** HER, SABER, HGR and PER each have seeds 6-10 at 2M steps,
plus twenty retained historical seed-level references (1-5). No incomplete or selectively chosen seeds are included.
This update adds only five new PER executions. Previously published HER/SABER/HGR run files are unchanged.

## Results

Values are mean +/- sample SD (ddof=1).

| Cohort | Method | Seeds | AUC | FQ-AUC | Final success | Hit90 | Steps/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| historical_1_5 | HER | 1-5 | 0.447035 +/- 0.410848 | 0.600800 +/- 0.542523 | 0.600000 +/- 0.547723 | 3/5 | 37.924 +/- 4.007 |
| historical_1_5 | SABER (Ours) | 1-5 | 0.448492 +/- 0.410829 | 0.599200 +/- 0.546540 | 0.600000 +/- 0.547723 | 3/5 | 20.158 +/- 2.826 |
| historical_1_5 | HGR | 1-5 | 0.706583 +/- 0.065872 | 0.995200 +/- 0.006535 | 1.000000 +/- 0.000000 | 5/5 | 11.782 +/- 0.362 |
| historical_1_5 | PER (full replay) | 1-5 | 0.797588 +/- 0.072233 | 0.984000 +/- 0.014916 | 1.000000 +/- 0.000000 | 5/5 | 39.485 +/- 3.238 |
| extension_6_10 | HER | 6-10 | 0.526332 +/- 0.383685 | 0.777200 +/- 0.437263 | 0.800000 +/- 0.447214 | 4/5 | 30.580 +/- 0.031 |
| extension_6_10 | SABER (Ours) | 6-10 | 0.541608 +/- 0.334595 | 0.796600 +/- 0.445340 | 0.800000 +/- 0.447214 | 4/5 | 20.537 +/- 0.710 |
| extension_6_10 | HGR | 6-10 | 0.675829 +/- 0.099596 | 0.999200 +/- 0.001789 | 1.000000 +/- 0.000000 | 5/5 | 11.955 +/- 0.008 |
| extension_6_10 | PER (full replay) | 6-10 | 0.737286 +/- 0.153576 | 0.995400 +/- 0.005814 | 1.000000 +/- 0.000000 | 5/5 | 28.079 +/- 0.055 |
| pooled_1_10 | HER | 1-10 | 0.486683 +/- 0.377089 | 0.689000 +/- 0.473746 | 0.700000 +/- 0.483046 | 7/10 | 34.252 +/- 4.703 |
| pooled_1_10 | SABER (Ours) | 1-10 | 0.495050 +/- 0.356622 | 0.697900 +/- 0.481381 | 0.700000 +/- 0.483046 | 7/10 | 20.348 +/- 1.953 |
| pooled_1_10 | HGR | 1-10 | 0.691206 +/- 0.081239 | 0.997200 +/- 0.004984 | 1.000000 +/- 0.000000 | 10/10 | 11.869 +/- 0.258 |
| pooled_1_10 | PER (full replay) | 1-10 | 0.767437 +/- 0.117522 | 0.989700 +/- 0.012248 | 1.000000 +/- 0.000000 | 10/10 | 33.782 +/- 6.387 |

### Paired AUC: SABER Minus HER

| Cohort | Mean difference | Pointwise 95% CI | Nominal 98.333333% CI | Positive pairs |
| --- | ---: | --- | --- | ---: |
| historical_1_5 | +0.001457 | [-0.046633, +0.041283] | [-0.056784, +0.043568] | 2/5 |
| extension_6_10 | +0.015276 | [-0.108945, +0.145930] | [-0.133417, +0.153819] | 2/5 |
| pooled_1_10 | +0.008367 | [-0.059777, +0.076912] | [-0.075049, +0.094180] | 4/10 |

### Paired AUC: SABER Minus HGR

| Cohort | Mean difference | Pointwise 95% CI | Nominal 98.333333% CI | Positive pairs |
| --- | ---: | --- | --- | ---: |
| historical_1_5 | -0.258090 | [-0.596533, +0.065075] | [-0.715980, +0.068693] | 3/5 |
| extension_6_10 | -0.134221 | [-0.366131, +0.027990] | [-0.408442, +0.040151] | 2/5 |
| pooled_1_10 | -0.196156 | [-0.413446, -0.011833] | [-0.459733, +0.027149] | 5/10 |

### Paired AUC: SABER Minus PER

| Cohort | Mean difference | Pointwise 95% CI | Nominal 98.333333% CI | Positive pairs |
| --- | ---: | --- | --- | ---: |
| historical_1_5 | -0.349095 | [-0.674724, -0.023467] | [-0.803266, -0.006985] | 1/5 |
| extension_6_10 | -0.195678 | [-0.470151, +0.032060] | [-0.546432, +0.070201] | 1/5 |
| pooled_1_10 | -0.272387 | [-0.498065, -0.068810] | [-0.548727, -0.029166] | 2/10 |

### New Seed-Level Results

| Method | Seed | AUC | FQ-AUC | Final success | Hit90 step | Runtime (h) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HER | 6 | 0.773869 | 1.000000 | 1.000 | 500000 | 18.191 |
| HER | 7 | 0.757035 | 1.000000 | 1.000 | 560000 | 18.182 |
| HER | 8 | 0.000000 | 0.000000 | 0.000 | Not reached | 18.153 |
| HER | 9 | 0.863568 | 1.000000 | 1.000 | 270000 | 18.163 |
| HER | 10 | 0.237186 | 0.886000 | 1.000 | 1500000 | 18.148 |
| SABER (Ours) | 6 | 0.744472 | 1.000000 | 1.000 | 520000 | 26.924 |
| SABER (Ours) | 7 | 0.850000 | 1.000000 | 1.000 | 320000 | 25.618 |
| SABER (Ours) | 8 | 0.000000 | 0.000000 | 0.000 | Not reached | 27.835 |
| SABER (Ours) | 9 | 0.651005 | 0.988000 | 1.000 | 710000 | 27.157 |
| SABER (Ours) | 10 | 0.462563 | 0.995000 | 1.000 | 1110000 | 27.848 |
| HGR | 6 | 0.672111 | 1.000000 | 1.000 | 700000 | 46.477 |
| HGR | 7 | 0.838442 | 0.996000 | 1.000 | 300000 | 46.508 |
| HGR | 8 | 0.587940 | 1.000000 | 1.000 | 850000 | 46.482 |
| HGR | 9 | 0.678894 | 1.000000 | 1.000 | 640000 | 46.454 |
| HGR | 10 | 0.601759 | 1.000000 | 1.000 | 810000 | 46.428 |
| PER (full replay) | 6 | 0.843467 | 1.000000 | 1.000 | 370000 | 19.798 |
| PER (full replay) | 7 | 0.883166 | 1.000000 | 1.000 | 200000 | 19.826 |
| PER (full replay) | 8 | 0.717588 | 0.986000 | 1.000 | 560000 | 19.764 |
| PER (full replay) | 9 | 0.489950 | 0.994000 | 1.000 | 1080000 | 19.728 |
| PER (full replay) | 10 | 0.752261 | 0.997000 | 1.000 | 470000 | 19.809 |

## Interpretation and Scope

Across ten paired seeds, SABER AUC is higher in 4, lower in 5, and equal in 1. This sentence compares SABER with HER.
SABER has a slightly higher mean AUC, but both the pointwise and adjusted intervals include zero.
These data establish neither superiority nor equivalence. They do not demonstrate a causal support mechanism.
HER and SABER have mean final success 0.700 and Hit90 coverage 7/10. The final-zero seeds are 2, 3 and 8 in both methods.
All zero/low-success seeds are retained. Do not pool the historical, extension and combined summaries as independent cohorts.
The twenty historical references overlap the [original five-seed artifact](../drawer_open_sac_her_saber_hgr_per_2m/README.md).
Only five new PER executions are added by this update; HER/SABER/HGR results are preserved, not additional executions.

HGR pooled AUC is 0.691206 +/- 0.081239, with final success 1.0 in all ten runs.
SABER-minus-HGR AUC is -0.196156. Its pointwise 95% interval is negative,
but the predeclared three-comparison adjusted interval includes zero. The mean difference and lower HGR seed variability
favor HGR in this DrawerOpen-SAC cohort, without an adjusted superiority or causal mechanism claim.
SABER AUC is higher in five paired seeds and lower in five; its failed seeds contribute large negative differences.
This is not evidence of a code defect or a general explanation of SAC behavior. Such causes require separate diagnostics.

PER pooled AUC is 0.767437 +/- 0.117522, with final success 1.0 in all ten runs.
SABER-minus-PER AUC is -0.272387; both its pointwise and three-comparison-adjusted intervals are below zero.
This supports lower SABER AUC than PER in this DrawerOpen-SAC cohort under the specified bootstrap analysis, not a universal ranking.
SABER has higher AUC in 2/10 paired seeds. All runs, including the shared HER/SABER final-zero seeds, remain in the analysis.
This comparison does not identify why the methods differ or establish that SAC and HER are inherently incompatible.

## Protocol, Statistics and Caveats

- FPP-reference automatic-temperature SAC: twin critics, stochastic actor, initial alpha=0.2, target entropy=-4. This is not the legacy fixed-alpha prototype.
- DrawerOpen-v3 sparse reward v2, 2M steps, UTD=1, HER ratio=0.8, horizon=200, gamma=0.98, batch=256, replay capacity=300000.
- SABER reweights critic losses only and queries support at the squashed policy mean: K=5, minimum local count=4, tail-512. Actor objectives and Bellman targets remain the inherited SAC definitions.
- Each run has 1,995,001 optimizer updates and 20 evaluation episodes at each of 200 checkpoints.
- Success is eval/env_success_mean, ever-success within an evaluation episode. AUC integrates the exact 10k..2M grid divided by 1,990,000; FQ-AUC integrates 1.5M..2M divided by 500,000. No missing-grid truncation or smoothing.
- Hit90 is the first crossing, not sustained success. Missing Hit90 stays missing.
- Pair by seed, not row order. NumPy default_rng(2027), 10,000 seed-row bootstrap resamples; all metrics share the same sampled indices within each cohort. Percentile 95% endpoints are 0.025 and 0.975.
- Adjusted endpoints are 0.05/6 and 1-0.05/6, retaining the predeclared three-AUC-comparison family (SABER versus HER/HGR/PER), with all three contrasts now published. Small-sample coverage is approximate, not an exact simultaneous guarantee.
- FQ-AUC/final-success intervals are secondary descriptive analyses. The three-contrast adjustment does not cover every metric and all three overlapping cohort summaries jointly.
- All new seeds began from scratch without continuation, splicing, replacement or outcome-based exclusion. Historical HER seeds 1-3 were complete pre-outage runs; the other retained HER/SABER references are complete fresh recovery runs. Historical HGR/PER references are completed original five-seed runs.
- Recorded scientific configs match across historical and new seeds within each method after documented seed/output metadata is removed. Source files match the already published executed snapshot byte-for-byte.
- The inherited original-goal observation components plus the separately HER-relabeled goal are unchanged in all four methods; their causal effect is unresolved.
- Core recorded package versions matched the prelaunch audit. This is not a claim of byte-for-byte historical host identity. Runtime includes evaluation, checkpoints and concurrent workloads, not a controlled wall-clock comparison.
- The fixed-size extension was selected after historical outcomes were observed. All four planned methods and all seeds are now included; historical, extension and pooled analyses remain separate.
- HGR uses hierarchical episode/goal priority exponents 0.6/0.6, epsilon=1e-6 and beta annealing from 0.4 to 1.0 over 1,995,001 updates, with SABER support weighting disabled (lambda_support=0). See the exact recorded configs and unchanged HGR/PER source.
- PER uses full-replay priority sampling and critic importance weights, with SABER support weighting disabled. Exact PER exponents, beta schedule and priority handling are retained in every config and in the unchanged HGR/PER source.

## Files and Verification

- [Seed metrics](analysis/seed_metrics.csv), [aggregates](analysis/aggregate_metrics.csv), [evaluation curves](analysis/learning_curves.csv), [report](analysis/report.json).
- [Paired statistics](analysis/paired_statistics.csv) and [seed-level differences](analysis/paired_seed_differences.csv).
- [Provenance](provenance.json), [HER completion](provenance/her_completion.json), [SABER completion](provenance/saber_completion.json), [HGR completion](provenance/hgr_completion.json), [PER completion](provenance/per_completion.json), [protocol](protocol.json), [environment](provenance/environment.json).
- [Executed source](../../../ICML_27/experiments/draweropen_drawerclose_sac_her_vs_saber/source/) is unchanged and hash-verified. Configs, summaries, evaluation rows, completion and launch records are included.
- Full-log finite-loss counts are retained for new runs. Workspace prefixes are redacted; optional unavailable JSON diagnostics become null. Evaluation strings are not rounded.
- Checkpoints, replay buffers, full raw logs and CPU-pilot results are omitted. This compact result artifact is not a standalone training bundle.
- [Executed HGR/PER source](../../../ICML_27/experiments/draweropen_drawerclose_sac_hgr_per/source/) is also unchanged and hash-verified against the extension snapshot.

From the repository root with Python 3.12 and NumPy:

```bash
python3 ICML_27/tools/sac_seed_extension_results.py verify
```

This read-only CPU check regenerates metrics and paired intervals from 8,000 evaluation points, checks completion/config/source hashes and verifies the manifest. It does not import trainers or change the live queue.

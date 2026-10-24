# DrawerClose SAC: HER, SABER, HGR and PER

Completed post-AAAI cohort: four methods x seeds 1-5 x 2M steps = 20 unique runs.
All seeds, including zero-success and collapsed runs, are included. This is not a 10-seed cohort.

## Results

Values are mean +/- sample SD (ddof=1).

| Method | Final success | AUC | FQ-AUC | Hit90 coverage | Steps/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| HER | 0.200 +/- 0.447 | 0.261 +/- 0.285 | 0.247 +/- 0.396 | 5/5 | 37.748 +/- 4.714 |
| SABER (Ours) | 0.000 +/- 0.000 | 0.103 +/- 0.056 | 0.018 +/- 0.010 | 5/5 | 21.194 +/- 1.636 |
| HGR | 0.030 +/- 0.067 | 0.212 +/- 0.131 | 0.062 +/- 0.124 | 5/5 | 11.791 +/- 0.385 |
| PER (full replay) | 0.550 +/- 0.505 | 0.487 +/- 0.304 | 0.591 +/- 0.499 | 5/5 | 39.366 +/- 3.201 |

| Paired AUC contrast | Mean difference | 95% CI | Positive pairs |
| --- | ---: | --- | ---: |
| saber-hgr | -0.108467 | [-0.234950, +0.004271] | 1/5 |
| saber-per | -0.383995 | [-0.623945, -0.144045] | 0/5 |
| saber-her | -0.157312 | [-0.419070, +0.025025] | 1/5 |

## Interpretation

SABER has a lower sample mean AUC than all three controls and zero final success in every seed.
All methods reach Hit90 at least once in each seed, but this does not imply sustained success:
the low final-quarter AUC and final success expose later collapse, particularly for SABER.
SABER-minus-PER AUC is negative with a 95% interval below zero in this task.
Intervals crossing zero establish neither superiority nor equivalence. Results are limited to this cohort.

## Executed Protocol

- FPP-reference SAC: stochastic Gaussian actor, twin critics, automatic temperature, initial alpha=0.2, target entropy=-4.
- MetaWorld drawer v3, sparse reward version v2; 2M environment steps, UTD=1, HER ratio=0.8, horizon=200.
- SABER applies support weights only to critic losses and queries support at the squashed policy mean.
- HGR is the hierarchical prioritized-goal baseline; PER is full prioritized replay with IS critic weights, not loss-only TD weighting.
- Each run has 1,995,001 updates and 20 evaluation episodes at each of 200 checkpoints.
- AUC integrates exactly 10k..2M, divided by 1,990,000; FQ-AUC integrates 1.5M..2M, divided by 500,000. No smoothing or missing-grid truncation.
- Success is `eval/env_success_mean`: ever-success within an evaluation episode. Final success means this metric at 2M, not final-state success.
- Hit90 is the first evaluation with success >=0.90, not a stability criterion; a blank seed-level Hit90 means not reached.
- Paired AUC bootstrap uses seed rows, 10,000 resamples and NumPy default_rng(2027). Quantiles: 95%=(2.5,97.5), 97.5%=(1.25,98.75), 98.75%=(0.625,99.375).
- The two-task SABER/HER family uses 97.5% per contrast; the four SABER/HGR and SABER/PER contrasts use 98.75%. These are nominal Bonferroni adjustments with approximate small-sample bootstrap coverage.
- HGR/PER are an exploratory extension after the HER/SABER outcomes were observed. Do not describe all four arms as one preregistered experiment.
- Steps/s and runtime are recorded whole-run measurements, including instrumentation and varying concurrent workloads, not a controlled efficiency comparison.

## Recovery and Known Caveats

Three DrawerOpen HER runs (seeds 1-3) were already complete before the power interruption and were retained.
Five interrupted attempts were rerun from step zero. The recovery's 17 fresh runs plus three retained runs
form one 20-run HER/SABER cohort across the two tasks. HGR/PER add 20 new runs. No checkpoint continuation or spliced curves.
The task-specific attempt inventory and run origins are in [provenance.json](provenance.json).

The inherited MetaWorld observation includes the original goal in its last three components while HER
changes the separately concatenated goal. All compared SAC methods share this representation.
Its causal effect has not been established; the executed implementation is preserved rather than silently corrected.
Older fixed-temperature DrawerOpen prototype results are separate and are not pooled here.

## Files and Verification

- [Seed-level results](analysis/seed_metrics.csv), [aggregate results](analysis/aggregate_metrics.csv), [paired CIs](analysis/paired_auc.csv), [full curves](analysis/learning_curves.csv), [report](analysis/report.json).
- `data/<method>/seed_<n>/`: recorded config, summary, simulator description, execution environment, completion record and all evaluation rows/columns plus elapsed time and throughput.
- JSON exports redact the workspace prefix and replace unavailable nonfinite diagnostic fields with null; required loss/performance fields must remain finite. CSV evaluation strings are retained without rounding.
- `reference_report.json` is the task's original validated analysis, used to cross-check independently recomputed results. Original input and exported hashes are recorded.
- Checkpoints, replay buffers, full training logs and live control files are not published. These compact data regenerate performance metrics, not full training trajectories or every diagnostic.
- Exact executed [HER/SABER source](../../../ICML_27/experiments/draweropen_drawerclose_sac_her_vs_saber/source/) and [HGR/PER source](../../../ICML_27/experiments/draweropen_drawerclose_sac_hgr_per/source/) remain in ICML_27; no generic/legacy trainer is substituted.

From the repository root, with Python 3.12 and NumPy (recorded analysis version: 2.3.5):

```bash
python3 ICML_27/tools/sac_results.py verify --task drawer_close
```

Verification is read-only and CPU-only. It checks every checkpoint, completion, source snapshot, metric, CI, derived CSV and artifact SHA256; it never launches training.

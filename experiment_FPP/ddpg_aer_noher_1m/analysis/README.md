# FPP DDPG no-HER AER analysis

`analyze_ddpg_aer_noher.py` performs the frozen paired-seed analysis for:

- `ddpg_aer` versus `ddpg_er`
- `ddpg_aer` versus `ddpg_annealing_er`

All three methods must have HER disabled (`her_ratio=0`) and use the same
FetchPickAndPlace DDPG backbone. The analyzer requires every evaluation on the
exact `10k, 20k, ..., 1M` grid. It does not interpolate, smooth, or
forward-fill missing evaluations.

The primary endpoint is trapezoidal `eval/success_mean` AUC divided by
`1,000,000 - 10,000`. For both paired comparisons, the analyzer reports a
10,000-resample paired bootstrap confidence interval and a two-sided exact
sign-flip test over every seed-level sign assignment.

## Usage

If all run directories are below one common root:

```bash
python3 experiment_FPP/ddpg_aer_noher_1m/analysis/analyze_ddpg_aer_noher.py \
  --runs-root experiment_FPP/runs/fpp_ddpg_aer_noher_1m
```

Directory names do not matter. Runs are discovered recursively using
`config.json` fields `method`, `seeds.env_seed`, and `total_steps`.

If each method has a different root:

```bash
python3 experiment_FPP/ddpg_aer_noher_1m/analysis/analyze_ddpg_aer_noher.py \
  --ddpg-aer-root /path/to/aer-runs \
  --ddpg-er-root /path/to/er-runs \
  --ddpg-annealing-er-root /path/to/annealing-er-runs \
  --runs-root /path/used/only/for/default-output \
  --output-dir /path/to/artifact/analysis
```

The default paired seeds are 1-10. A pilot subset can be analyzed explicitly,
for example `--seeds 101-103`; exact sign-flip enumeration is supported for up
to 20 pairs.

## Outputs

- `per_seed_results.csv`: seed-level AUC, final/best success, first-hit steps,
  runtimes, and both paired differences
- `method_summary.csv`: aggregate method metrics
- `pairwise_summary.csv`: paired effect sizes, bootstrap intervals, exact
  p-values, and win/tie/loss counts
- `mean_curves.csv`: unsmoothed pointwise mean and sample standard deviation
- `analysis_results.json`: complete machine-readable results and provenance
- `analysis_summary.md`: concise human-readable tables

The script fails closed on incomplete grids, duplicate evaluations, incomplete
runs, non-sparse/non-FPP configs, nonzero HER ratio, seed mismatches, and common
backbone config differences.

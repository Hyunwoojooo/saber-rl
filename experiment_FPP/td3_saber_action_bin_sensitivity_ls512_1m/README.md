# FPP TD3 SABER Action-Bin Sensitivity

This package measures sensitivity to SABER's action discretization on
FetchPickAndPlace with TD3. The existing Main result uses five bins per action
dimension (`K=5`). These two new paired arms use `K=3` and `K=7`.

The frozen trainer source under `code/` is byte-identical to
`experiment_FPP/td3_saber_critic_only_v4_ls512/code`. The launcher explicitly
changes only `action_bins_per_dim` and the run/output identifiers. In
particular, it retains the original primary future-goal HER implementation,
critic-only normalized weighting, `local_search_max=512`, UTD 1, and the
TD3 Main settings.

## Registered Cohort

- task: FetchPickAndPlace-v4
- backbone: TD3
- variants: `k3` (`action_bins_per_dim=3`) and
  `k7` (`action_bins_per_dim=7`)
- reference: existing Main `K=5`
- seeds: 1, 2, 3, 4, 5
- budget: 1,000,000 environment steps per seed
- evaluation: 20 deterministic episodes every 10,000 steps
- primary metric: normalized success AUC over the exact 10k--1M grid
- secondary checks: final-quarter success AUC and support-positive rate

Production commands reject nonregistered seeds and budgets. Shorter runs are
allowed only with `--smoke`, `--validate_only`, or `--dry-run`.

## Completed K=3 Results

The K=3 arm completed all five registered seeds on the exact 10k--1M
evaluation grid. K=7 remains pending, so the full action-bin sensitivity
comparison is not yet complete.

| Seed | AUC | Final-quarter AUC | Final success | Hit90 |
|---:|---:|---:|---:|---:|
| 1 | 0.369949 | 0.627000 | 1.00 | 1,000,000 |
| 2 | 0.343687 | 0.569000 | 0.80 | not reached |
| 3 | 0.421970 | 0.801000 | 1.00 | 890,000 |
| 4 | 0.289646 | 0.506000 | 0.45 | not reached |
| 5 | 0.339141 | 0.514000 | 0.50 | not reached |

K=3 achieves mean AUC `0.352879 +/- 0.048313`, mean final-quarter AUC
`0.603400 +/- 0.120691`, and mean final success
`0.750000 +/- 0.264575`. Hit90 is reached on 2/5 seeds, with a median of
945k among the reached seeds. Its mean support-positive rate is
`0.396297 +/- 0.012079`, compared with `0.219063 +/- 0.009077` for Main
K=5.

The paired AUC difference, K=3 minus Main K=5, is `-0.089495` with a 95%
paired-bootstrap interval of `[-0.108687, -0.070303]`; K=3 is lower on all
five paired seeds. This completed arm is consistent with coarse action bins
activating the support predicate more broadly while reducing performance.
The interpretation remains conditional until the registered K=7 arm is
available.

The compact artifact under
`experiment_FPP/paper_artifacts/fpp_td3_saber_action_bin_sensitivity_ls512_1m/k3/` contains sanitized configurations,
environment snapshots, summaries, exact-grid metrics, recomputed tables, and
file hashes. Checkpoints and full per-update logs are excluded.

## Validation

Verify source parity, variant isolation, launch contracts, and command parity:

```bash
python3 \
  experiment_FPP/td3_saber_action_bin_sensitivity_ls512_1m/test_experiment_contract.py
```

Validate each environment without training:

```bash
python3 experiment_FPP/td3_saber_action_bin_sensitivity_ls512_1m/run.py \
  --variant k3 --seed 1 --validate_only --device cuda
python3 experiment_FPP/td3_saber_action_bin_sensitivity_ls512_1m/run.py \
  --variant k7 --seed 1 --validate_only --device cuda
```

Smoke-test both arms:

```bash
python3 experiment_FPP/td3_saber_action_bin_sensitivity_ls512_1m/run.py \
  --variant k3 --seed 1 --smoke --device cuda
python3 experiment_FPP/td3_saber_action_bin_sensitivity_ls512_1m/run.py \
  --variant k7 --seed 1 --smoke --device cuda
```

Set `SABER_PYTHON=/path/to/python` when a checkout does not contain `.venv`.
The selected interpreter must provide Numba; the launcher fails rather than
silently changing the registered support-scan implementation.

If Numba is not installed in the project environment, install the validated
versions into an isolated target and expose that target without mutating the
environment:

```bash
.venv/bin/pip install --target /tmp/saber-numba-0.66.0 \
  -r experiment_FPP/td3_saber_action_bin_sensitivity_ls512_1m/requirements-numba.txt
export PYTHONPATH=/tmp/saber-numba-0.66.0${PYTHONPATH:+:$PYTHONPATH}
```

## Production

The launchers run seeds concurrently and cap the number of active workers with
`MAX_PARALLEL`. To assign one variant to each of two machines:

```bash
# Machine A
MAX_PARALLEL=5 \
experiment_FPP/td3_saber_action_bin_sensitivity_ls512_1m/run_k3_all_seeds.sh \
  --device cuda

# Machine B
MAX_PARALLEL=5 \
experiment_FPP/td3_saber_action_bin_sensitivity_ls512_1m/run_k7_all_seeds.sh \
  --device cuda
```

To run all ten K=3/K=7 jobs on one machine with seven concurrent workers:

```bash
MAX_PARALLEL=7 \
experiment_FPP/td3_saber_action_bin_sensitivity_ls512_1m/run_k3_k7_parallel.sh \
  --device cuda
```

`run_all_seeds.sh {k3|k7|both}` exposes the same interface for schedulers.
It validates the frozen experiment contract and all commands before launching,
interleaves K=3 and K=7 jobs by seed for a combined run, records launch
metadata, writes one log per job under the experiment `_execution` directory,
and returns nonzero if any worker fails. Reduce `MAX_PARALLEL` if the target
machine cannot sustain the default worker count.

After all runs finish, use `analyze_results.py` to enforce the exact evaluation
grid and summarize the three-bin, Main five-bin, and seven-bin cohorts.

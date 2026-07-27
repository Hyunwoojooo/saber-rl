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

# TD3 SABER Action-Diversity Ablation

This experiment tests whether FetchPickAndPlace TD3+HER+SABER needs more than
policy-action-bin membership in an exact local replay context. It reuses the
settings and source snapshot from `experiment_FPP/td3_saber_critic_only_v4_ls512`
and adds one support predicate plus a matched random-count control.

No training result is included in this source folder. Threshold selection and
production runs must be performed in the order below.

## Support Definition

For query `i`, let `L_i` be replay rows in the same exact full context and let
`B_i` be their distinct replay-action bins. The scanned replay subset follows
the existing deterministic-tail `local_search_max=512` behavior.

The existing predicate is:

```text
main_i = 1[|L_i| >= 4]
         * 1[min(|L_i|, 16) >= 4]
         * 1[b(a_policy_i) in B_i]
```

The diversity predicate is:

```text
u_i       = |B_i|
diverse_i = main_i * 1[u_i >= Dmin]
```

`Dmin=1` has the same support predicate as the existing SABER implementation.
Policy-bin membership remains mandatory at every `Dmin`.

The Numba predicate scanner may exit early only after the count, membership,
and diversity conditions all pass. Its cache key includes `Dmin`. The 50k probe
disables both early exit and support short-circuits, so its unique-bin and
counterfactual-rate diagnostics are exact within the deterministic scanned
subset. Production runs restore the existing shortcuts; their logged unique
count can then be a threshold-sufficient lower bound, marked by
`support/unique_action_bin_count_is_exact=0`.

## Fixed Training Settings

- task: FetchPickAndPlace-v4 through `task_name=fetch_pick_and_place`
- total production steps: 1,000,000
- seeds: 1, 2, 3, 4, 5
- batch size: 256
- HER ratio: 0.8
- updates per environment step: 1
- evaluation: 20 episodes every 10,000 steps
- critic-only `batch_norm` support weighting with `lambda_support=1.0`
- `local_search_max=512`, `M_B=16`, `M_Q_min=4`, `N_local_min=4`
- five action bins per action dimension
- TD3 policy delay/noise/clip: 2 / 0.2 / 0.5
- environment, Torch, NumPy, action-noise, and evaluation seeds preserve the
  baseline mapping; weight seed is `20000 + seed`

The config-parity test rejects changes to baseline training arguments outside
the experiment name/output, diversity settings, and exact probe diagnostics.

Production uses `numba_support_scan=true`. `run.py` checks that the selected
Python environment can import Numba before it launches the trainer and fails
instead of silently using the Python fallback. If Numba is supplied as an
isolated target, expose it before running either condition:

```bash
.venv/bin/pip install --target /tmp/saber-diversity-numba-test \
  -r experiment_FPP/td3_saber_action_diversity_ablation/requirements-numba.txt
export PYTHONPATH=/tmp/saber-diversity-numba-test${PYTHONPATH:+:$PYTHONPATH}
```

## 1. Outcome-Blind 50k Probe

Run only seed 1 with `Dmin=1`. This keeps existing weighting and records
counterfactual positive counts for `Dmin in {2, 4, 8}` without applying them.

```bash
experiment_FPP/td3_saber_action_diversity_ablation/run_probe.sh --device cuda
```

Select `D*` from exact cumulative counter snapshots at 20k and 50k:

```bash
.venv/bin/python \
  experiment_FPP/td3_saber_action_diversity_ablation/select_diversity_threshold.py \
  --metrics experiment_FPP/runs/td3_saber_diversity_probe_d1_50k/RUN_DIR/metrics.csv \
  --output experiment_FPP/runs/td3_saber_diversity_probe_d1_50k/selected_threshold.json
```

For each candidate, the selector computes its 20k-50k positive rate `r_D` and
the ratio `r_D / r_1`. Candidates require `r_D >= 0.03`. It prefers a ratio in
`[0.4, 0.7]`, then the ratio closest to 0.55, then smaller `Dmin`. If no eligible
ratio lies in the interval, distance to the interval is considered first.
The selector reads only cumulative support counters. Evaluation success and AUC
are not used, and the JSON records `success_metrics_used=false`.

## 2. Production Runs

The selected JSON is mandatory. `run.py` rejects production commands without it
or if it does not encode the registered candidates, window, and outcome-blind
flag.

```bash
experiment_FPP/td3_saber_action_diversity_ablation/run_all_diverse_seeds.sh \
  experiment_FPP/runs/td3_saber_diversity_probe_d1_50k/selected_threshold.json \
  --device cuda
```

The mechanism control computes the Diverse reference mask first and then
assigns the high critic weight to exactly the same number of uniformly sampled
batch rows, without replacement:

```bash
experiment_FPP/td3_saber_action_diversity_ablation/run_all_random_count_seeds.sh \
  experiment_FPP/runs/td3_saber_diversity_probe_d1_50k/selected_threshold.json \
  --device cuda
```

Use `--dry-run` to print commands without creating an environment or running
training. `--validate_only --device cpu` runs the existing CPU setup validation.

## Diagnostics

The metrics CSV includes:

- `support/unique_action_bin_count_mean`
- `support/unique_action_bin_count_main_mean`
- `support/unique_action_bin_count_is_exact`
- `support/pre_gate_membership_rate`
- `support/pre_diversity_support_rate`
- `support/diversity_pass_rate`
- `support/diversity_pass_given_main_rate`
- `support/diverse_positive_rate`
- `support/random_count_reference_count`
- `support/random_count_assigned_count`
- cumulative probe counters/rates for D2, D4, and D8

`support/random_count_match` is asserted to be one for every random-count batch.

## Analysis

After all production artifacts exist, recompute AUC, final success, Hit90, and
the registered paired differences:

```bash
.venv/bin/python \
  experiment_FPP/td3_saber_action_diversity_ablation/analyze_results.py \
  --repo-root . \
  --diverse-root experiment_FPP/runs/td3_saber_diverse_dD_critic_only_v4_ls512 \
  --random-count-root experiment_FPP/runs/td3_saber_random_diversity_count_dD_critic_only_v4_ls512
```

The analyzer requires seeds 1-5 and exact evaluations at 10k, 20k, ..., 1M.
It does not interpolate or forward-fill. Success uses
`eval/env_success_mean`, falling back to `eval/success_mean` only when needed.
AUC is trapezoidal area normalized by `1,000,000 - 10,000`; Hit90 is the first
exact evaluation with success at least 0.90. Aggregates use arithmetic mean and
sample standard deviation (`ddof=1`). AUC CIs use 10,000 paired percentile
bootstrap resamples with RNG seed 27027.

## Tests

The scanner and runtime tests need Numba. They were verified with NumPy 2.3.5,
Numba 0.66.0, and llvmlite 0.48.0 in an isolated target so the repository venv
was not modified:

```bash
.venv/bin/pip install --target /tmp/saber-diversity-numba-test \
  -r experiment_FPP/td3_saber_action_diversity_ablation/requirements-numba.txt

PYTHONPATH=/tmp/saber-diversity-numba-test:experiment_FPP/td3_saber_action_diversity_ablation/code/src \
  .venv/bin/python experiment_FPP/td3_saber_action_diversity_ablation/code/src/test_numba_support_scan_fpp.py

PYTHONPATH=/tmp/saber-diversity-numba-test:experiment_FPP/td3_saber_action_diversity_ablation/code/src \
  .venv/bin/python experiment_FPP/td3_saber_action_diversity_ablation/code/src/test_numba_runtime_opt_fpp.py

.venv/bin/python experiment_FPP/td3_saber_action_diversity_ablation/test_select_diversity_threshold.py
.venv/bin/python experiment_FPP/td3_saber_action_diversity_ablation/test_run_config_parity.py
.venv/bin/python experiment_FPP/td3_saber_action_diversity_ablation/test_analyze_results.py
```

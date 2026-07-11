# FPP TD3 SABER Support-Query Ablation (LS512)

This experiment tests whether SABER needs the current policy action as the action-membership query. The completed main row in `experiment_FPP/td3_saber_critic_only_v4_ls512` is the source implementation and reusable `current_policy` result.

No 5k, 50k, or 1M training run is started by the preparation and test commands in this folder.

## One-Factor Definition

The exact local context, replay pool, count thresholds, action discretization, TD3/HER update, and support weight are fixed. Only `support_query_mode` changes:

| Mode | Support query | Interpretation |
|---|---|---|
| `current_policy` | current policy action for the same sample | Completed main SABER row |
| `replay_action` | sampled replay action | Removes current-policy relevance |
| `shuffled_policy` | a derangement of policy actions within the batch | Preserves the policy-action distribution but removes sample alignment |
| `context_count_only` | no action query | Keeps only the exact-context count gates |

For the first three modes, support is positive when the exact local pool has at least four transitions, `min(local_count, 16) >= 4`, and the query action bin appears in that pool. `context_count_only` omits the last condition. It can therefore change the number of positive samples and is not a strict count-matched control.

Replay-query intentionally does not remove the sampled transition from the local pool. Its possible self-support is a limitation of this ablation; removing it would also change the support pool.

`shuffled_policy` uses `make_derangement()` and the existing checkpointed `weight_rng`. It consumes no query-shuffle randomness in the other modes, and checkpoint resume restores the RNG state.

## Fixed Production Settings

- `FetchPickAndPlace-v4`, sparse reward
- TD3 + HER, `1_000_000` steps, seeds `1..5`
- batch size 256, HER ratio 0.8, one update per environment step
- evaluation every 10k with 20 episodes
- `M_B=16`, `M_Q_min=4`, `N_local_min=4`, `local_search_max=512`
- five bins per action dimension, `lambda_support=1`
- `weight_scope=critic_only`, `weighting_mode=batch_norm`
- `support_scan_mode=deterministic_tail_v1`
- TD3 delay/noise/clip `2 / 0.2 / 0.5`
- environment, Torch, NumPy, and action-noise seed `s`
- weight seed `20000+s`; evaluation seed `10000+s`

Mode output roots are fixed in `config.json`. `test_run_config_parity.py` verifies that mode, run name, and output root are the only mode-dependent command options.

## Dependencies

The runtime-optimized production path requires Numba and fails fast if the scanner is unavailable or JIT warmup fails.

```bash
.venv/bin/pip install -r experiment_FPP/td3_saber_query_ablation_ls512/requirements-numba.txt
```

Pinned scanner environment: NumPy 2.3.5, Numba 0.66.0, llvmlite 0.48.0.

## Validation

```bash
.venv/bin/python experiment_FPP/td3_saber_query_ablation_ls512/code/src/test_support_query_modes.py
.venv/bin/python experiment_FPP/td3_saber_query_ablation_ls512/code/src/test_numba_support_scan_fpp.py
.venv/bin/python experiment_FPP/td3_saber_query_ablation_ls512/code/src/test_numba_runtime_opt_fpp.py
.venv/bin/python experiment_FPP/td3_saber_query_ablation_ls512/code/src/test_current_policy_legacy_parity.py
.venv/bin/python experiment_FPP/td3_saber_query_ablation_ls512/test_run_config_parity.py
.venv/bin/python experiment_FPP/td3_saber_query_ablation_ls512/test_analyze_results.py
```

The legacy parity test compares all predicate/weight ndarrays byte-for-byte and the final `weight_rng` state in separate baseline/new source processes. The Numba scanner tests cover exact and early-exit paths, Python-reference parity, and cache on/off parity for every mode.

CPU setup validation, without training:

```bash
for mode in replay_action shuffled_policy context_count_only; do
  .venv/bin/python experiment_FPP/td3_saber_query_ablation_ls512/run.py \
    --mode "$mode" --seed 1 --device cpu --validate_only
done
```

## Staged Execution

Run these stages in order. Inspect every run before proceeding to the next stage.

1. Seed-1 5k smoke for each new mode:

```bash
for mode in replay_action shuffled_policy context_count_only; do
  .venv/bin/python experiment_FPP/td3_saber_query_ablation_ls512/run.py \
    --mode "$mode" --seed 1 --device cuda --smoke
done
```

2. Seed-1 50k pilot for each new mode:

```bash
for mode in replay_action shuffled_policy context_count_only; do
  .venv/bin/python experiment_FPP/td3_saber_query_ablation_ls512/run.py \
    --mode "$mode" --seed 1 --steps 50000 --device cuda
done
```

3. Only after smoke and pilot validation, run the three new modes for seeds 1-5:

```bash
for mode in replay_action shuffled_policy context_count_only; do
  bash experiment_FPP/td3_saber_query_ablation_ls512/run_all_seeds.sh \
    --mode "$mode" --steps 1000000 --device cuda
done
```

The completed main SABER runs are reused because `current_policy` parity passes. If that parity test fails in a future revision, do not reuse the main row; rerun `current_policy` for seeds 1-5.

## Logged Mechanism Metrics

The CSV/JSON logs retain the legacy `lite/psi_A_policy_rate` field and add the unambiguous `lite/psi_A_query_rate` alias. They also record:

- exact context-count eligibility, support-positive rate, and query-bin presence conditional on context eligibility
- high-weight count per batch
- raw and normalized weight mean, standard deviation, and ESS
- query-bin agreement with current-policy and replay-action bins
- support-positive rate by approach, grasp, lift, and place phase
- Numba/cache/early-exit diagnostics and a query-named audit alias

## Analysis

The analyzer requires exact evaluations at 10k, 20k, ..., 1M and never interpolates, smooths, extrapolates, or forward-fills. AUC is trapezoidal area normalized by `1_000_000 - 10_000`. Hit90 is the first exact 10k checkpoint with success at least 0.90. Aggregates use arithmetic means and sample standard deviations (`ddof=1`).

Paired AUC outputs use Main SABER minus each variant, 10,000 paired percentile bootstrap resamples with seed 27027, an exact two-sided sign-flip calculation over all 32 assignments, and leave-one-seed-out means.

```bash
.venv/bin/python experiment_FPP/td3_saber_query_ablation_ls512/analyze_results.py \
  --repo-root . \
  --output-dir experiment_FPP/reports/td3_saber_query_ablation_ls512
```

Primary interpretation is Main SABER versus each variant, not whether a variant exceeds HER. Mechanism snapshots are taken only at exact 10k evaluation rows. Missing legacy mechanism fields remain blank rather than being inferred.

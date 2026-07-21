# FPP DDPG SABER shuffled-policy query ablation

This directory implements the FetchPickAndPlace DDPG SABER query ablation
`ddpg_saber_query_shuffled_ls512`. It is a minimal extension of
`experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/`.

## Intervention

Canonical SABER queries support for sample `i` with the deterministic
current-policy action

`a_policy[i] = pi(observation[i], relabeled_goal[i])`.

This arm draws a fixed-point-free minibatch permutation `sigma` and uses

`a_query[i] = a_policy[sigma(i)]`, with `sigma(i) != i` for every sample.

Only the policy query action is reassigned. The following remain at their
original minibatch position `i`:

- observation, HER-relabeled goal, and remaining horizon;
- coarse and full context;
- replay action, reward, Bellman target, and Bellman residual;
- the critic-loss term receiving the resulting normalized weight.

The permutation preserves the policy-action multiset exactly. Different policy
actions may still occupy the same discretized action bin, so a zero
query-bin/current-policy-bin match rate is neither required nor expected.

This is **not** shuffled-weight and **not** random-support. The method remains
`b3_support_runtime_opt`; only `support_query_mode` changes from
`current_policy` to `shuffled_policy`.

## Preserved DDPG semantics

- single-critic DDPG, not TD3;
- uniform replay and future-goal HER ratio 0.8;
- Bellman regression at `Q(state[i], replay_action[i])`;
- unchanged actor and Bellman target;
- SABER weight applied only to the critic residual;
- `omega_actor == 1` for every sample;
- minibatch mean-one critic-weight normalization;
- deterministic recent-tail local scan capped at 512;
- canonical cache, short-circuit, and Numba scanner behavior;
- sampled transitions are not removed from the local replay pool.

## RNG and resume

Derangements use only the dedicated weight RNG with seed `20000 + seed`.
They do not consume global NumPy, replay-sampling, exploration-noise, Torch, or
evaluation RNG streams. The weight RNG state is stored in each checkpoint and
restored on resume. Resume fails before provenance files are written when the
checkpoint query mode differs from the requested mode.

## Canonical run

Production settings are seeds 1–5, 1,000,000 environment steps, deterministic
20-episode evaluation every 10,000 steps, and checkpoints every 50,000 steps.

Run a separate 20k correctness smoke:

```bash
python experiment_FPP/ddpg_saber_query_shuffled_ls512/run.py \
  --seed 1 --steps 20000 --device cuda \
  --output-root experiment_FPP/ddpg_saber_query_shuffled_ls512/smoke
```

Run all production seeds:

```bash
SEEDS="1 2 3 4 5" STEPS=1000000 DEVICE=cuda \
  bash experiment_FPP/ddpg_saber_query_shuffled_ls512/run_all_seeds.sh
```

Production results belong under
`experiment_FPP/runs/ddpg_saber_query_shuffled_ls512/`. Smoke, pilot, raw
logs, full metrics, and checkpoints must not enter the compact paper artifact.

## Correctness tests

```bash
SRC=experiment_FPP/ddpg_saber_query_shuffled_ls512/code/src
PYTHONPATH="$SRC" python "$SRC/test_support_query_modes.py"
QUERY_NUMBA=1 PYTHONPATH="$SRC" python "$SRC/test_support_query_modes.py"
PYTHONPATH="$SRC" python "$SRC/test_current_policy_legacy_parity.py"
PARITY_NUMBA=1 PYTHONPATH="$SRC" python "$SRC/test_current_policy_legacy_parity.py"
PYTHONPATH="$SRC" python "$SRC/test_ddpg_query_invariants.py"
python experiment_FPP/ddpg_saber_query_shuffled_ls512/test_analyze_results.py
python experiment_FPP/ddpg_saber_query_shuffled_ls512/test_artifact_tooling.py
```

The parity gate treats the legacy Main config's missing
`support_query_mode` as its historically implicit `current_policy` value.
Allowed resolved-config differences are exactly `run_name`, `output_root`,
and `support_query_mode`.

## Analysis and artifact

The analysis uses only `eval/env_success_mean` on the exact 100-point
10k–1M grid, without smoothing, interpolation, forward fill, or extrapolation.
AUC is trapezoidal area divided by 990,000. Main–shuffled paired inference uses
10,000 bootstrap resamples with seed 27027 and all 32 exact sign-flip
assignments.

The compact artifact is built at
`experiment_FPP/paper_artifacts/fpp_ddpg_saber_query_shuffled_ls512_1m/` and
must verify standalone with:

```bash
python verify_artifact.py --artifact-dir .
```

The existing Main seeds 1–5 are reused rather than retrained. Their archived
source commit is known, but their original execution HEAD was not recorded;
the artifact reports that provenance limitation rather than inventing an
execution SHA.

# FPP DDPG Random-Support (Batch-Wide Count Matched)

This submission-source package implements the FetchPickAndPlace-v4 DDPG+HER
Random-Support assignment control. It is derived from the verified strict-control
package at `experiment_FPP/ddpg_saber_context_eligible_count_matched_ls512/`
on repository base `549706f25b6b9f257464df3237cd5baa86e9972a`.

## Scientific definition

For every HER-relabeled minibatch, the canonical runtime-optimized Main SABER
query is evaluated first. Its current-policy action query, context population
predicate, and occupied replay-action-bin membership produce the binary Main
mask `M`. Let `k = sum(M)`.

The assignment pool is the complete minibatch, including Main-positive and
context-ineligible positions. A dedicated NumPy generator samples exactly `k`
distinct positions uniformly without replacement:

```text
P = {0, ..., batch_size - 1}
Z ~ Uniform({S subset of P : |S| = k})
w_raw[i] = 1 + lambda_support * 1[i in Z]
Omega[i] = w_raw[i] / mean(w_raw)
```

Main-positive positions remain ordinary candidates, accidental overlap is
allowed, and no disjointness or derangement constraint is imposed. The sampled
transition order and minibatch membership are never changed. `k=0` and
`k=batch_size` are deterministic and do not consume assignment RNG state.

The control recomputes `M` and `k` from its own current policy, replay buffer,
and relabeled batch on every update. It never imports a mask or count sequence
from another run.

Only the critic Bellman residual at the replay action is multiplied by `Omega`.
Actor weights are exactly one, and the actor objective and Bellman target remain
canonical DDPG+HER.

With binary SABER weights, this batch-wide exact-count control is
distributionally equivalent to uniformly shuffling the Main raw-weight vector.
Random-Support and shuffled-weight therefore must not be interpreted as
independent mechanism evidence.

This is not Random-Fixed-Count, Bernoulli random weighting, strict
context-eligible count matching, Replay-Action Query, Shuffled-Policy Query, or
a fully random control that skips support lookup.

## Intervention and parity

The production method remains `b3_support_runtime_opt`. Main and control differ
scientifically only in:

```text
support_assignment_mode: support -> batch_wide_count_matched
```

Bookkeeping-only additions are the dedicated assignment RNG and 1,000-update
diagnostic window. Allowed run-specific differences are `run_name`,
`output_root`, experiment metadata, `seeds.assignment_seed`, and
`assignment_diagnostic_window_updates`.

Fixed settings include DDPG single critic, FetchPickAndPlace-v4, sparse reward,
future HER 0.8, horizon 50, batch 256, replay 300,000, warm-up 5,000, random
actions 10,000, UTD 1, 256x256 networks, actor/critic LR 3e-4, gamma 0.98, tau
0.005, exploration noise 0.2, `M_B=16`, `M_Q_min=4`, `N_local_min=4`, local
search 512, five bins per action dimension, `lambda_support=1`, current-policy
query, deterministic-tail scan, runtime caches, short circuit, Numba scanner,
critic-only scope, and minibatch-mean normalization.

Seed mapping for experiment seed `s`:

- environment, torch, NumPy, and action noise: `s`
- evaluation: `10000 + s`
- weight computation: `20000 + s`
- assignment: `30000 + s`

The assignment RNG and diagnostic-window state are checkpointed. Strict resume
fails if either state or the assignment seed/mode contract is missing or
different; it never silently reseeds.

## Diagnostics and invariants

Every update records batch size, true context-eligible count, full assignment
pool count/rate, Main and assigned positive counts/rates, overlap, overlap/k,
Jaccard, raw and normalized mean/std/ESS for Main and assignment weights,
normalized sums, histogram mismatch counters, and invariant failures.

The trainer fails immediately unless:

- `M` is a subset of the canonical context-eligible mask;
- the assignment pool covers the entire minibatch;
- `sum(Z) == sum(M)`;
- Main and assigned raw and normalized histograms match exactly;
- critic weights are finite and positive, with mean one and sum batch size;
- actor weights are exactly one.

Non-overlapping 1,000-update windows and the final partial window are written to
`assignment_diagnostics_1000.csv`. Diagnostic aggregation consumes no RNG.

## Source validation workflow

From the repository root, with the project environment available:

```bash
PYTHON_BIN=.venv/bin/python \
  bash experiment_FPP/ddpg_saber_random_support_count_matched_ls512/run_tests.sh

DEVICE=cuda \
  bash experiment_FPP/ddpg_saber_random_support_count_matched_ls512/run_smoke_5k_seed1.sh

DEVICE=cuda \
  bash experiment_FPP/ddpg_saber_random_support_count_matched_ls512/run_technical_gate_50k_seed1.sh
```

Smoke and technical-gate success is based only on completion, exact step/update
counts and evaluation grid, finite critical values, zero assignment/histogram
violations, actor-weight isolation, resume equivalence, and source fingerprint.
Performance is never a gate and must not be used to tune this control.

The production launcher can be inspected without launching training:

```bash
DRY_RUN=1 MAX_PARALLEL=5 DEVICE=cuda \
  bash experiment_FPP/ddpg_saber_random_support_count_matched_ls512/run_main_1m_seeds1_5_parallel.sh
```

Production output is reserved for:
`experiment_FPP/runs/ddpg_saber_random_support_count_matched_ls512/`.
The legacy `experiment_FPP/runs/b3_support_random_critic_only_ls512/` data has
unknown source-commit provenance and is neither overwritten nor used as a
correctness gate.

Do not run seeds 1--5 x 1M or push this package without explicit approval.

Repository rule `experiment_FPP/.gitignore` ignores nested `analysis/`
directories. At a separately approved source-commit step, every file under
this package's `analysis/` directory must therefore be explicitly force-added
and confirmed with `git ls-tree`; the production Git preflight fails closed if
any frozen file is absent.

## Future production analysis and artifact

After separately approved production completion, analysis uses only
`eval/env_success_mean` on the exact observed 10k--1M grid. AUC is trapezoidal
integration divided by 990,000, without smoothing, interpolation, forward fill,
extrapolation, or seed exclusion. The paired comparison is Main minus
Random-Support, with 10,000 paired bootstrap resamples (seed 27027), all 32
sign-flip assignments, seed-wise differences, and leave-one-seed-out means.

The compact artifact target is
`experiment_FPP/paper_artifacts/fpp_ddpg_saber_random_support_count_matched_ls512_1m/`.
Its standalone verifier rebuilds exact-grid statistics, assignment invariants,
source/config parity, provenance, and manifests. Checkpoints, full raw metrics,
JSONL, logs, caches, and smoke/gate raw directories are excluded.

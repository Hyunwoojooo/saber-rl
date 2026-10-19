# HER PAL follow up protocol

## Question and evidence scope

Under the fixed DrawerOpen-DDPG HER training protocol, does SABER outperform
the published PAL critic objective in normalized success AUC?

This comparison concerns a prior method, not the assignment-specific mechanism
test. It is planned after seeing intermediate results of the assignment cohort.
It is not a previously unseen confirmation of the earlier paper. PAL settings
are fixed before any PAL training, using the author's published implementation,
not tuned to the observed SABER runs. The existing assignment experiment and
its two primary comparisons remain unchanged.

## PAL definition and adaptation

Let delta_i = Q(o_i, g_i, a_replay_i) - y_i, with the same pre-update DDPG target
as the reference experiment. With alpha=0.4 and threshold=1:

    ell_i = 0.5 * delta_i^2                    if |delta_i| < 1
            |delta_i|^(1+alpha)/(1+alpha)      otherwise
    Z_B = stop_gradient(mean_i(max(|delta_i|, 1)^alpha))
    L_PAL = mean_i(ell_i) / Z_B

Both the half-squared term and the author's outer-branch constant are preserved.
Do not add a continuity correction, substitute weighted MSE, cap priorities at
two, apply SABER's mean-one normalization, or differentiate through Z_B. With a
single critic, the author's max(|delta_1|, |delta_2|) is simply |delta|.
The official code estimates the normalizer using the minibatch. The theoretical
buffer-wide expression and exact expected-gradient statements are not claimed
to apply unchanged to stochastic HER plus this minibatch approximation.

The source is Fujimoto, Meger, and Precup, NeurIPS 2020:
https://proceedings.neurips.cc/paper/2020/hash/a3bf6e4db673b6449c2f7d13ee6ec9c0-Abstract.html
Author implementation and exact source provenance are in THIRD_PARTY_NOTICES.md.

Only the loss is adapted from the author's TD3 implementation. DDPG retains one
critic, its existing actor/target update cadence, Adam settings, and gradient
clipping max_norm=10. No twin critics or target-action smoothing are introduced.
This is HER+PAL on DDPG, not a reproduction of the original TD3 benchmark.
The loss-scale change relative to the reference mean(delta^2) is part of PAL;
no compensating learning-rate or factor-of-two tuning is applied.

## Frozen training settings

Ten fresh PAL runs, seeds 1--10, 2,000,000 environment steps per seed.
DrawerOpen-v3, sparse reward, reward function v2, horizon=200.
UTD=1; batch=256; replay capacity=300,000; warmup=5,000; random actions=10,000.
gamma=.98; tau=.005; actor/critic Adam lr=.0003; hidden=256; exploration sigma=.2.
Actor L2 action penalty=0. Success does not terminate; timeouts bootstrap.
HER nominal probability=.8; only stored transitions u>t are eligible, using
their next achieved goal. Empty future sets retain the desired goal.

Replay sampling, reward recomputation, terminal handling, action clipping, and
evaluation code remain copied from the frozen assignment trainer. PAL does not
populate the support-specific index, but HER's episode-to-slot bookkeeping is
unaltered. Replay tensors and sampled HER batches must match in the dedicated
index-enabled/index-disabled regression test, including buffer wrap.

No support lookup, eligibility mask, top-k, random count matching, or policy-action
support query is used. Legacy support fields in the inherited CSV schema are NA,
not measurements of zero support. PAL normalizer, base loss, and the proportion
|delta| >= 1 are recorded separately. These are not SABER activation statistics.

## Seeds and evaluation

Environment/Torch/NumPy/exploration seed=s. Replay/HER generator seed=s+17.
Unused weight/assignment generators retain their reference seeds but are not used
by PAL. The reference's selection generators were separate from replay and
exploration, so omitting them does not consume either training stream.

Evaluation: 20 deterministic episodes at 10k, 20k, ..., 2M. Reset seed is
10,000 + training_seed + 10*checkpoint_step + episode_index, starting at zero.
Methods share this evaluation protocol, not their evolving replay buffers.

## Analysis and reference reuse

Primary: trapezoidal success AUC over the exact 10k--2M grid, divided by 1,990,000.
FQ-AUC: exact 1.5M--2M grid, divided by 500,000. Final: success at 2M.
Hit90: first checkpoint >= .9, with hit coverage and conditional median.
Report seed metrics, sample mean/SD, all signed paired differences, and positive,
negative, and tied pair counts. Do not select methods, seeds, or endpoints after
observing PAL performance.

The sole primary contrast is SABER-minus-PAL AUC. Bootstrap 10,000 joint rows
of the ten fixed seed pairs using analysis seed 20260930. The 95% percentile CI
uses the 2.5% and 97.5% quantiles. It is approximate at ten seeds, not an exact
coverage guarantee and not an equivalence test. No two-contrast Bonferroni claim
is imported from the assignment experiment. Other metrics are descriptive.

All SABER seeds 1--10 must come from the support arm of the 2026-09-25 assignment
snapshot d281263084807ec6010986dc762301dce420b004e022c88985e51dcc50ccf496.
The source, complete configs, original attempt history, evaluation grid, and
physical host/software environment are audited read-only. No earlier five-seed
paper results or favorable subset may be substituted. This is a new, explicit
reference-use contract; it does not alter the original cohort's analysis.

Inference requires all ten compliant pairs. Every missing, interrupted, or failed
attempt remains in the status report. No artificial zero return is assigned to
an incomplete or crashed run. Conversely, a completed run with zero success is
a valid outcome and is never dropped or restarted for poor performance.

## Failures and execution gates

NaN/Inf loss, target, TD error, gradients, or parameters fail the run. No sanitizing
to zero. Infrastructure failure and learning failure are distinct statuses.
Infrastructure retries require a recorded diagnosis, same seed, new attempt
directory, and training from zero. Checkpoint continuation is forbidden.
Unexpected exceptions are not automatically classified as infrastructure.

The default launcher only prints a plan. Execution needs --execute. Production
also needs tests and a completed same-device real-environment smoke on the same
source digest and software environment. CPU certification does not authorize GPU
production. A separate, explicitly authorized one-shot waiter may start these
checks only after all 30 assignment runs finish and their processes exit. It
does not modify the existing experiment or its launcher. GPU production also
requires `final_readiness.json`, bound to the same source and environment,
covering all ten reference runs, CUDA loss parity, and checkpoint re-evaluation.

Smoke: 6k steps, random actions for 1k, warmup 5k, 1,001 critic updates, batch256,
evaluation at 2k/4k/6k with two episodes. Smoke results are excluded from science
tables and never used to tune PAL. No test or smoke is run during code preparation.

V2 corrects a synthetic-test environment fixture and policy-Q log columns only;
the PAL loss, training protocol, and baseline settings are unchanged from v1.
The old SABER policy-Q log columns are not used for a policy-query accuracy claim.
Before reference completion, only standard-library scheduling tests, AST checks,
and source hashes are executed. A failed runtime check blocks production rather
than triggering an unattended code change or hyperparameter adjustment.

## Cost and retained artifacts

Each attempt retains config, environment, pip freeze, status/tracebacks, training
and evaluation logs, runtime, and latest/best checkpoints. Checkpoints keep the
replay arrays, networks, optimizers, and RNG states for analysis, not resumption.
Per-checkpoint elapsed time enables a future time-axis analysis, but the current
SABER runs have additional assignment diagnostics. Their wall-clock numbers must
not be directly ranked against PAL or used to claim a speedup.

A future matched-instrumentation timing experiment must equalize hardware,
concurrency, evaluation, logging, and checkpoint policy, retaining algorithmic
support costs for SABER and none for PAL. That experiment is outside this launch.

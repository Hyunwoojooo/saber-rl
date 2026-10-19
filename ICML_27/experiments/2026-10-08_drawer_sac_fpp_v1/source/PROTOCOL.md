# Frozen scientific contract

## Question and scope

Under the same automatic-temperature SAC+HER pipeline on each Drawer task,
does critic-only support weighting improve learning? This extends the FPP SAC
implementation to two Drawer tasks. It is not a direct comparison to the old
fixed-alpha DrawerOpen baseline and does not alone prove the support mechanism.
Keep all seeds 1-10, including zero-success runs. Do not tune using their results.

## Common SAC core

The frozen FPP SAC implementation supplies the Gaussian actor, twin critics,
target critics, gradient clipping (norm 10), optimizer, replay, and training loop.
Alpha starts at 0.2 and is learned with Adam at 3e-4; target entropy is -4 for
the four-dimensional action space. Actor and critic rates are 3e-4, hidden
layers 256, gamma 0.98, tau 0.005, and action L2 penalty zero.

The common target is
`y = r + gamma * nonterminal * (min(Q1_target, Q2_target)(s', a') - alpha * log pi(a'|s'))`.
The same sampled next action is used for both target critics. Regression is at
the stored replay action: `mean(omega * (Q1-y)^2) + mean(omega * (Q2-y)^2)`.
There is no extra one-half factor. The actor uses the unweighted SAC objective;
temperature optimization is also unweighted. Updating the critic can indirectly
change later actor gradients; 'unchanged actor' means its rule, not its trajectory.
`noise_std=0.2` in the inherited API selects stochastic SAC actions; it does not
add DDPG Gaussian noise. Evaluation and support queries use the squashed mean.

## Drawer adapter and support

DrawerOpen and DrawerClose use their original MetaWorld v3 goal wrappers, sparse
reward v2, 200-step episodes, and task-specific context thresholds. The handle
position is the achieved goal and the environment target is the desired goal.
Synthetic target fallback is rejected. Reward is zero at Euclidean distance
<=0.05, otherwise -1. Success does not terminate; time limits bootstrap.

Uniform replay, capacity 300,000, batch 256, nominal HER probability 0.8, and
strict-future stored-transition selection are unchanged. HER uses that future
transition's next achieved goal; the last stored transition may not be relabeled.
The effective relabel fraction is logged. Replay actions are never substituted.

For SABER, query the deterministic current-policy action on the relabeled
observation-goal input. Retrieve up to 512 candidates from the maintained
coarse-index tail, then match full context under the relabeled goal. Require
at least four local matches and policy-action bin occupancy (K=5 per dimension).
The inherited M_B=16 and M_Q_min=4 conditions are preserved. Raw weights are
`1 + indicator`, then divided by their batch mean; actor weights remain one.
The query is no-grad and consumes no action/replay sampling RNG. Caches are
versioned by index state, query goal, full context and policy bin.

Tail is the index's maintained order, not guaranteed temporal recency. The 512
bound applies to the exact-matching scan, not all index maintenance costs. Early
exit preserves the indicator but reported local counts may be lower bounds.
Do not interpret these counts as exact whole-replay support counts. No extra
post-review assignment diagnostics are added to production timing.

## Schedule, evaluation and statistics

Each run takes 2M environment steps, UTD=1 after warmup: the first update occurs
at step 5,000, giving 1,995,001 updates. Random actions precede step 10,000.
Evaluate every 10,000 steps using 20 deterministic-policy episodes. For the same
training seed s, checkpoint t and episode j, evaluation reset seed is
`10000 + s + 10*t + j`. Training reset seed is `s + episode_index`.
Uniform replay RNG uses `s+17`; support RNG is `20000+s`.
The installed MetaWorld v3 ignores the seed passed to `reset`. The new local
wrapper explicitly calls its environment `seed` method before random task
selection and reset. This is applied to both training and evaluation in both
arms, and tested for identical reset observations after different prior actions.
It is a documented new-cohort reset convention, not a retroactive modification
of historical Drawer or FPP results.

The primary metric is official environment success-ever within each evaluation
episode, `eval/env_success_mean`. Goal success and success at the final timestep
are logged separately. Final denotes the final-checkpoint success-ever mean.
AUC is trapezoidal on the exact 10k..2M grid divided by 2M-10k. FQ-AUC is on
1.5M..2M divided by 0.5M. Hit90 is the first checkpoint at or above 0.9; report
coverage and leave unreached values missing, not artificially equal to 2M.

Primary contrasts are SABER-HER within each task, paired by seed. Report all ten
seed values, mean and SD, 10,000-resample percentile bootstrap 95% CIs, and
97.5% CIs for the two predeclared contrasts (Bonferroni nominal family coverage).
Percentile-bootstrap coverage with ten seeds is approximate, not exact. Missing
checkpoints or failed runs block the final analysis; no complete-case selection.
More steps, extra seeds, or replacement runs require an explicit amended protocol.

## Failure, provenance and differences from upstream

Each execution records source SHA256 and package/OS/Python/PyTorch metadata.
Claims are exclusive, outputs are fresh, and automatic checkpoint continuation
is disabled. Numerical failure is recorded, not sanitized into zeros. Interrupts
and other execution errors are retained separately; hardware failure versus
learning divergence requires inspection of the retained exception/log, not guesswork.
Any approved fresh rerun belongs to a new attempt protocol with the old failure
retained. No successful seed is silently substituted for a failed one.

Integration changes are limited to Drawer factories/config fields, SAC method
dispatch in the Drawer scanner, explicit MetaWorld reset seeding, run isolation/provenance, fail-fast finite/shape
checks, and terminal-mask extraction. The original FPP finite update is checked
numerically against the frozen original, including optimizer and target states.
Both HER and SABER share exactly this modified code. Both still maintain the
replay index as in FPP; HER does not perform support queries or Numba warmup.

The four CPU smokes retain the 256-unit networks, batch 256, horizon 200, SAC
rules and support thresholds; only duration, warmup, capacity and evaluation
schedule are shortened. They test integration, not convergence, GPU numerical
equivalence, full-buffer performance or long-run stability. Environment and
library packages are not installed or changed by this experiment.

## Deferred CUDA gate (authorized 2026-10-08)

Wait for the preceding Random512 ten-seed experiment and both of its post-analyses
to succeed, then require an idle GPU. Recheck CPU tests first. CUDA validation
uses the 42-dimensional Drawer input, batch 256 and hidden layers 256 to compare
three updates (uniform, mixed mean-one weights, uniform) to the frozen FPP agent,
including optimizer/temperature/target state (rtol 1e-6, atol 1e-7).

Next run all four task/method combinations for 600 steps with the smoke settings,
then for 100,000 steps with production settings except duration and output paths.
Stability runs retain 5k warmup, 10k random actions, 300k capacity, 10k evaluation
frequency, 20 evaluation episodes and 50k checkpoint cadence. Each must perform
95,001 SAC updates. At most four validation runs execute concurrently; each stage
has a fixed 24-hour execution timeout, not a performance-based stopping rule.

Require normal completion, the full declared evaluation grid, finite active
loss/alpha/Q logs, positive alpha, no Numba fallback for SABER, finite saved model
and optimizer state, consistent replay action bins and successful CPU checkpoint
reload. Preserve hashes of validation evidence and recheck them before production.
Validation seed 1 is fixed, but these shortened runs are never reused as paper
results. Zero success is a valid outcome, not a failed validation or replacement
criterion. The 100k gate cannot establish full-buffer or full-2M stability.

Only after all gates pass, schedule DrawerOpen's 20 fresh runs, its complete-pair
analysis, then DrawerClose's 20 runs and joint analysis. Five production jobs may
run concurrently. A predecessor/gate/source mismatch halts the queue. If a
production job fails, stop launching further jobs, retain already active jobs
until they finish, record the failure, and do not start the next task. A service
stop terminates the whole service group. No automatic resume, restart, seed
replacement, dependency installation or source mutation is allowed.

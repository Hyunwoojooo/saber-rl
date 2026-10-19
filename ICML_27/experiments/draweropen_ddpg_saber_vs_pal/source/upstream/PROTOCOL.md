# Assignment-specific value of current-policy support

## Question and limits

Holding context eligibility and the SABER-induced high-weight count fixed within
each online minibatch, does assigning high weights to support-positive positions
improve learning relative to random or TD-error assignment?

All arms retain SABER information through E and k. This is NOT a test of whether
support computation as a whole is necessary, an unbiased Q-error diagnostic, a
PAL comparison, or a claim that policy-action gradients become more accurate.

## Three arms

For batch positions i, E is the set passing the frozen context-population rule.
M is the original current-policy action-bin support mask, with M subset E.
Let m=|E| and k=|M|. Duplicate replay transitions in a sampled batch remain distinct
batch positions, exactly as in the inherited training objective.

- `support`: select M.
- `random`: sample exactly k positions uniformly without replacement from E.
- `td_topk`: select exactly k positions with largest absolute pre-update Bellman
  residual within E. Randomize only boundary ties, uniformly without replacement.

Each arm computes E, M, k using its OWN current policy, replay and relabeled batch.
Different online runs are NOT constrained to have identical replay, E, or k.
All arms set raw w_i=1+selected_i and Omega_i=w_i/mean(w). Thus the raw ratio is
2:1 where both groups are present, mean Omega is one, and actor weights are one.
The optimized residual is evaluated at the replay action, not the query action.
The TD selector receives abs(delta.detach()) from the same forward pass whose
attached delta is used by the critic loss; no second target is constructed.

Eligibility is computed even when a missing coarse action bin proves M_i=0.
This is the frozen Strict eligibility rule, not a new context approximation.
The original tail-512 index ordering, HER future transition convention, reward,
termination rules, and task thresholds are unchanged. The scan is not recency.

## Fixed production contract

DrawerOpen-v3, reward function v2, DDPG+HER, 2,000,000 environment steps per run.
Seeds 1--10 in all three arms; no optional stopping or result-dependent seed count.
UTD=1, batch=256, replay=300,000, horizon=200, warmup=5,000, random actions=10,000.
gamma=.98, tau=.005, actor/critic lr=.0003, hidden=256, exploration sigma=.2.
HER nominal probability .8 with u>t stored-transition indexing. Success does not
terminate; timeout bootstrapping is enabled. K=5, N_scan=512, N_local=4, M_B=16,
M_Q_min=4, lambda_support=1, mean-one critic-only weighting.

All remaining frozen defaults, task bins and constants are in the copied config;
`requested_config.json` and `run/config.json` serialize the entire effective config.
Production exposes no overrides for scientific hyperparameters. Seed/device/path
are launcher parameters; `smoke` is an explicitly separate profile.

Evaluation uses deterministic actions, every 10,000 steps, 20 episodes/checkpoint.
Reset seed = 10,000 + training_seed + 10*checkpoint_step + episode_index (0-based),
the inherited formula, identical across arms for corresponding evaluations.
Training random streams: environment/Torch/NumPy/exploration seed=s;
replay/HER=s+17; legacy weight RNG=20,000+s; random assignment=30,000+s;
TD boundary ties=40,000+s. Selection and logging never use replay/exploration RNGs.
Hypothetical random and TD masks are calculated on all arms for diagnostics;
their dedicated streams are advanced regardless of whether logging is enabled.

## Diagnostics and edge cases

- k=0: identical uniform loss for all arms. Intersection is zero; overlap/k and
  empty-union Jaccard are NA, not zero.
- k=m>0: every arm selects E. Weights coincide (not necessarily uniform unless E
  is the full batch).
- 0<k<m: assignment rules can differ; report these updates separately.
- Under random selection E[|M intersect R|]=k^2/m and E[|M intersect R|/k]=k/m
  for k>0. For m=0 the intersection is identically zero. We do not report the
  Jaccard expectation by substituting expected intersection into a ratio.

Every update logs to an accumulator. `assignment_windows.csv` contains all-update
and informative-update rows every 1,000 updates, including the final partial
window. Means have explicit valid-update counts. Aggregation across windows is
count-weighted; across seeds each seed is one statistical unit. Support activation,
eligible count, empty/saturated/informative fractions, support--TD/random/assigned
intersection, overlap/k, observed Jaccard and random expectations are retained.
The inherited live-target post-update residual diagnostic is DISABLED for these
three arms; no fixed-target or Q-accuracy claim is made from these logs.

## Metrics and inference

Primary outcome: trapezoidal success AUC on the EXACT shared scheduled grid
10,000, 20,000, ..., 2,000,000, divided by 2,000,000-10,000. Missing or duplicate
checkpoints fail validation; no intersection-based shortening or interpolation.
Secondary: final-checkpoint success, FQ-AUC on 1.5M--2M divided by 500k, first
checkpoint with success >=.9 (Hit90, with coverage and conditional median),
recorded steps/s, and seed-level assignment diagnostics.

Two preregistered contrasts: support-minus-random and support-minus-td_topk.
Bootstrap 10,000 resamples with analysis RNG seed 20260925. Draw ten seed rows
with replacement and reuse the SAME row indices for both contrasts. Compute
percentile CIs at [.025,.975] (95%) and [.0125,.9875] (97.5%). The latter are the
Bonferroni pairwise intervals for a nominal 95% family level, NOT an exact finite
sample simultaneous-coverage guarantee. Ten-seed bootstrap coverage is approximate.
Reference: https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm
Report all seed differences, positive-pair counts, intervals and mean/SD; do not
switch the primary comparison or metric after seeing outcomes.

## Failures and execution policy

All runs are fresh. Production checkpoint continuation is rejected. Retain all
attempts. Loss, target, TD, gradient, or parameter non-finites fail the run; they
are never converted to zero. Poor but finite performance is a valid observation.

- Learning failure: record `failed_learning`; no replacement seed, automatic retry,
  or silent successful-only inference. Publish the failure alongside partial curves.
- OS/device/storage interruption: retain evidence and classify infrastructure
  separately. After a documented infrastructure diagnosis, rerun from step zero
  with the SAME seed, in a new attempt folder. No trajectory stitching.
- Unexpected/code error: `failed_unclassified`, not automatically infrastructure.
  Fixes require a new frozen source and a new cohort plan if production has begun.
- SIGINT/SIGTERM: record `interrupted`. SIGKILL/power loss can leave `running`;
  audit the process/host evidence before any manually documented reclassification.

The launcher never automatically retries. `--retry-infrastructure REASON` accepts
only infrastructure/interrupted attempts, never a completed or learning-failed run.
Analysis first writes all 30 statuses and all attempt histories. With any missing,
failed or noncompliant arm/seed it withholds the primary paired CI rather than
selecting complete cases. This does not assign artificial zero returns to failures.

## Historical reuse and audit

Default plan is 30 fresh runs. Existing 1--5 cohorts are NOT automatically imported.
Reuse requires a separate recorded contract and provenance review, independent of
observed performance. Two different checks are required:

1. Update parity: identical batch, model, optimizer state, target and mask must
   yield identical loss and parameter/optimizer/target updates.
2. Selection-rule parity: sample exactly k eligible batch positions uniformly
   without replacement. A changed independent RNG stream alone does not invalidate
   an old run; eligibility, reward, termination, or selection-rule changes do.

Tests cover both. Current random selection uses the frozen Strict RNG seed and
choice rule; bytewise seed-trajectory equivalence is not assumed. Historical main
configs reconstructed from proxies are not proof of the original environment.
Check evaluation grid/seeds, versions and physical host matching before reuse.
Unavailable new overlap traces must be marked unavailable, not reconstructed from
aggregate historical performance. If reuse is approved later, write a new explicit
cohort catalog and audit; do not merge folders to bypass this validator.

## Provenance, smoke, and cost

`upstream/` is an immutable copy of the frozen Strict source. `code/` is the only
modified trainer. `source_manifest.json` binds source/tests/protocol with SHA256;
each attempt records that digest, full config, installed package versions, pip
freeze, system/GPU details, status and logs. Checkpoints retain replay, index order,
networks, optimizers and RNG states for offline audits, not for production resume.
Only the latest and best checkpoint are retained, matching the frozen storage
policy; no all-checkpoint retrieval study is claimed.

Smoke uses 6k steps, random-action prefix 1k, warmup 5k, batch256, evaluation every
2k with two episodes; each arm performs 1,001 updates. It validates execution and
logging only. Production needs passing regression tests and all three completed
smokes on the same source digest. No smoke success is used to tune the protocol.
Additional counterfactual masks and non-finite checks incur instrumentation cost;
steps/s measures this experiment and is not directly comparable to old speed logs.

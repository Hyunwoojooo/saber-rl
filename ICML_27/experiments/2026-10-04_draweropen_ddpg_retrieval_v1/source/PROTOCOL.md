# Online retrieval comparison: fixed before new production runs

## Question

Does replacing the maintained-index tail by uniform random candidate retrieval,
with the same 512-candidate cap, change learning in DrawerOpen DDPG+HER+SABER?
Both arms assign high weight to their own support-positive positions. This is not
random minibatch weight assignment, a test of exact continuous support, or proof
of greater policy-query Q accuracy. Historical tail results and the final-state
retrieval audit were observed before this experiment was designed.

## Candidate-selection intervention

Use the inherited live, deduplicated coarse-key bucket. Tail selects its last
min(512, n) positions in maintained index order, not temporal recency. Random
samples min(512, n) uniformly without replacement before full-context filtering;
when n<=512 both inspect the complete bucket. There is no replacement of invalid
or full-context-mismatching candidates to fill a quota of 512 exact matches.

Random selects once per coarse key per critic update; every query in that bucket
in the update shares this subset. Redraw at the next update, even if the bucket
has not changed. The retrieval generator seed is 50000+training_seed, separate
from replay/HER, exploration, assignment diagnostics, TD ties and evaluation.
Update-local candidate cache and cleared random support cache prevent unintended
cross-update reuse. Tail keeps its unchanged mathematically exact versioned cache.
The inherited optimized matching/early-exit kernel is used by both arms. The cap
is on available scan candidates, not a guarantee of 512 examined slots or equal
wall-clock cost. Cache hits, early exits and enumeration costs can differ.

Candidate observations are re-contextualized using the query's HER goal. Context
eligibility and current-policy action-bin occupancy retain the original rules.
High-weight count is NOT matched across arms; changes to this count are part of
the retrieval intervention. Each run has its own evolving policy/replay/batches.
Raw weights are 1+support_indicator, normalized to mean one. Actor weights are
one and Bellman regression remains at the replay action, not the query action.

## Fixed contract

Seeds 1--10. DrawerOpen-v3, sparse reward v2, DDPG, 2M steps/run, UTD=1,
batch256, replay300000, horizon200, warmup5000, random-action prefix10000.
Gamma .98, tau .005, actor/critic lr .0003, hidden256, exploration sigma .2.
HER nominal .8, strict future stored-transition indexing, success does not end
episodes, timeout bootstrapping enabled. K=5, N_scan=512, N_local=4, M_Q_min=4,
M_B=16, lambda=1; context thresholds unchanged. base_protocol.py retains the full
original configuration. The only scientific config difference is support_scan_mode.
Both modes retain all original assignment instrumentation, including hypothetical
random/TD masks; these are diagnostics only and do not control the critic weights.

Evaluate every 10k steps with 20 deterministic-action episodes, reset seed
10000+training_seed+10*checkpoint_step+episode_index. Use the same physical host,
verified libraries and one numerical-library thread per worker. Never upgrade
the environment during this experiment. New random production uses five workers,
seeds 1--5 followed by 6--10 as slots become available; allocation ignores outcomes.

## Historical reuse and safety gates

The default reuses ALL ten completed Tail/support runs, including zero-success
seed7, only after source SHA256, complete configs, environment, budget and curves
are audited. Require frozen-to-new Tail mask/weight/update/optimizer/target parity
and identical upstream/new short GPU trajectories. Two independent real-env
smokes must pass on the final source digest. If reuse fails, do not mix results:
freeze a documented two-fresh-arm cohort before training, independently of scores.
The new random launches do not claim unseen-data confirmation or a contemporaneous
runtime comparison with the historical Tail runs.

## Metrics

Primary: Random512-minus-Tail normalized success AUC, on all 200 checkpoints
10k,20k,...,2M, denominator 2M-10k. Missing, duplicated or reordered checkpoints
fail completion; do not shorten the interval or interpolate. Report all ten seed
differences and equal-seed mean/SD. Paired percentile bootstrap: 10000 draws of
ten seed rows, fixed analysis seed20261004, 2.5/97.5 percentiles. One primary
contrast; coverage is approximate, not an exact finite-sample guarantee. CI
including zero does not establish equivalence. No significance-dependent stopping,
seed extension, seed exclusion or switch of the primary outcome/direction.

Secondary: FQ-AUC1.5M--2M, final checkpoint success, Hit90 coverage and median
among reached runs, zero-evaluation-success count, steps/s, support and eligibility
rates, empty/informative/saturated-update fractions and original timing/scan logs.
Final-state tail/random/full search audits use each arm's own checkpoint; no
earlier unretained historical replay states will be invented. Query repetitions
are not additional independent training seeds. Logging statistics are descriptive.
Timing is instrumentation-inclusive; historical load differences preclude a strict
algorithmic wall-clock superiority claim from these runs alone.

## Failures, retention and scope

All new runs start from scratch; production continuation is prohibited. Preserve
every attempt. Nonfinite loss/TD/gradient/parameters are failed_learning, not zero
success and not silently replaced. Poor finite performance remains a valid result.
Infrastructure interruption may be rerun with the same seed from zero only after
documented diagnosis, in a new attempt directory. Unexpected errors are not
automatically excused as hardware errors. No automatic retries. With incomplete
or noncompliant paired data, withhold the primary CI and report every status.

Smokes use the original 6k profile and are excluded from results. Checkpoints,
replay, index order, models, optimizers, RNGs and random retrieval RNG state are
retained for verification/offline analysis, not permission for resumed production.
Conclusions are restricted to DrawerOpen DDPG and this fixed context/scan budget.

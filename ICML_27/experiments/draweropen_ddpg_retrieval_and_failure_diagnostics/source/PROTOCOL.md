# Post-PAL retrieval and failure diagnostics

This is a new, independent, read-only analysis experiment. It does not modify or
resume either the assignment experiment or PAL. The user authorized these two
diagnostics to run after PAL, not any further online training.

## Completion barrier and resource policy

Wait for PAL seeds 1--10, 2M steps each, and its final paired analysis to complete
successfully. Require the frozen source identities, fresh single attempts, full
training budgets, successful analysis gate, no remaining upstream training or
analysis process, and an idle GPU process list. Zero success is a valid scientific
outcome and is never a launch/retry criterion. PAL failure blocks this queue.

The waiting process uses only the standard library and polls every 300 seconds.
No Torch/NumPy/Numba import, checkpoint deserialization, simulator, GPU work, or
timing benchmark runs in the waiting process. After the barrier, all diagnostics
run sequentially on CPU with one numerical-library thread. Current PAL processes
are never signalled or modified. No dependencies are installed or upgraded.

Pipeline: synthetic regression tests, 30-run failure analysis, one-checkpoint
retrieval pilot, ten-checkpoint retrieval audit, artifact verification and report.
A failed stage stops the queue. No automatic retry, outcome-based resampling,
partial-cohort completion, or output overwrite is allowed. A stage timeout means
incomplete work, not authorization to reduce the sample count.

## Retrieval question and scope

For fixed replay, policy, HER queries, context thresholds, and action bins, how
much do tail-512 and random-512 disagree with exhaustive support matching?

- Dataset: final checkpoints of all ten SABER support runs in the frozen
  2026-09-25 DrawerOpen-DDPG assignment experiment. No outcome-based selection.
- Retained latest checkpoints are at 2M steps. Best checkpoints are not included
  in this first audit, avoiding performance-selected snapshot comparisons.
- 1024 queries per seed, in four batches of 256. Use the copied replay sampler,
  strict-future achieved-goal indexing, and nominal HER probability 0.8.
- Query RNG seed: 2026100200 + training seed, separate from retained training RNGs.
- All candidates are re-contextualized under the query's HER-relabeled goal.
- Policy inference is CPU-only from the restored frozen actor. The exact same
  query action bins are used for all three retrieval methods.
- Tail is the last 512 valid unique entries in maintained index order, NOT an
  assertion of temporal recency. Random selects up to 512 uniformly without
  replacement from the same coarse bucket, with five independent repetitions.
- Random seed stream: SeedSequence([2026100300, training_seed, query, repetition]).
- Full evaluates every valid unique slot in the same coarse bucket. Validate
  coarse-index membership against every live replay slot first. Because full
  context contains all coarse-key components, exhaustive bucket matching is the
  same reference as full-replay matching. Also directly cross-check the first
  16 queries per seed against all live replay slots, bypassing the index.
- Keep K=5, N_local=4, M_Q_min=4, M_B=16, and all original context thresholds.
- Require exact eligibility and support-mask agreement between the tail reference
  and the copied optimized production weight computer on every audited query.
  Also check critic-only, 2:1 raw weights, and mean-one normalization.
- A 256-query seed-1 pilot checks correctness and runtime only. It cannot choose
  seeds, thresholds, query distributions, or scientific sample size for the audit.
  Pilot queries are not additional independent evidence.

Report support/eligibility activation, disagreements, conditional support misses,
population-related versus membership-related misses, candidate/local counts,
tail/random disagreement, and per-query timing. Conditional rates with zero
denominator are NA/null. Report all seeds and repetitions, including zero support.
Aggregate equally across training seeds after averaging nested random/query
measurements. Do not treat thousands of queries as thousands of independent runs.

Timing uses one common exhaustive NumPy exact-matching kernel for all methods.
It includes coarse-candidate enumeration and subset selection, but excludes actor
inference, setup/JIT, index maintenance, and training. Method order rotates across
queries. This is explicitly REFERENCE CPU lookup cost, not a benchmark of the
optimized Numba/cached production implementation or end-to-end bounded complexity.
Production tail computation is a correctness reference, not a time-matched arm.

Full matching is a reference under the existing discrete definition, not truth
about continuous support. This study does not test online learning with random
retrieval, and final checkpoints cannot characterize all earlier training stages.

## Failure and saturation analysis

Use every run in support/random/td_topk x seeds 1--10: 30 fresh runs. Verify the
200 shared evaluation checkpoints per run and all 1,995,001 diagnostic updates.
Do not fabricate support diagnostics for PAL, which has no support instrumentation.

Keep the original AUC, FQ-AUC, final success, Hit90, and paired bootstrap intervals
unchanged. Define separately: zero evaluation success over the full run, reaching
Hit90 at least once, training-episode successes, and execution failure. Do not call
every non-Hit90 run an execution failure or infer no exploration success from zero
evaluation success.

Saturation is k=|E|>0. It makes assignment rules coincide for a fixed batch, but
does not imply E is the entire batch or the three online trajectories are equal.
Log first saturation as a start/end WINDOW, not an invented exact update. Retain
all per-window trajectories and summarize 100k environment-step bands by assigning
whole diagnostic windows according to their end step. Report first-positive
training/evaluation observations and their temporal relation to saturation.

Compute diagnostic averages using valid-update counts within runs and equal seed
weights across runs. Critic/TD averages describe logged snapshots, not every
gradient update or true Q error. Include first-100k and first-500k associations
with final outcomes as exploratory summaries only.

This analysis was selected after seeing the completed assignment outcomes.
Outcome groups, temporal associations, and the observed saturation pattern are
post hoc, not causal evidence or validated failure predictors. No new significance
tests or seed exclusions are introduced.

## Artifacts and provenance

Preserve byte-identical upstream code with its original source manifest. Seal all
new scripts/tests/protocol with SHA256. Original checkpoints/logs are read-only;
write solely into this experiment's outputs. Artifact manifests cover input and
output hashes and fail if inputs change during analysis. Checkpoints are trusted
local experiment files; no arbitrary external pickle is accepted.

Final outputs: per-query and per-seed retrieval CSVs, random-repeat summaries,
30-run failure/trajectory/time-band CSVs, original primary inference, JSON reports,
SHA256 manifests, and outputs/final/REPORT.md. No commit/push is scheduled.

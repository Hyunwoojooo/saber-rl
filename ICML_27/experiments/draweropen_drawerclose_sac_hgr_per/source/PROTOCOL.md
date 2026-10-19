# Frozen baseline protocol

## Question and provenance

Do the retained FPP HGR and full-PER baselines outperform or underperform SAC HER
and critic-only SABER on DrawerOpen and DrawerClose under the same common SAC
and evaluation contract? This addresses baseline coverage, not isolation of the
SABER support predicate. Reference outcomes had already been observed when this
extension was chosen; all comparisons are exploratory.

The canonical shared source is the FPP-derived, automatic-alpha SAC used by the
current recovery cohort. No earlier source is edited. `upstream/current`,
`upstream/hgr`, `upstream/per`, `upstream/fpp`, and `upstream/drawer` preserve the
115 original files byte-for-byte. SHA256s and original paths are recorded in
`upstream_manifest.json`. Changes are confined to the independent local copy.

## HGR

Preserve the existing FPP implementation, not a newly tuned surrogate:

1. Maintain priorities for all completed-episode transition/achieved-goal pairs.
   For episode length T there are T(T+1)/2 pairs. A transition at position j may
   select `next_achieved_goal` from a slot k >= j, i.e., achieved state g_(k+1).
   This includes the immediate next achieved goal. HER, SABER and PER instead
   use the reference strict-future stored-slot rule k > j. This difference is
   intentional baseline behavior and must be disclosed in comparison captions.
2. With nominal HER ratio 0.8, select a completed episode according to its mean
   pair priority raised to 0.6, then a pair according to pair priority^0.6.
   Non-relabeled samples remain uniformly drawn. No uniform-mixture component.
3. Pair priority is min(|delta_1|, |delta_2|)+1e-6, from the SAME stochastic
   Bellman target used by that critic step, before any optimizer update. Pair
   priorities initialize at the historical maximum. No priority cap. Repeated
   occurrences of the same pair use the retained FPP sequential-update rule.
4. The two-level importance coefficient is
   `(N_episode * P_episode)^(-beta_episode) * (N_pair * P_pair)^(-beta_goal)`.
   Normalize by the maximum among relabeled batch entries only; non-relabeled
   entries have weight one. Do NOT renormalize the batch mean. Both critics use
   this coefficient; actor/alpha losses have no importance coefficient.
5. Both beta values linearly anneal from 0.4 to 1 over 1,995,001 critic updates.
   At an update, beta uses the number of already-completed updates. The saved
   final counter reaches beta=1. No schedule is accelerated during validation.
6. Overwriting any slot removes its old completed-episode pair table. Only
   complete, uncorrupted episodes can re-enter the eligible table set.

HGR is a replay/goal-selection baseline; do not describe this as a loss-only
control or assert identical relabeled minibatches across methods.

## Full PER

1. Before HER, draw stored transitions with replacement using
   `P(i) = priority(i)^0.6 / sum(priority^0.6)`.
2. Apply the unchanged strict-future HER relabeling to the selected slots (0.8).
3. Raw IS coefficients are `(N * P(i))^(-beta)`, normalized by their sampled
   maximum. As in the retained FPP full-PER baseline, normalize these coefficients
   once more by their minibatch mean for the critic loss. Actor/alpha are
   unweighted. This is not the separate PER-style loss-only experiment.
4. beta(step)=0.4+0.6*min(step/2,000,000,1). Priority is
   `0.5 * (|delta_1| + |delta_2|) + 1e-6`, using the same pre-update target as
   the critic. Duplicate replay slots take the largest sampled priority.
   Newly inserted slots get the active maximum (floor=1), matching FPP.

Both priority extractors reuse the existing target tensor: they do not sample a
second stochastic target action or alter the RNG stream for a fixed batch.
Regression tests compare uniform replay and prioritized sampling to immutable
references and compare complete model/optimizer updates at fixed batches.

## Shared evaluation and analysis

- Drawer sparse reward v2, nominal success threshold 0.05, 200-step episodes.
- Success is not terminal; time-limit truncation is bootstrapped.
- Environment/task selection and reset seeding match the current reference.
- Seeds 1-5, 2M environment steps, UTD=1, evaluation every 10k with 20 episodes.
- For training seed s, checkpoint t and evaluation episode e, reset seed is
  `10000 + s + 10*t + e`. Same rule for all four methods.
- AUC uses the complete 10k..2M grid; normalization divisor is 1,990,000.
- FQ-AUC uses 1.5M..2M; divisor is 500,000. Final success is the 2M evaluation.
- Hit90 is the first checkpoint with success >=0.9; report non-attainment.
- Primary contrasts: SABER-HGR and SABER-PER for EACH task, four contrasts total.
  Paired seed bootstrap: 10,000 resamples, RNG seed 2027, 95% percentile CIs.
  Also report 98.75% marginal intervals (0.625/99.375 percentiles), Bonferroni
  for four contrasts at nominal 95% family coverage. Bootstrap coverage is
  approximate with only five seeds. Reusing the same resampling indices couples
  all contrast draws by seed row. SABER-HER is a secondary reference comparison.
- Report every declared seed, all curves and failures. Do not shorten a grid or
  use only completed/successful seeds for final paired comparisons.
- Throughput/runtime are logged. Runs share a GPU, and reference runs were
  collected earlier; these timings are not a controlled wall-clock benchmark.

## Validation, failure policy and launch controls

CPU tests and four 600-step CPU smokes run before queue registration, one thread,
low scheduling priority, CUDA hidden. Actual GPU smoke and 100k stability tests
wait for all 20 reference runs, successful final analysis, verified source/config
provenance and an idle GPU. Software versions are checked against every reference
execution claim. No shared-environment package changes are performed.

Finite losses, TD errors, priorities, targets and alpha are mandatory. Invalid
values fail the run; no replacement with zero/uniform priorities. Validation
checks sampling/IS contracts and serialization, not favorable success rates.
Failed gates stop the queue. A production failure stops scheduling new jobs;
already active jobs can finish and all failed records remain visible.

Hardware interruption and learning non-finites are recorded separately. No
automatic checkpoint continuation or fresh rerun is authorized. Any later
recovery requires a separate explicit decision with original attempts retained.

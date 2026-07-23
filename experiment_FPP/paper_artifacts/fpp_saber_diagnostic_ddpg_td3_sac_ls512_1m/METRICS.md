# Diagnostic metric contract

This document defines the quantities emitted by `code/rebuild_results.py`.
The definitions apply separately to DDPG, TD3, and SAC; cross-backbone pooling
is forbidden. Cross-backbone pooling is forbidden.

## Notation and critic families

For backbone \(a\), seed \(s\), diagnostic update \(b\), and minibatch sample
\(i\):

- \(B=256\) is the full diagnostic minibatch size;
- \(\psi_{sbi}\in\{0,1\}\) is the SABER support-membership indicator;
- \(\Omega_{sbi}>0\) is the mean-one normalized critic weight;
- \(y_{sbi}\) is the TD target captured before the critic update and held
  fixed for the post-update diagnostic;
- \(d^{\mathrm{pre}}_{sbi}\) and \(d^{\mathrm{post}}_{sbi}\) are absolute TD
  residuals before and immediately after that one critic update.

DDPG is a single-critic method:

\[
d_{sbi}=\lvert y_{sbi}-Q(o_{sbi},a_{sbi})\rvert.
\]

TD3 and SAC are twin-critic methods. Their common paper-facing residual is the
arithmetic mean of the two absolute critic residuals:

\[
d_{sbi}=\frac{1}{2}\left(
\lvert y_{sbi}-Q_1(o_{sbi},a_{sbi})\rvert+
\lvert y_{sbi}-Q_2(o_{sbi},a_{sbi})\rvert
\right).
\]

This is not a minimum, maximum, or absolute residual of an averaged Q-value.
The two individual critic means remain available in
`results/diagnostic_by_step.csv`.

## Diagnostic support-positive rate

For one diagnostic minibatch,

\[
r_{sb}=\frac{1}{B}\sum_{i=1}^{B}\psi_{sbi}.
\]

The per-seed diagnostic support-positive rate is sample-weighted over the 100
diagnostic minibatches:

\[
r_s=\frac{\sum_b\sum_i\psi_{sbi}}{\sum_b B}.
\]

All batches have size 256 in this frozen artifact, but the implementation
retains the count-weighted definition. Values are written to
`results/support_rate.csv`.

This diagnostic-window rate must not be confused with the final-checkpoint
training support rate stored in `summary.json` or the support-rate series in
`metrics_selected_10k.csv`. The diagnostic-window support rate and the
final-checkpoint support rate are distinct quantities. The diagnostic rate is computed from the exact
full minibatches instrumented every 10,000 steps; the training metric has a
different logging and aggregation grain.

## TD-error association and enrichment

Pearson and Spearman association are computed within each full minibatch
between \(\psi_{sbi}\) and \(d^{\mathrm{pre}}_{sbi}\). Spearman correlation
uses average ranks for ties. The per-seed value is the equal-update mean of
the valid batch correlations.

For top fraction \(q\in\{0.10,0.20\}\), define

\[
k_q=\lceil qB\rceil,
\]

so the frozen batch size gives \(k_{0.10}=26\) and \(k_{0.20}=52\). Let
\(\mathcal{T}_{sbq}\) be the indices of the \(k_q\) largest pre-update absolute
TD residuals in a batch. Stable descending sorting supplies deterministic tie
handling. The top-k support overlap is

\[
O_{sbq}=
\frac{1}{k_q}\sum_{i\in\mathcal{T}_{sbq}}\psi_{sbi}.
\]

The TD-error enrichment is

\[
E_{sbq}=\frac{O_{sbq}}{r_{sb}}.
\]

Thus \(E=1\) is the batch support-rate baseline, \(E>1\) denotes support
enrichment among the highest-residual samples, and \(E<1\) denotes depletion.
Overlap and enrichment are calculated per batch and then averaged with equal
weight over valid updates within a seed. The frozen data have no undefined
primary enrichment rows. Results are in
`results/td_error_enrichment.csv`.

## Immediate residual reduction

For each sample,

\[
\Delta_{sbi} =
d^{\mathrm{pre}}_{sbi}-d^{\mathrm{post}}_{sbi}.
\]

Equivalently, residual reduction is `|δ_before| - |δ_after|`.

Positive \(\Delta\) means that the absolute TD residual decreased during the
one observed update. Support-positive and support-negative summaries are
computed using \(\psi=1\) and \(\psi=0\), respectively.

Two secondary normalizations are retained:

\[
\Delta^{/\Omega}_{sbi}
=\frac{\Delta_{sbi}}{\max(\Omega_{sbi},10^{-8})},
\qquad
R_{sbi}
=\frac{d^{\mathrm{post}}_{sbi}}
       {\max(d^{\mathrm{pre}}_{sbi},10^{-8})}.
\]

The focused table `results/residual_reduction.csv` includes pre-update
residual, post-update residual, reduction, weight-normalized reduction, and
post/pre ratio for each support group and aggregation rule.

Residual reduction is a same-minibatch, in-sample, immediate one-update
measurement with a target fixed from the pre-update computation. It is not a
held-out metric, a generalization estimate, a long-horizon learning effect, or
causal evidence that support weighting improves policy performance.

## Within-seed aggregation

The artifact exposes two aggregation rules.

For a batch-level group mean \(m_{sbg}\), where \(g\) is support-positive or
support-negative, equal-update aggregation is

\[
\bar m^{\mathrm{EU}}_{sg}
=\frac{1}{|\mathcal{V}_{sg}|}
\sum_{b\in\mathcal{V}_{sg}}m_{sbg},
\]

where \(\mathcal{V}_{sg}\) contains batches with a nonempty group. Each valid
diagnostic update receives equal weight. This is the primary within-seed
aggregation for correlation, top-k enrichment, and the headline residual
reductions.

Sample-weighted aggregation is

\[
\bar m^{\mathrm{SW}}_{sg}
=\frac{\sum_{b\in\mathcal{V}_{sg}} n_{sbg}m_{sbg}}
       {\sum_{b\in\mathcal{V}_{sg}} n_{sbg}},
\]

where \(n_{sbg}\) is the number of group-\(g\) samples in that minibatch. This
equals pooling the full-batch observations within one seed while retaining
the seed as the eventual statistical unit. Both variants are reported for
the residual quantities. The diagnostic support-positive rate uses its
sample-weighted count definition.

## Between-seed aggregation

Analysis is seed-first. For a per-seed value \(x_s\), the backbone-level mean
and sample standard deviation are

\[
\bar x=\frac{1}{5}\sum_{s=1}^{5}x_s,
\qquad
s_x=\sqrt{\frac{\sum_{s=1}^{5}(x_s-\bar x)^2}{5-1}}.
\]

The 384,000 minibatch observations are not treated as independent replicates.
No DDPG/TD3/SAC cross-backbone grand mean is calculated. Backbone-level
aggregates are written to `results/aggregate_by_backbone.csv`, and the
underlying seed values are in `results/per_seed_diagnostics.csv`.

## Interpretation rules

- Enrichment is dimensionless and should be read against its within-batch
  baseline of one.
- Absolute TD residuals and reductions depend on the algorithm's target,
  critic family, and learned value scale. DDPG, TD3, and SAC residual
  magnitudes should not be compared as though they shared one measurement
  scale.
- The appropriate mechanism statement is descriptive: support-positive
  samples are enriched among high-TD-error samples and show larger immediate
  in-sample residual correction in these frozen runs.
- The diagnostics do not replace the separately reported task-success
  experiments and do not establish that the observed association causes
  performance differences.

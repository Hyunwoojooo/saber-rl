# FPP B0 vs B3 Critic-Only Runtime-Opt v4 Deep Analysis

Date: 2026-05-29

Source files:
- `experiment_FPP/results/b0_vs_b3_runtime_opt_v4_summary.csv`
- `experiment_FPP/runs/B0/*/metrics_selected_10k.csv`
- `experiment_FPP/runs/b3_critic_only_runtime_opt_v4/*/metrics_selected_10k.csv`
- local raw metrics under `SABER/FetchPickAndPlace/runs/*/metrics.csv` for deeper diagnostics

## 1. Executive conclusion

The current result is not paper-dead, but it cannot be sold as wall-clock efficient yet.

The strongest positive signal is not final single-row success. It is:

1. B3 v4 reaches success thresholds in fewer environment steps than B0.
2. B3 v4 has higher mean eval AUC than B0.
3. B3 v4 has much better late-window stability when measured over the last 50k to 300k steps, rather than the final noisy 20-episode eval only.

The biggest failure mode is:

1. B3 v4 is about 1.65x slower wall-clock than B0.
2. The added wall-clock cost is almost exactly explained by `time/local_context_lookup`.
3. Current support lookup is still too expensive for a clean main result.

Therefore the next paper-worthy direction should be:

- Keep the support-weighting idea, because it improves sample efficiency and late stability.
- Do not claim runtime improvement yet.
- Attack lookup cost aggressively, and tune support strength/schedule so that the B3 advantage survives with less overhead.

## 2. Main aggregate numbers

| metric | B0 | B3 critic-only v4 | interpretation |
|---|---:|---:|---|
| final eval success, single final eval | 0.94 +/- 0.042 | 0.92 +/- 0.091 | B3 looks slightly worse if judged only by final row |
| best eval success | 1.00 | 1.00 | both can solve the task |
| eval AUC success | 0.539 +/- 0.125 | 0.592 +/- 0.044 | B3 is better in sample efficiency / learning curve area |
| final steps/s | 65.02 | 39.51 | B3 is 0.61x B0 speed |
| mean runtime | 4.27 h | 7.03 h | B3 adds about 2.76 h |
| final local lookup time | 0.0 s | 0.0099 s | lookup explains the extra wall-clock |
| final support rate | 0.0 | 0.441 | about 44 percent of critic samples receive support weight |
| final mean raw weight | 1.0 | 1.441 | current lambda=1 makes supported samples 2x raw weight |

Important correction: the final single eval is noisy because each eval has only 20 episodes. Tail-window averages are more reliable.

## 3. Tail-window success is the hidden positive signal

Mean eval success over the last part of training:

| tail window | B0 mean +/- sd | B3 v4 mean +/- sd | readout |
|---|---:|---:|---|
| last 50k | 0.944 +/- 0.043 | 0.948 +/- 0.018 | B3 slightly higher, much lower variance |
| last 100k | 0.940 +/- 0.034 | 0.960 +/- 0.009 | B3 better and more stable |
| last 200k | 0.921 +/- 0.088 | 0.956 +/- 0.004 | B3 much more stable |
| last 300k | 0.906 +/- 0.107 | 0.950 +/- 0.007 | B3 late training is much more stable |

This is the strongest current result. The final-row mean hides it because seed 1 and seed 2 ended with low final eval rows, but their tail averages are good.

For a paper, do not emphasize only final checkpoint success. Report:

- final 100k mean success,
- final 200k mean success,
- AUC,
- time-to-threshold in environment steps,
- seed variance.

## 4. Sample efficiency: B3 reaches thresholds earlier in environment steps

Mean first step reaching eval success threshold:

| threshold | B0 mean step | B3 v4 mean step | B3 gain |
|---|---:|---:|---:|
| 0.5 | 384k | 318k | 66k earlier |
| 0.8 | 542k | 478k | 64k earlier |
| 0.9 | 642k | 576k | 66k earlier |
| 1.0 | 702k | 618k | 84k earlier |

This supports a sample-efficiency claim.

But wall-clock threshold time is worse because lookup slows B3:

| threshold | B0 mean wall-clock | B3 v4 mean wall-clock |
|---|---:|---:|
| 0.5 | 1.64 h | 2.24 h |
| 0.8 | 2.32 h | 3.36 h |
| 0.9 | 2.74 h | 4.05 h |
| 1.0 | 3.00 h | 4.34 h |

So the current result is sample-efficient but not compute-efficient.

## 5. Runtime decomposition

B0 final speed is about 65 steps/s, so one environment step costs about 15.4 ms.
B3 final speed is about 39.5 steps/s, so one environment step costs about 25.3 ms.
The difference is about 9.9 ms/step.

B3 final  is about 9.9 ms/update.

That means almost the entire runtime gap is local support lookup.
The actor/critic torch update itself is not slower. In raw metrics:

| window | B0 actor_critic_update | B3 actor_critic_update |
|---|---:|---:|
| mid | 0.00951 s | 0.00760 s |
| late | 0.00958 s | 0.00772 s |

The optimization target is clear: eliminate or amortize local lookup.

## 6. What the support mechanism is doing

B3 v4 support and weight behavior:

| window | support rate | fallback rate | mean raw weight | ESS normalized |
|---|---:|---:|---:|---:|
| early | 0.384 | 0.616 | 1.384 | 0.879 |
| mid | 0.502 | 0.498 | 1.502 | 0.901 |
| late | 0.469 | 0.531 | 1.469 | 0.897 |

The support weight is not collapsing. ESS stays around 0.90. This is not a priority-collapse failure.

Across the whole learning curve, support rate correlates positively with eval success:

- corr(`lite/psi_q_rank_rate`, eval success) = +0.50
- corr(`lite/q_candidate_fallback_rate`, eval success) = -0.50

But across final seeds only, higher final support rate correlates negatively with final eval success:

- corr(final support rate, final success) = -0.88
- corr(final mean weight, final success) = -0.88

Interpretation:

- Support helps the agent climb early/mid training.
- Late in training, constant lambda=1 may be too blunt and can over-regularize or over-emphasize stale local support.
- The next algorithmic move should be support scheduling or confidence calibration, not simply more support.

## 7. Phase-level diagnosis

Weighted phase support rates:

| window | approach share / support | grasp share / support | lift share / support | place share / support |
|---|---:|---:|---:|---:|
| early | 0.912 / 0.378 | 0.077 / 0.534 | 0.009 / 0.414 | 0.002 / 0.226 |
| mid | 0.709 / 0.459 | 0.189 / 0.642 | 0.039 / 0.586 | 0.063 / 0.518 |
| late | 0.445 / 0.473 | 0.247 / 0.572 | 0.021 / 0.515 | 0.287 / 0.371 |

Late training has many `place` phase samples, but place has the lowest support rate.
Grasp is heavily supported. Place is under-supported.

This is probably where the final stability gap comes from. For PickAndPlace, final success depends on precise placement. If B3 mainly boosts grasp-like transitions but leaves place under-supported, it can improve learning speed while not fully locking final success.

## 8. Lookup diagnostics

B3 v4 runtime lookup behavior:

| window | full context scan rate | predicate early exit | exact negative short-circuit | support cache hit | local scan cache hit |
|---|---:|---:|---:|---:|---:|
| early | 0.425 | 0.390 | 0.128 | 0.090 | 0.996 |
| mid | 0.434 | 0.502 | 0.051 | 0.023 | 0.993 |
| late | 0.469 | 0.469 | 0.051 | 0.020 | 0.993 |

Local scan cache works. Support cache barely works because HER goal + policy bin makes exact support queries almost unique.
Batch coalescing also barely works:

- batch query unique rate is about 0.999 in mid/late.
- coalescing saving rate is near 0.

So cache/coalescing is not enough. The path needs structural reduction.

## 9. local_search_max=1024 is likely overkill for FPP

Late window:

- mean coarse candidates: about 5608
- mean scanned candidates: about 642
- full scan filter count when not early-exited: about 981
- mean local support count: about 51
- local support p50: about 33
- M_Q_min = 4
- N_local_min = 4
- M_B = 16

The support thresholds are tiny compared with observed local counts. This means `local_search_max=1024` is probably doing far more work than necessary.

A 512 scan may still leave the median local count well above 4, while roughly halving the worst part of the lookup. It changes bounded-scan semantics and should be reported as an ablation, but it is one of the most practical immediate levers.

## 10. Best next experiments

### Priority 1: B3 v4 with local_search_max=512

Reason:

- Current support local counts are far above thresholds.
- Runtime is dominated by scanning.
- This is the fastest way to test whether sample-efficiency survives with lower lookup cost.

Expected outcome:

- steps/s should improve substantially.
- support rate may drop slightly but probably not catastrophically.
- if tail success remains near 0.95 to 0.96, this is the main practical variant.

Run first as 3 seeds x 300k or 500k, then 5 seeds x 1M if promising.

### Priority 2: lambda support schedule

Current constant `lambda_support=1.0` gives 2x raw weight to 40 to 50 percent of samples.
This helps learning but may be too strong late.

Test:

- `lambda_support=0.5` constant
- linear decay: 1.0 until 600k, then decay to 0.25 by 1M
- success-adaptive decay: when rolling eval success > 0.9, reduce lambda to 0.25 or 0.5

Hypothesis:

- preserve B3's early threshold gains,
- reduce late over-weighting,
- improve final single-row and tail stability.

### Priority 3: phase-balanced support weighting

Late place phase has high sample share but low support rate.
Grasp gets the strongest support.

A new variant can make the paper story stronger:

- `lambda_phase`: place/lift support gets higher lambda or easier support confidence threshold.
- Or support budget is balanced by phase: normalize support weights so phase-level high-weight mass is not dominated by grasp.

This is not a runtime patch. It is a real algorithmic improvement direction.

Paper framing:

- B3/SABER is not just prioritizing transitions.
- It selectively reinforces locally supported action-context pairs.
- Phase imbalance diagnostics reveal where support is missing.
- A phase-balanced extension targets compositional manipulation bottlenecks.

### Priority 4: support confidence instead of binary support

Current weight is binary:

- unsupported: 1
- supported: 2 before batch normalization

But local support counts are often much larger than thresholds. Binary support discards this information.

Try continuous confidence:

- confidence = clipped log/local count score
- include action-bin frequency among local matches
- use `omega = 1 + lambda * confidence`, normalized by batch mean

This can reduce noise from barely-supported samples and distinguish strong local evidence from weak evidence.

### Priority 5: support update interval / stale support reuse

Since B3 improves sample efficiency but loses wall-clock, compute support less often:

- update support weights every K gradient updates,
- reuse last support label for replay indices within a short replay-version window,
- or apply support lookup only to a random fraction of batch rows.

This changes the training approximation and needs correctness framing, but it directly attacks wall-clock.

## 11. What not to optimize first

Do not spend much more time on exact support-cache hit rate. It is structurally low because the key includes HER goal and policy bin.

Do not rely on batch query coalescing. The batch query unique rate is about 0.999.

Do not treat final single eval as the only success metric. It is noisy and hides the strongest B3 signal.

Do not increase lambda_support above 1.0. The evidence points toward late over-weighting, not under-weighting.

## 12. Proposed paper angle

Current defensible claim:

> Local support-shaped critic updates improve sample efficiency and late-training stability on FetchPickAndPlace, but naive exact support lookup is computationally expensive.

To make this paper-worthy, the next result must show either:

1. Same or better sample efficiency with acceptable wall-clock using `local_search_max=512` or support subsampling, or
2. Clearer final/tail success gain using lambda schedule or phase-balanced support, even if runtime remains higher.

The most promising route is a combined variant:

- `local_search_max=512`
- `lambda_support` decay after success reaches 0.9
- phase-balanced support mass, especially boosting place/lift or preventing grasp from dominating

This would directly target all observed failure modes:

- runtime: too much scan work
- final stability: too much constant late support
- manipulation bottleneck: place phase under-supported

## 13. Immediate ablation matrix

Minimal next set:

| variant | purpose | seeds/steps |
|---|---|---|
| B3 v4 ls512 | runtime rescue | 3 seeds x 300k, then 5 x 1M if good |
| B3 v4 lambda0.5 | reduce late over-weighting | 3 seeds x 500k |
| B3 v4 lambda decay | preserve early boost, reduce late harm | 3 seeds x 500k |
| B3 v4 ls512 + lambda decay | combined practical variant | 5 seeds x 1M |
| phase-balanced B3 | algorithmic/paper novelty | smoke then 3 seeds x 500k |

Primary metrics for the next report:

- tail100k success mean/std
- AUC success
- first step to 0.8/0.9/1.0
- steps/s and wall-clock to threshold
- support rate by phase
- final support rate vs final/tail success
- lookup time per update

## 14. Reproducible analysis artifacts

The report is backed by generated tables in `experiment_FPP/results/`:

- `fpp_b0_b3_v4_method_summary.csv`: method-level B0/B3 aggregate.
- `fpp_b0_b3_v4_tail_success.csv`: final 50k/100k/200k/300k success windows.
- `fpp_b0_b3_v4_thresholds.csv`: first-hit step and wall-clock-to-threshold.
- `fpp_b0_b3_v4_paired_seed_deltas.csv`: per-seed B3 minus B0 deltas.
- `fpp_b0_b3_v4_paired_effects.csv`: paired mean/std/t/effect-size summaries.
- `fpp_b0_b3_v4_b3_raw_windows.csv`: raw-metric B3 runtime/support windows, if local raw metrics exist.
- `fpp_b0_b3_v4_phase_support.csv`: phase-weighted support diagnostics, if local raw metrics exist.
- `fpp_b0_b3_v4_deep_analysis_tables.json`: all generated tables in one JSON file.

The script that regenerates these files is:

```bash
python experiment_FPP/shared/analyze_b0_b3_v4_deep.py
```

The committed `metrics_selected_10k.csv` files are enough to regenerate the main method/tail/threshold/paired tables. The phase and detailed runtime-window tables require local raw `metrics.csv` files, because the raw files were intentionally not committed due size.

## 15. Statistical caution

There are only five seeds. Treat the statistics as decision support, not final proof.

Paired B3-minus-B0 effects from the generated table:

| metric | paired mean | interpretation |
|---|---:|---|
| final success | -0.020 | final single eval is not a B3 win |
| AUC | +0.053 | B3 has a positive sample-efficiency signal |
| tail100k success | +0.020 | B3 has a late-stability signal |
| tail200k success | +0.0345 | B3 has a stronger late-stability signal |
| runtime hours | +2.758 | B3 is clearly too slow |
| steps/s ratio | 0.608 | B3 runs at only about 61 percent of B0 speed |

The right interpretation is therefore:

- The current method has a real learning-curve signal.
- The current method does not yet have a clean wall-clock story.
- The next paper-enabling experiment must preserve AUC/tail-window gains while cutting lookup cost.

## 16. Decision criteria for the next variant

For `local_search_max=512`, continue only if a 3-seed 300k/500k run satisfies:

- steps/s improves by at least 20 percent over B3 v4 1024,
- support rate does not collapse below about 0.30 after 300k,
- first-hit 0.8 is no worse than B0,
- tail-window success remains within about 0.03 of B3 v4 1024.

For `lambda_support` scheduling, continue only if:

- AUC does not drop materially versus B3 v4,
- final/tail seed variance improves,
- final support rate remains useful but late mean raw weight moves closer to 1.2-1.3 rather than 1.45.

For phase-balanced support, continue only if:

- late `place` support rate increases materially,
- grasp does not dominate the high-weight mass,
- final/tail success improves without increasing lookup time.


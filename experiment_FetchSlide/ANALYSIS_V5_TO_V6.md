# FetchSlide V5 to V6 Analysis

Date: 2026-05-29

This note records the current evidence for the next FetchSlide SABER variant.
V5 result artifacts are tracked in this branch. The B0 aggregate numbers below
come from the completed local FetchSlide B0 analysis used during this iteration;
push the B0 artifacts separately if the GitHub branch must be fully
self-contained.

## Bottom Line

The completed FetchSlide V5 runs are not a publishable final method by raw
score or runtime:

| Method | Final Success | AUC Success | Runtime | Steps/s |
| --- | ---: | ---: | ---: | ---: |
| B0 DDPG+HER | 0.600 +/- 0.154 | 0.313 +/- 0.014 | 2.33 h | 119.1 |
| V5 runtime opt | 0.590 +/- 0.114 | 0.301 +/- 0.028 | 12.13 h | 22.9 |

The salvage signal is not final success. V5 is useful earlier in training:

| Step | B0 Success | V5 Success | V5 - B0 |
| ---: | ---: | ---: | ---: |
| 300k | 0.16 | 0.22 | +0.06 |
| 400k | 0.33 | 0.26 | -0.07 |
| 500k | 0.33 | 0.35 | +0.02 |
| 900k | 0.59 | 0.38 | -0.21 |
| 1M | 0.60 | 0.59 | -0.01 |

At 300k, V5 also improves task mechanics:

| Metric at 300k | B0 | V5 |
| --- | ---: | ---: |
| object-goal distance mean | 0.307 | 0.258 |
| object-goal distance p90 | 0.604 | 0.498 |
| strike/contact-like rate | 0.587 | 0.670 |
| progress rate | 0.828 | 0.899 |
| overshoot rate | 0.0106 | 0.0000 |

Interpretation: current SABER helps acquire contact/progress, but hurts or
fails to help the late precision-refinement phase.

## Failure Mode

Current B3/SABER support-positive means that a local context/action-bin is
represented in replay. It does not mean the action is good for solving
FetchSlide. In FetchSlide, common approach/strike transitions can be frequent
without being the transitions that finish the slide.

The final V5 phase support distribution shows the mismatch:

| Phase | Count Rate | Support Rate | Mean Weight |
| --- | ---: | ---: | ---: |
| approach | 0.298 | 0.429 | 1.429 |
| strike | 0.283 | 0.324 | 1.324 |
| slide | 0.015 | 0.300 | 1.300 |
| near_goal | 0.405 | 0.052 | 1.052 |

Near-goal is a large part of late training, but current support weighting
barely upweights it. The signal mostly emphasizes approach and strike, which
explains why V5 improves early contact/progress but does not dominate final
success.

Runtime is also a separate failure mode. V5 remains about 5.2x slower than B0
because most updates still scan almost the full `local_search_max=1024`
candidate set for each batch item. Cache hit rates are too low to change the
algorithmic cost.

## V6 Hypothesis

Use SABER as an early curriculum signal, then remove it before late precision
learning:

```text
0 - 300k:
  lambda_support = 1.0
  keep V5 support weighting

300k - 500k:
  linearly anneal lambda_support from 1.0 to 0.0

500k - 1M:
  skip support lookup entirely
  use uniform B0-style critic weights
```

This keeps the best observed V5 window and avoids the slow/weak late V5
regime. Estimated 1M runtime is about 4.2h if the method uses V5-like runtime
to 300k and B0-like runtime after that. This is still slower than B0, but no
longer prohibitively slow.

## Implementation

The new method name is:

```text
b3_critic_only_runtime_opt_v6_anneal_300k
```

The implementation lives in:

```text
experiment_FetchSlide/src/saber_fetch_slide/experiment.py
```

Key runtime metrics to inspect:

```text
runtime/support_lambda_effective
runtime/support_lookup_enabled
runtime/support_lookup_skipped
runtime/support_anneal_progress
time/local_context_lookup
runtime/steps_per_sec
eval/success_mean
fetch_slide/object_goal_dist_mean
fetch_slide/strike_contact_like_rate
fetch_slide/progress_rate
phase/near_goal/count_rate
phase/near_goal/support_rate
```

Acceptance criterion for V6 is not just final success. The intended win is:

1. Keep or improve the V5 300k contact/progress advantage.
2. Recover B0-like late precision learning after support is annealed away.
3. Reduce wall-clock runtime substantially versus V5.
4. Avoid a final success/AUC regression versus B0.

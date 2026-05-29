# Progress-Gated SABER Proposal

Date: 2026-05-29

## 1. Motivation

Current FetchSlide results show that naive SABER / V5 is not strong enough as a
final performance method.

```text
B0 final success: 0.600
V5 final success: 0.590
V5 runtime: about 5.2x slower than B0
```

So the claim should not be:

```text
Naive SABER improves FetchSlide final success.
```

However, V5 is not useless. It improves early object transport quality:

```text
300k object-goal distance mean: V5 better in 5/5 seeds
300k object-goal distance p90:  V5 better in 5/5 seeds
475k success bump:              V5 better in 5/5 seeds
```

The useful signal is therefore not final success. The useful signal is:

```text
Naive SABER helps the agent discover object-transport behavior.
```

The failure is:

```text
Naive SABER does not reliably turn object transport into final precision success.
```

## 2. Core Diagnosis

The problem is that support is not the same as utility.

```text
support:
  This behavior appeared often enough in local replay.

utility:
  This behavior actually helps solve the task.
```

Current SABER mostly asks:

```text
Was this action bin available in the local replay context?
```

For FetchSlide, that is not enough.

Example:

```text
The agent hits the object.
The object moves.
But it moves away from the goal.

This behavior can still be replay-supported.
But it is not useful.
```

So naive SABER can reinforce frequent behavior even when that behavior is not
task-useful.

## 3. Existing Failure Pattern

V5 phase support at the end of training:

```text
approach support:  0.429
strike support:    0.324
slide support:     0.300
near_goal support: 0.052
```

This means:

```text
approach / strike:
  frequently supported
  get relatively high weight

near_goal:
  rarely supported
  gets almost no extra weight
```

But final FetchSlide success depends heavily on near-goal precision and
stability. This explains the observed pattern:

```text
V5 improves early object transport.
V5 does not improve final precision success.
```

## 4. Proposed Direction

The next method should not simply make support lookup faster. It should change
which supported transitions receive extra weight.

The proposed direction is:

```text
Progress-Gated SABER
```

Minimal formula:

```text
psi_useful_i = psi_support_i * psi_progress_i
w_i = 1 + lambda * psi_useful_i
```

Where:

```text
psi_support_i:
  Existing SABER support-positive predicate.

psi_progress_i:
  A task-progress or task-quality predicate.
```

Plain-language summary:

```text
Old SABER:
  Trust behavior that was replay-supported.

Progress-Gated SABER:
  Trust behavior that was replay-supported and actually helped the task.
```

## 5. Minimal FetchSlide Version

The simplest first implementation should only use object-goal progress.

```text
d_t   = ||object_t - goal||
d_tp1 = ||object_tp1 - goal||

progress_delta = d_t - d_tp1
psi_progress = 1[progress_delta > eps_progress]
```

Then:

```text
w_i = 1 + lambda * psi_support_i * psi_progress_i
```

This version is intentionally simple. Do not add phase-aware gates, schedule,
or near-goal precision logic in the first test. The goal is to check whether
progress gating preserves the useful part of V5.

## 6. Phase-Wise Quality Support

If the minimal progress-gated version is promising, the stronger method should
be phase-wise quality support.

The core idea:

```text
Different phases need different definitions of useful behavior.
```

Suggested phase quality predicates:

```text
approach:
  hand-object distance decreases

strike:
  object speed is above a minimal useful threshold
  object velocity is aligned toward the goal

slide:
  object velocity remains aligned toward the goal
  overshoot risk is low

near_goal:
  object speed is low
  object remains near the goal
  action is stabilizing rather than pushing hard
```

More concrete first phase-wise version:

```text
approach:
  hand_object_dist_delta > eps_hand

strike / slide:
  vel_align > 0
  obj_speed_xy > v_slide_min

near_goal:
  obj_speed_xy < v_stop_max
  object_goal_dist_next < near_goal_threshold
```

Do not start with all possible quality terms. Add them gradually so the result
is interpretable.

## 7. Recommended Experiment Names

Recommended method names:

```text
b3_critic_only_progress_gated
saber_progress_gated
saber_phase_quality_support
saber_slide_quality_support
```

Suggested ordering:

```text
1. b3_critic_only_progress_gated
   Minimal progress gate.

2. saber_phase_quality_support
   Phase-specific quality predicates.

3. saber_phase_quality_support_scheduled
   Phase quality support plus reduced lookup after the useful window.
```

## 8. First Experiment Plan

Do not start with a full 1M seed-5 experiment.

Run:

```text
1. smoke
2. 300k, seed 3
3. if it passes, 1M, seed 5
```

### Smoke Checks

Log and inspect:

```text
lite/progress_positive_rate
lite/support_and_progress_rate
lite/progress_delta_mean
lite/progress_delta_support_mean
lite/mean_w_P
runtime/steps_per_sec
```

### 300k Checks

Main metrics:

```text
300k object-goal distance mean
300k object-goal distance p90
300k progress rate
300k support_and_progress_rate
300k progress_delta_support_mean
runtime/steps_per_sec
```

Kill criteria:

```text
support_and_progress_rate < 0.03
mean_w_P is almost 1.0
300k object-goal distance is worse than B0
runtime is worse than V5
```

Pass criteria:

```text
300k object-goal distance mean/p90 is better than B0
progress_delta_support_mean > 0
support_and_progress_rate is not too sparse
steps/sec is not worse than V5
```

### 1M Checks

Only run 1M if the 300k result passes.

Success criteria:

```text
475k success bump is preserved
900k collapse is reduced
final success >= B0
runtime < V5
```

## 9. Required Logging

Add these diagnostics:

```text
lite/progress_positive_rate
lite/support_and_progress_rate
lite/progress_delta_mean
lite/progress_delta_support_mean
lite/progress_delta_support_p50
lite/progress_delta_support_p90

phase/approach/progress_positive_rate
phase/approach/support_and_progress_rate
phase/approach/mean_progress_delta

phase/strike/progress_positive_rate
phase/strike/support_and_progress_rate
phase/strike/mean_progress_delta

phase/slide/progress_positive_rate
phase/slide/support_and_progress_rate
phase/slide/mean_progress_delta

phase/near_goal/progress_positive_rate
phase/near_goal/support_and_progress_rate
phase/near_goal/mean_progress_delta
```

For phase-wise quality support, add:

```text
phase/*/quality_positive_rate
phase/*/support_and_quality_rate
phase/*/mean_quality_weight
```

## 10. Paper Framing

Strongest framing:

```text
Support-only weighting improves early object transport but fails at final
precision because support is frequency, not utility.
Progress-gated support converts replay coverage into task-useful replay
weighting.
```

Short version:

```text
Support tells us what the agent can repeatedly do.
Progress tells us what is worth reinforcing.
```

Suggested claim:

```text
Local replay support is a feasibility signal. Sparse robotic control also
requires a utility signal. Quality-conditioned local support combines both.
```

## 11. What Not To Do

Do not claim:

```text
Naive SABER improves final FetchSlide success.
```

Do not do next:

```text
Continue runtime optimization only.
Run full 1M seed-5 immediately.
Add progress gate, phase gate, schedule, and precision gate all at once.
```

The next concrete action should be:

```text
Implement minimal b3_critic_only_progress_gated.
Run smoke.
Run 300k seed 3.
Kill or continue based on the diagnostics above.
```

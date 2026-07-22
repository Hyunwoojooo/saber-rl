# FPP DDPG SABER Random-Support (Batch-Wide Count Matched) (1M)

Compact seeds 1--5 artifact for Main versus Random-Support (Batch-Wide Count Matched).

- execution source: `77e496f134ba2a141d90e4b951c75674656d05eb`
- canonical Main source: `c43d40502b84ae713924ebf995be623e7acc7ca8`
- factor: `support_assignment_mode: support -> batch_wide_count_matched`
- query remains `current_policy`; actor weights remain one; critic-only scope
- each update samples exactly Main's positive count uniformly without replacement
  from the full minibatch, including Main-positive and context-ineligible positions
- for binary SABER weights this control is distributionally equivalent to uniformly
  shuffled-weight; the two must not be interpreted as independent mechanisms

Primary normalized success AUC uses only exact observed 10k--1M evaluations,
trapezoidal integration, and denominator 990,000. No smoothing, interpolation,
extrapolation, or forward filling is used. Main minus Random-Support mean AUC is
`+0.056162` with paired-percentile 95% CI
`[0.013687, 0.100859]`
(10,000 resamples, seed 27027); exact sign-flip enumerates 32 assignments.

`data/control` includes every seed's complete non-overlapping 1,000-update
assignment diagnostics. `technical_gate` preserves validated 5k smoke/resume,
50k gate, and source-freeze evidence. Checkpoints, full metrics, JSONL, raw logs,
and raw smoke/gate directories are excluded.

Verify standalone: `python3 code/verify_artifact.py --artifact-root .`

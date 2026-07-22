# FPP DDPG SABER Strict Context-Eligible Count-Matched Control (1M)

Compact seeds 1--5 artifact for Main versus Strict Context-Eligible Count-Matched Control.

- execution source: `1ddf7ba11cb221a9c0a9fca7c18a21cb54591715`
- canonical Main source: `c43d40502b84ae713924ebf995be623e7acc7ca8`
- factor: `support_assignment_mode: support -> context_eligible_count_matched`
- query remains `current_policy`; actor weights remain one; critic-only scope

Primary normalized success AUC uses only exact observed 10k--1M evaluations,
trapezoidal integration, and denominator 990,000. No smoothing, interpolation,
extrapolation, or forward filling is used. Main minus Strict mean AUC is
`+0.032525` with paired-percentile 95% CI
`[-0.006263, 0.075556]`
(10,000 resamples, seed 27027); exact sign-flip enumerates 32 assignments.

`data/control` includes every seed's complete non-overlapping 1,000-update
assignment diagnostics. `technical_gate` preserves validated 5k smoke/resume,
50k gate, and source-freeze evidence. Checkpoints, full metrics, JSONL, raw logs,
and raw smoke/gate directories are excluded.

Verify standalone: `python3 code/verify_artifact.py --artifact-root .`

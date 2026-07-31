# High-RR Execution-Metadata Boundary

All DDPG and TD3 High-RR runs began from scratch at environment step 0 and completed without checkpoint continuation. Some retained TD3 configuration fields and runtime-counter segments use resume-oriented labels; these are stale logging metadata and are not execution history.

- `checkpoint_continuation`: `false`
- `recorded_resume_metadata_stale`: `true`
- `resume_launcher_used_for_production`: `false`

The unused resume launcher is intentionally excluded from this reviewer package. Original retained per-seed configurations are preserved so the metadata boundary remains auditable. Package integrity is established by `SELECTION_MANIFEST.json` and the top-level `MANIFEST.sha256`.

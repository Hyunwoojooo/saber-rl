# FPP TD3/SAC HER Full-PER 1M Artifact

This artifact packages the completed FetchPickAndPlace-v4 sparse-reward
TD3+HER+PER(full) and SAC+HER+PER(full) experiments for seeds 1-5.

| Method | Final success mean +/- std | Success AUC mean +/- std |
|---|---:|---:|
| TD3+HER+PER(full) | 0.660000 +/- 0.204328 | 0.368889 +/- 0.033609 |
| SAC+HER+PER(full) | 0.990000 +/- 0.022361 | 0.674872 +/- 0.042158 |

## Contents

- `data/`: sanitized config, environment snapshot, summary, and compact
  evaluation-only metrics for each of the ten runs.
- `results/`: per-seed, aggregate, paired descriptive results, report, and
  machine-readable validation record.
- `input_manifest.json`: SHA256 provenance for raw inputs and experiment source.
- `artifact_manifest.json`: SHA256 inventory of the packaged artifact.

The runnable source is versioned in the same repository commit under
`experiment_FPP/td3_her_per_full_ls512` and
`experiment_FPP/sac_her_per_full_ls512`. Regenerate from repository root with:

```bash
.venv/bin/python experiment_FPP/package_her_per_full_results.py
```

Raw checkpoints, replay state, full metrics, JSONL, and logs are excluded.
See `results/report.md` for the complete result table and evaluation semantics.

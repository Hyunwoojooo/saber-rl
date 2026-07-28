# FPP TD3 SABER K=3 Results

This compact result package contains the completed K=3 action-bin sensitivity arm
(seeds 1-5, 1M environment steps). K=7 remains pending.

## Aggregate

- AUC: 0.352879 +/- 0.048313
- Final-quarter AUC: 0.603400 +/- 0.120691
- Final success: 0.750000 +/- 0.264575
- Hit90: 2/5; median among reached seeds: 945000
- Support-positive rate: 0.396297 +/- 0.012079

Against Main K=5, the paired AUC difference (K=3 minus K=5) is
-0.089495, with a 95% paired-bootstrap interval of
[-0.108687, -0.070303] and 0/5 positive pairs.

## Contents

- `data/`: sanitized configs, environment snapshots, summaries, and exact-grid
  `metrics_selected_10k.csv` files.
- `results/`: per-seed, aggregate, paired-comparison, and JSON analysis outputs.
- `artifact_manifest.json`: SHA256 and size for every packaged file.

Checkpoints, full per-update metrics, JSONL logs, and run logs are intentionally
excluded. No smoothing, interpolation, extrapolation, or forward filling is used.

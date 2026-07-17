# FPP TD3-AER no-HER, seeds 1-5, 1M steps

This is the compact paper artifact for attention experience replay (AER) with
TD3 in `FetchPickAndPlace-v4`, sparse rewards, and HER disabled. It contains
five paired seeds, the exact 10k-step evaluation grid, AER sampling diagnostics,
fixed 100-episode final evaluations, aggregate tables, and provenance.

## Frozen protocol

- Artifact ID: `fpp_td3_aer_noher_seeds1_5_1m_arm`
- Method: `td3_aer` (cosine top-k attention replay plus AER LR annealing)
- Seeds: 1, 2, 3, 4, 5
- Training: 1,000,000 environment steps
- Evaluation grid: 10,000 through 1,000,000, every 10,000 steps
- Online evaluation: 20 deterministic episodes per grid point
- Fixed final evaluation: 100 deterministic episodes
- HER ratio: 0.0
- Source commit: `8c53c789f6f5cb6a45beceb4462a209cf6100d7e`

Each seed directory under `data/td3_aer/{1..5}/` contains `summary.json`, `config.json`,
`env_snapshot.json`, `final_evaluation_100.json`, `best_checkpoint.json`, and
the reduced `metrics_selected_10k.csv`.  Model checkpoints, full-resolution
metrics, JSONL streams, and logs are intentionally excluded.
SHA256SUMS covers every artifact file except SHA256SUMS itself.

## Verification

From this directory:

```bash
python code/verify_artifact.py .
```

To additionally check the result-producing source against the frozen source
manifest copied into `provenance/`:

```bash
python code/verify_artifact.py . --experiment-source-root /path/to/experiment/source
```

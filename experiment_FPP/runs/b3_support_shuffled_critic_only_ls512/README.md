# FPP DDPG Shuffled-Weight Control

Artifact for the FetchPickAndPlace DDPG shuffled-weight control.

- Task: FetchPickAndPlace
- Backbone: DDPG + HER
- Method: `b3_support_shuffled`
- Weight scope: critic only
- Local search max: 512
- Total steps: 1,000,000
- Seeds: 1-5

Each seed directory contains `summary.json`, `config.json`, `env_snapshot.json`, `run.log`, and `metrics_selected_10k.csv`.
Large checkpoints and raw full-resolution metrics are intentionally excluded from this artifact path.

# FetchSlide B3 Critic-Only Runtime Optimization Artifacts

This directory contains a lightweight snapshot of experiment outputs copied from the active local run directory.

Included:

- `config.json`
- `env_snapshot.json`
- `metrics.csv`
- `metrics.jsonl`
- `run.log`
- `best_checkpoint.json`
- completed-run `summary.json` files when available

Excluded:

- `checkpoint.pt`
- `best_checkpoint.pt`

PyTorch checkpoint files are intentionally excluded because several files are larger than GitHub's normal file-size limit and would require an external artifact store or Git LFS.

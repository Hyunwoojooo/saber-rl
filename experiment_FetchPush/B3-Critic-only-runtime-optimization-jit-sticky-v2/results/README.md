# B3-Critic-only Runtime-Opt (JIT, local_search_max=512)

This directory contains the FetchPush 1M-step, 5-seed results for the main B3/SABER comparison against the B0 CUDA baseline.

Internal snapshot folder:

`B3-Critic-only-runtime-optimization-jit-sticky-v2`

Experiment identity:

- Method: B3 / SABER critic-only
- Runtime optimization: ReplayActionIndex v2, bounded sticky-random support scan, support context cache, local scan cache, FetchPush exact JIT support scanner
- Device: CUDA
- `local_search_max`: 512
- Seeds: 1-5
- Total steps per seed: 1,000,000

Included files per seed:

- `best_checkpoint.json`
- `config.json`
- `env_snapshot.json`
- `metrics.csv`
- `run.log`
- `summary.json`

`metrics.jsonl` is intentionally omitted from Git because each file exceeds GitHub's 100 MB regular-file limit. The corresponding step-wise metrics are retained in `metrics.csv`.

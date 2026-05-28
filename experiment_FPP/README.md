# SABER FetchPickAndPlace Experiments

This source tree is generated from `SABER/SABER.md` for the FetchPickAndPlace benchmark. Each method has its own folder with `config.json`, `run.py`, and `run_all_seeds.sh`.

## Experiment Folders

- `B0`: Clean DDPG+HER baseline with Omega=1.
- `B3-original`: Required actor-critic B3 support-only reference.
- `B3-critic-only`: SABER main candidate: support reliability weighting on critic loss only.
- `B3-critic-only-random`: Positive-count control: random K high-weight critic samples with B3 diagnostics retained.
- `B3-critic-only-shuffled`: Weight-distribution control: shuffle B3 raw weights within batch for critic loss.
- `B3-critic-only-runtime-optimization`: Exact-semantics runtime-optimization path for critic-only B3 support weighting.
- `B3-critic-only-fast-512`: FetchPickAndPlace local_search_max sensitivity run at 512.
- `B3-critic-only-fast-2048`: FetchPickAndPlace local_search_max sensitivity run at 2048.
- `TD-error`: Critic-only TD-error loss-weighting baseline.
- `PER-style`: PER-style TD-error priority-derived critic loss weighting with uniform replay sampling.

Generate every command without running:

```bash
python3 SABER/FetchPickAndPlace/shared/generate_commands.py
```

Analyze completed runs:

```bash
python3 SABER/FetchPickAndPlace/shared/analyze_results.py
```

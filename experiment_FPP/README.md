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
- `td3_her_baseline_ls512`: TD3+HER baseline matched to SABER v4 ls512.
- `td3_saber_critic_only_v4_ls512`: TD3+SABER critic-only runtime optimized ls512.
- `td3_her_per_style_loss_weighting_ls512`: TD3+HER with PER-style TD-error priority-derived critic loss weighting and uniform replay sampling.
- `td3_her_per_full_ls512`: TD3+HER with slot-level full PER sampling and critic-only importance-sampling correction.
- `sac_her_baseline_ls512`: SAC+HER baseline with uniform critic weighting.
- `sac_her_per_full_ls512`: SAC+HER with slot-level full PER sampling and critic-only importance-sampling correction.
- `sac_saber_critic_only_v4_ls512`: SAC+HER+SABER critic-only support weighting with v4 exact Numba lookup.

Generate every command without running:

```bash
python3 SABER/FetchPickAndPlace/shared/generate_commands.py
```

Analyze completed runs:

```bash
python3 SABER/FetchPickAndPlace/shared/analyze_results.py
```

## Packaged Results

- `paper_artifacts/fpp_td3_sac_her_per_full_1m`: TD3 and SAC HER+PER(full),
  seeds 1-5 at 1M steps, with sanitized run summaries and compact evaluation
  curves.
- `paper_artifacts/fpp_td3_her_per_style_loss_weighting_ls512_1m`: TD3+HER
  PER-style, seeds 1-5 at 1M steps, with validated summaries and 10k-step
  evaluation curves.

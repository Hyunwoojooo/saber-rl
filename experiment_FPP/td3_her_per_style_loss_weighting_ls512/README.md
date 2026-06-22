# TD3 HER PER-style Loss Weighting LS512

Purpose: compare TD3+SABER against TD3+HER with PER-style TD-error priority-derived critic loss weighting on FetchPickAndPlace.

This is not full replay sampling PER. Replay sampling remains uniform; TD-error-derived priorities only shape the critic loss weights.

Main settings:

- `method=td3_her_per_style_loss_weighting`
- TD3 target policy smoothing: `td3_target_policy_noise=0.2`, `td3_target_noise_clip=0.5`
- TD3 delayed actor update: `td3_policy_delay=2`
- `weight_scope=critic_only`
- `weighting_mode=batch_norm`
- `local_search_max=512` for run naming and parity; no SABER support lookup is used
- `per_alpha=0.6`
- `eps_PER=1e-6`
- `U_PER_max=2.0`
- seeds `1..5`
- total steps `1_000_000`

Dry-run:

```bash
python3 SABER/FetchPickAndPlace/td3_her_per_style_loss_weighting_ls512/run.py --seed 1 --dry-run
```

Validate setup:

```bash
python3 SABER/FetchPickAndPlace/td3_her_per_style_loss_weighting_ls512/run.py --seed 1 --validate_only --device cuda
```

Smoke:

```bash
python3 SABER/FetchPickAndPlace/td3_her_per_style_loss_weighting_ls512/run.py --seed 1 --steps 5000 --device cuda --smoke
```

Main seed:

```bash
python3 SABER/FetchPickAndPlace/td3_her_per_style_loss_weighting_ls512/run.py --seed 1 --steps 1000000 --device cuda
```

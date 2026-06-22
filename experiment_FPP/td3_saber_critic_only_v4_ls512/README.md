# TD3 SABER Critic-Only V4 LS512

Purpose: TD3 + SABER critic-only runtime optimized support weighting for FetchPickAndPlace.

This experiment uses TD3 plus SABER critic-only support weighting. HER replay remains uniform and SABER support lookup semantics match v4 ls512.

Main settings:

- `method=td3_saber_runtime_opt`
- `weight_scope=critic_only`
- `weighting_mode=batch_norm`
- `local_search_max=512` with v4 SABER runtime support lookup
- `lambda_support=1.0`
- `td3_policy_delay=2`
- `td3_target_policy_noise=0.2`; `td3_target_noise_clip=0.5`
- seeds `1..5`
- total steps `1_000_000`

Dry-run:

```bash
python3 SABER/FetchPickAndPlace/td3_saber_critic_only_v4_ls512/run.py --seed 1 --dry-run
```

Validate setup:

```bash
python3 SABER/FetchPickAndPlace/td3_saber_critic_only_v4_ls512/run.py --seed 1 --validate_only --device cuda
```

Smoke:

```bash
python3 SABER/FetchPickAndPlace/td3_saber_critic_only_v4_ls512/run.py --seed 1 --steps 5000 --device cuda --smoke
```

Main seed:

```bash
python3 SABER/FetchPickAndPlace/td3_saber_critic_only_v4_ls512/run.py --seed 1 --steps 1000000 --device cuda
```

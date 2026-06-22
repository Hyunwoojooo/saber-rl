# TD3 HER Baseline LS512

Purpose: TD3 + HER baseline for FetchPickAndPlace. Support weighting is disabled.

This experiment uses TD3 twin critics, delayed actor update, and target policy smoothing. HER replay remains uniform.

Main settings:

- `method=td3_her`
- `weight_scope=none`
- `weighting_mode=batch_norm`
- `local_search_max=512` for parity with SABER ls512 naming; no support lookup is used
- `td3_policy_delay=2`
- `td3_target_policy_noise=0.2`
- `td3_target_noise_clip=0.5`
- seeds `1..5`
- total steps `1_000_000`

Dry-run:

```bash
python3 SABER/FetchPickAndPlace/td3_her_baseline_ls512/run.py --seed 1 --dry-run
```

Validate setup:

```bash
python3 SABER/FetchPickAndPlace/td3_her_baseline_ls512/run.py --seed 1 --validate_only --device cuda
```

Smoke:

```bash
python3 SABER/FetchPickAndPlace/td3_her_baseline_ls512/run.py --seed 1 --steps 5000 --device cuda --smoke
```

Main seed:

```bash
python3 SABER/FetchPickAndPlace/td3_her_baseline_ls512/run.py --seed 1 --steps 1000000 --device cuda
```

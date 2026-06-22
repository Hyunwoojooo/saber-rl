# SAC HER Baseline LS512

Purpose: SAC + HER baseline for FetchPickAndPlace with uniform critic weighting.

Main settings:

- `method=sac`
- `agent_kind=sac`
- `total_steps=1_000_000`
- `batch_size=256`, `buffer_capacity=300_000`, `her_ratio=0.8`
- `start_steps=10_000`, `warmup_steps=5_000`, `updates_per_step=1`
- `eval_freq=25_000`, `eval_episodes=20`
- `reward_type=sparse`
- `gamma=0.98`, `tau=0.005`
- `actor_lr=3e-4`, `critic_lr=3e-4`, `hidden_dim=256`
- `sac_alpha=0.2`, `sac_auto_alpha=true`, `sac_alpha_lr=3e-4`
- `sac_target_entropy=null`, so the code uses `-action_dim`
- SABER support: disabled; critic weights are uniform
- `local_search_max=512`, `M_B=16`, `M_Q_min=4`, `N_local_min=4`
- `support_diagnostic_audit_interval=0`

Dry-run:

```bash
python3 SABER/FetchPickAndPlace/sac_her_baseline_ls512/run.py --seed 1 --dry-run
```

Validate setup:

```bash
python3 SABER/FetchPickAndPlace/sac_her_baseline_ls512/run.py --seed 1 --validate_only --device cuda
```

Smoke:

```bash
python3 SABER/FetchPickAndPlace/sac_her_baseline_ls512/run.py --seed 1 --steps 6000 --device cuda --output_root /tmp/sac_her_baseline_ls512_debug
```

Main seed:

```bash
python3 SABER/FetchPickAndPlace/sac_her_baseline_ls512/run.py --seed 1 --steps 1000000 --device cuda
```

# b3_critic_only_runtime_opt_v2

Critic-only B3 with runtime optimization v2:

- predicate-only early exit for support-positive/support-negative decisions
- FetchPickAndPlace fast full-context filtering
- local scan cache with FPP feature block
- support cache keyed by HER goal and policy action bin
- exact diagnostic audit pass every `runtime_exact_audit_interval` updates

Existing B3/runtime-opt folders are not modified.

Dry run:

```bash
python3 SABER/FetchPickAndPlace/b3_critic_only_runtime_opt_v2/run.py --seed 1 --dry-run
```

Short run:

```bash
python3 SABER/FetchPickAndPlace/b3_critic_only_runtime_opt_v2/run.py --seed 1 --steps 300000 --device auto
```

All seeds:

```bash
bash SABER/FetchPickAndPlace/b3_critic_only_runtime_opt_v2/run_all_seeds.sh --steps 300000 --device auto
```

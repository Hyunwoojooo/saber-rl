# b3_critic_only_runtime_opt_v3

Lean critic-only B3 runtime optimization for FetchPickAndPlace.

Kept from the old patched runtime path:

- deterministic bounded scan: `deterministic_tail_v1`
- bucket-versioned support context cache
- local scan cache
- batch query coalescing with HER goal in the key
- truncated scan cache support

Kept from v2:

- coarse bucket action-bin count lookup
- exact coarse candidate count short-circuit
- exact `min(coarse_count, M_B) < M_Q_min` short-circuit

Intentionally not included by default:

- feature fast path
- predicate chunk early-exit
- diagnostic audit path
- feature block cache

Dry run:

```bash
python3 SABER/FetchPickAndPlace/b3_critic_only_runtime_opt_v3/run.py --seed 1 --dry-run
```

Smoke run:

```bash
python3 SABER/FetchPickAndPlace/b3_critic_only_runtime_opt_v3/run.py --seed 1 --smoke --device auto
```

All seeds:

```bash
bash SABER/FetchPickAndPlace/b3_critic_only_runtime_opt_v3/run_all_seeds.sh --device auto
```

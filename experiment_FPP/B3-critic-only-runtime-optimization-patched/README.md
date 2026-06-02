# B3-critic-only-runtime-optimization-patched

Critic-only B3 with the patched runtime optimization path:

- deterministic bounded scan: `deterministic_tail_v1`
- bucket-versioned support context cache
- local scan cache
- batch query coalescing with HER goal in the key
- exact-negative short-circuit without caching policy-dependent negative results

This folder contains an independent `code/src` copy. Existing B3/runtime-opt sources are not modified.

Dry run:

```bash
python3 SABER/FetchPickAndPlace/B3-critic-only-runtime-optimization-patched/run.py --seed 1 --dry-run
```

Smoke run:

```bash
python3 SABER/FetchPickAndPlace/B3-critic-only-runtime-optimization-patched/run.py --seed 1 --smoke --device auto
```

All seeds:

```bash
bash SABER/FetchPickAndPlace/B3-critic-only-runtime-optimization-patched/run_all_seeds.sh --device auto
```

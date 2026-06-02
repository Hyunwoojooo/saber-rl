# b3_critic_only_runtime_opt_v4_ls512_lambda_schedule

Ablation A: critic-only B3 runtime optimization v4 for FetchPickAndPlace with `local_search_max=512`. All other v4 support/runtime settings are unchanged.

Only changed from `b3_critic_only_runtime_opt_v4`:

- `local_search_max`: `1024 -> 512`
- `default_local_search_max`: `1024 -> 512`
- `run_slug` / `output_root` use `b3_critic_only_runtime_opt_v4_ls512_lambda_schedule`

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
- Python predicate chunk early-exit
- feature block cache

Dry run:

```bash
python3 SABER/FetchPickAndPlace/b3_critic_only_runtime_opt_v4_ls512_lambda_schedule/run.py --seed 1 --dry-run
```

Smoke run:

```bash
python3 SABER/FetchPickAndPlace/b3_critic_only_runtime_opt_v4_ls512_lambda_schedule/run.py --seed 1 --smoke --device auto
```

3-seed screening run in parallel:

```bash
STEPS=300000 DEVICE=cuda bash SABER/FetchPickAndPlace/b3_critic_only_runtime_opt_v4_ls512_lambda_schedule/run_screening_3seeds_parallel.sh
```

Custom seeds for screening:

```bash
SEED_LIST="1 2 3" STEPS=500000 DEVICE=cuda bash SABER/FetchPickAndPlace/b3_critic_only_runtime_opt_v4_ls512_lambda_schedule/run_screening_3seeds_parallel.sh
```

All seeds:

```bash
bash SABER/FetchPickAndPlace/b3_critic_only_runtime_opt_v4_ls512_lambda_schedule/run_all_seeds.sh --device auto
```

Added in v4:

- exact FetchPickAndPlace Numba scanner for full-context support lookup
- no Q/TD/loss/gradient/neural-network caching
- same support semantics as v3, with legacy fallback if Numba is unavailable
- predicate-only early exit inside the Numba scanner for training support decisions
- exact per-update diagnostics are replaced by periodic audit metrics when predicate early exit fires
- runtime metrics: `runtime/numba_support_scan_enabled`, `runtime/numba_support_scan_fallback_rate`, `runtime/numba_support_queries`, `runtime/predicate_early_exit_rate`, `runtime/diagnostic_audit_rate`, `time/numba_support_scan`

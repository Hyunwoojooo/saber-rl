# B3-Critic-only-runtime-optimization-jit-sticky-v2

This is an isolated code snapshot for the FetchPush SABER/B3 critic-only
runtime-optimized JIT sticky-v2 experiment.

It corresponds to the code used by the current running smoke run:

```text
fetch_push_b3_critic_only_runtime_opt_jit_sticky_random_v2_s1_200000_ls512_cuda_smoke
```

Included runtime optimization components:

- ReplayActionIndex v2
- direct bounded local replay scan
- `support_scan_mode = sticky_random_v1`
- deterministic truncated scan cache
- versioned support context cache
- local scan cache
- incremental coarse action-bin metadata
- exact negative short-circuit enabled
- batch query coalescing disabled
- FetchPush exact Numba support scanner enabled

This snapshot intentionally contains code and launch metadata only. It does not
include smoke results, logs, or checkpoints.

Run command:

```bat
cmd /c call version5.0\metacog-lite-v0\experiment_FetchPush\B3-Critic-only-runtime-optimization-jit-sticky-v2\start_B3-Critic-only-runtime-optimization-jit-sticky-v2_200k_cuda_smoke.cmd
```

Expected diagnostic pattern:

- `runtime/replay_index_direct_scan_count > 0`
- `runtime/replay_index_candidates_materialized_count ~= 0`
- `runtime/numba_support_scan_enabled = 1`
- `runtime/numba_support_scan_fallback_rate = 0`
- `lite/support_scan_mode_is_sticky_random = 1`

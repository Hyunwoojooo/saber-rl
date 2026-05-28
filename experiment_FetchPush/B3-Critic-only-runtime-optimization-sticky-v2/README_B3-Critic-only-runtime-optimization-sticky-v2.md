# B3-Critic-only-runtime-optimization-sticky-v2

This folder is an isolated reconstruction of the pre-JIT code path used for:

`fetch_push_b3_critic_only_runtime_opt_sticky_random_v2_s1_200000_ls512_cuda_smoke`

Current folder:

`C:\Users\SS\Projects\metacog-rl\metacog-rl\version5.0\metacog-lite-v0\experiment_FetchPush\B3-Critic-only-runtime-optimization-sticky-v2`

The corresponding result directory is:

`C:\Users\SS\Projects\metacog-rl\metacog-rl\version5.0\metacog-lite-v0\experiment_FetchPush\B3-Critic-only-runtime-optimization\smoke_results\fetch_push_b3_critic_only_runtime_opt_sticky_random_v2_s1_200000_ls512_cuda_smoke`

Important properties:

- B3-Critic-only
- ReplayActionIndex v2
- sticky_random_v1 scan mode
- local_search_max = 512
- CUDA device
- versioned support context cache enabled
- local scan cache enabled
- exact negative short-circuit enabled
- batch query coalescing disabled
- FetchPush exact JIT scanner not present

The active runtime-optimization folder has since been patched with FetchPush exact JIT. This sticky-v2 snapshot is kept separate so it does not affect any currently running experiment.

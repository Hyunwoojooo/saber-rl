# FPP B3 runtime opt v6 anneal 300k

FPP-only experiment variant based on `b3_critic_only_runtime_opt_v4_ls512`.

Settings:
- method: `b3_critic_only_runtime_opt_v6_anneal_300k`
- seeds: `1 2 3 4 5`
- total_steps: `1000000`
- device: `cuda`
- local_search_max: `512`
- support_anneal_start_step: `300000`
- support_anneal_end_step: `500000`
- support_quality_mode: `none`

Anneal semantics:
- support weight scale is `0.0` before 300k
- linearly ramps from `0.0` to `1.0` between 300k and 500k
- stays `1.0` after 500k

The support lookup, replay sampling, Bellman target, TD error, actor loss, critic loss, target updates, action selection, and exploration noise are otherwise unchanged.

# FPP B3 v4 ls512 lambda schedule + quality floor025

FPP-only experiment variant. This combines the v4 exact Numba runtime optimizer, `local_search_max=512`, a `lambda_support` linear decay schedule, and the FPP quality gate applied to critic weights.

Default schedule:
- `lambda_support=1.0` through 300k steps
- linear decay to `0.5` from 300k to 700k
- hold `0.5` through 1M

Default quality setting:
- `support_quality_mode=fpp_phase_soft`
- `support_quality_apply_to_weight=true`
- `support_quality_floor=0.25`

Run all seeds with:

```bash
SEED_LIST="1 2 3 4 5" STEPS=1000000 DEVICE=cuda bash SABER/FetchPickAndPlace/b3_critic_only_runtime_opt_v4_ls512_lambda_schedule_quality_floor025/run_screening_3seeds_parallel.sh
```

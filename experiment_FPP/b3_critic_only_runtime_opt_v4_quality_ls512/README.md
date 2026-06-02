# B3 Critic-Only Runtime Optimization v4 Quality LS512

FPP-only quality/progress-gated support variant built from `b3_critic_only_runtime_opt_v4_ls512`.

Defaults preserve existing ls512 behavior:

- `support_quality_mode=none`
- `support_quality_apply_to_weight=false`
- `local_search_max=512`

Enable diagnostics without changing weights:

```bash
python3 run.py --seed 1 --steps 50000 --device cuda --support_quality_mode fpp_phase_soft
```

Enable quality-gated critic weights:

```bash
python3 run.py --seed 1 --steps 50000 --device cuda --support_quality_mode fpp_phase_soft --support_quality_apply_to_weight --verify_runtime_optimization
```

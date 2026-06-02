# b0_same_env_v4

Clean DDPG+HER B0 baseline for FetchPickAndPlace, using the same copied v4 source tree, `.venv` Python, CUDA trainer path, seeds, total steps, batch size, HER ratio, eval cadence, and checkpoint cadence as `b3_critic_only_runtime_opt_v4`.

This is a matched-control run for runtime/learning comparison. The trainer method is `b0_ddpg_her`, `lambda_support=0`, `weight_scope=none`, and `support_lookup=false`; no SABER support lookup, support loss weighting, or replay sampling change is applied.

The v4 Numba support scanner code remains present in the source tree and the same CLI flags are accepted so the environment and imports match the v4 experiment. B0 compute returns constant weights, so support lookup/JIT scan is not used for training semantics.

Dry run:

```bash
python3 SABER/FetchPickAndPlace/b0_same_env_v4/run.py --seed 1 --dry-run
```

Smoke run:

```bash
python3 SABER/FetchPickAndPlace/b0_same_env_v4/run.py --seed 1 --smoke --device auto
```

All seeds:

```bash
bash SABER/FetchPickAndPlace/b0_same_env_v4/run_all_seeds.sh --device auto
```

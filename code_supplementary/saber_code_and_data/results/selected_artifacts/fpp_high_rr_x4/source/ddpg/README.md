# B0-high-replay-ratio-x4

Purpose: DDPG+HER high replay-ratio x4 baseline with Omega=1.

This is a DDPG+HER high replay-ratio baseline. Support lookup is disabled, `lambda_support=0.0`, and `weight_scope=none`; the intended difference from `B0` is `updates_per_step=4`.

Dry-run one seed:

```bash
python3 SABER-FetchPickAndPlace-active/FetchPickAndPlace/B0-high-replay-ratio-x4/run.py --seed 1 --steps 50000 --device cuda --dry-run
```

Smoke run:

```bash
python3 SABER-FetchPickAndPlace-active/FetchPickAndPlace/B0-high-replay-ratio-x4/run.py --seed 1 --steps 50000 --device cuda --output_root experiment_FPP/runs/B0-high-replay-ratio-x4_smoke
```

Full 1M run for seeds 1-5 in parallel:

```bash
bash SABER-FetchPickAndPlace-active/FetchPickAndPlace/B0-high-replay-ratio-x4/run_fpp_ddpg_high_rr_x4_1m_5parallel.sh
```

# FPP TD3 HER High Replay-Ratio x4

This experiment is a compute/control baseline for FetchPickAndPlace. It keeps the TD3+HER baseline architecture and environment settings, disables support lookup and SABER weighting, and increases replay updates to four gradient updates per environment step.

## Full Run

- task: `fetch_pick_and_place`
- reward: `sparse`
- method recorded in config/summary: `td3_her_high_rr_x4`
- backbone: TD3+HER
- total steps: `1_000_000`
- seeds: `1 2 3 4 5`
- HER ratio: `0.8`
- updates per step: `4`
- batch size: `256`
- eval frequency: `10_000`
- eval episodes: `20`
- TD3 policy delay: `2`
- TD3 target policy noise: `0.2`
- TD3 target noise clip: `0.5`
- support lookup: disabled
- SABER weights: disabled (`lambda_support=0.0`, `weight_scope=none`)

Output root:

```bash
experiment_FPP/runs/td3_her_high_replay_ratio_x4/
```

Run names:

```bash
fetch_pick_and_place_td3_her_high_rr_x4_s{seed}_1000000_ls512
```

## Commands

Validate command construction:

```bash
python3 experiment_FPP/td3_her_high_replay_ratio_x4/run.py --seed 1 --steps 10000 --device cpu --dry-run
```

50k smoke:

```bash
bash experiment_FPP/td3_her_high_replay_ratio_x4/run_fpp_td3_her_high_rr_x4_smoke.sh
```

1M seeds 1-5 in parallel:

```bash
bash experiment_FPP/td3_her_high_replay_ratio_x4/run_fpp_td3_her_high_rr_x4_1m_5parallel.sh
```

The summary records `total_updates`, `runtime_hours`, `final_steps_per_sec`, `eval_AUC_success`, and `eval_AUC_success_per_hour`.

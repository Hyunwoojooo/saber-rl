# FPP DDPG-HGR Two-Step TD-Error Goal Ranking

This folder isolates a FetchPickAndPlace DDPG+HER baseline with Hindsight Goal Ranking (HGR). HGR is not a new RL backbone. It is a HER goal-prioritization baseline: the DDPG actor, critic, optimizers, exploration noise, reward convention, terminal handling, batch size, and one-update-per-step training loop are kept matched to the existing FPP DDPG+HER baseline.

The only intended intervention is replay/HER sampling.

## What Stays Matched To DDPG+HER

- task: `fetch_pick_and_place`
- reward: `sparse`
- backbone: DDPG
- actor/critic architecture: unchanged
- actor/critic optimizers: unchanged
- gamma/tau/lr/noise/warmup/start steps: unchanged from the local FPP DDPG runner defaults
- HER ratio: `0.8`
- HER strategy family: future achieved goals
- reward recomputation: environment `compute_reward(next_achieved_goal, desired_goal)`
- success/timeout convention: existing FPP convention, `success_terminates=false`, `timeout_bootstrap=true`
- critic loss weights: all ones
- SABER support lookup: off
- PER: off
- high replay ratio: off, `updates_per_step=1`

## What HGR Changes

HGR uses two-step TD-error ranking during HER relabel sampling:

1. Sample an episode with probability proportional to mean transition priority in that episode.
2. Within that episode, choose a transition with future achieved-goal candidates, then rank future goals by the DDPG TD-error under that candidate goal.

For a transition `t` and future achieved goal `g_i`:

```text
delta(t, i) = r(s_{t+1}, g_i)
              + gamma * Q_target(s_{t+1}, mu_target(s_{t+1}, g_i), g_i)
              - Q(s_t, a_t, g_i)
```

Then:

```text
P(episode=e) proportional to (mean_t priority[e,t] + eps) ^ alpha_episode
P(goal=g_i | e,t) proportional to (abs(delta(t,i)) + eps) ^ alpha_goal
```

Priority settings:

- `priority_eps = 1e-6`
- `priority_init = 1.0`
- `priority_clip_max = 50.0`
- `alpha_episode = 1.0`
- `alpha_goal = 1.0`
- `hgr_max_future_candidates = 50`

FPP horizon is 50, so the candidate cap is effectively full-future evaluation for normal episodes.

## Output

```bash
experiment_FPP/runs/ddpg_hgr_tdpriority_1m/
```

Run names:

```bash
fetch_pick_and_place_ddpg_hgr_tdpriority_s{seed}_1000000
```

## Validation Commands

Command construction only:

```bash
python3 experiment_FPP/ddpg_hgr_tdpriority_1m/run.py --seed 1 --steps 10000 --device cpu --dry-run
```

Setup validation:

```bash
PYTHONPATH=experiment_FPP/ddpg_hgr_tdpriority_1m/code/src \
python3 experiment_FPP/ddpg_hgr_tdpriority_1m/code/src/train_fetch_pick_and_place_lite_v0.py \
  --method hgr --task_name fetch_pick_and_place --reward_type sparse --device cpu --validate_only
```

Fake-batch HGR unit test:

```bash
PYTHONPATH=experiment_FPP/ddpg_hgr_tdpriority_1m/code/src \
python3 experiment_FPP/ddpg_hgr_tdpriority_1m/code/src/test_hgr_tdpriority_sampling.py
```

## Execution Plan

Do not launch full training before validation. The intended sequence is:

1. 1k dry-run: `bash experiment_FPP/ddpg_hgr_tdpriority_1m/run_fpp_ddpg_hgr_tdpriority_1k_dry.sh`
2. 50k smoke seed 1: `bash experiment_FPP/ddpg_hgr_tdpriority_1m/run_fpp_ddpg_hgr_tdpriority_smoke.sh`
3. 200k pilot seeds 1-2: `bash experiment_FPP/ddpg_hgr_tdpriority_1m/run_fpp_ddpg_hgr_tdpriority_pilot_200k_2seed.sh`
4. 1M full seeds 1-5: `bash experiment_FPP/ddpg_hgr_tdpriority_1m/run_fpp_ddpg_hgr_tdpriority_1m_5parallel.sh`

## Paper Interpretation

If HGR beats SABER, a safe interpretation is:

```text
goal-prioritization can be stronger in FPP DDPG, while SABER targets a different critic-loss allocation intervention under matched HER relabeling.
```

If SABER matches or beats HGR, a safe interpretation is:

```text
SABER remains competitive with a TD-error-based hindsight goal-prioritization baseline while changing only critic loss allocation.
```

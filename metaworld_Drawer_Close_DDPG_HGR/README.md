# MetaWorld DrawerClose DDPG+HER+HGR Baseline

This folder is an independent DrawerClose DDPG+HGR experiment snapshot. It keeps the existing DrawerClose DDPG+HER task/env convention and adds the same vectorized two-step HGR sampler used by the DrawerOpen HGR baseline. Existing DrawerClose DDPG/SABER/PER code remains unchanged.

## What HGR Is

Hindsight Goal Ranking (HGR) is not a backbone RL algorithm. The backbone here is still DDPG, and HER is still the sparse goal relabeling framework. HGR changes only the hindsight replay sampling policy:

1. sample episodes with higher average TD-error priority more often;
2. inside the selected episode, rank future achieved-goal candidates by current TD error and sample the hindsight goal from that distribution.

This implementation is the two-step TD-error HGR baseline, not HGR-lite. It implements both episode priority and future-goal priority.

## Held Fixed From DrawerClose DDPG+HER

- Task: `drawer_close`
- Environment: `drawer-close-v3`
- Reward: sparse
- Episode horizon: 200
- Actor architecture: unchanged DDPG actor
- Critic architecture: unchanged single-critic DDPG critic
- Target computation: unchanged DDPG target path
- Optimizers: Adam, actor/critic lr `3e-4`
- Gamma/tau: `0.98` / `0.005`
- Exploration noise: `0.2`
- Batch size: `256`
- Replay capacity: `300000`
- HER ratio: `0.8`
- Updates per env step: `1`
- Terminal convention: `success_terminates=false`, `timeout_bootstrap=true`
- Critic and actor loss weights: all ones

SABER support lookup, SABER critic loss weighting, PER, and high replay-ratio are off for `method=hgr`.

## HGR-Specific Settings

Defaults:

- `hgr_priority_eps = 1e-6`
- `hgr_priority_init = 1.0`
- `hgr_priority_clip_max = 50.0`
- `hgr_alpha_episode = 1.0`
- `hgr_alpha_goal = 1.0`
- `hgr_uniform_mix = 0.0`
- `hgr_max_future_candidates = 50`
- `hgr_episode_priority_ema = 0.0`

Priority storage:

- per replay-slot transition priority is stored in `HGRSampler.slot_priority`;
- episode priority is maintained as the mean of valid transition priorities;
- future-goal TD errors are computed on the fly and are not stored as a large goal table.

Goal relabel convention follows the existing DrawerClose HER code path and uses future `next_achieved_goal` values from replay.

## Files

- Source root: `metaworld_Drawer_Close_DDPG_HGR/`
- Trainer: `code/src/train_drawer_close_lite_v0.py`
- HGR sampler: `code/src/metacog_lite_v0/hgr.py`
- DrawerClose task spec: `code/src/metacog_lite_v0/task_drawer_close.py`
- Fake-batch test: `code/src/test_hgr_fake_batch.py`
- Launcher: `DDPG-HER-HGR-2M.sh`
- Default results: `metaworld_Drawer_Close_DDPG_HGR/runs/ddpg_hgr_tdpriority_2m/`
- Default logs: `metaworld_Drawer_Close_DDPG_HGR/logs/ddpg_hgr_tdpriority_2m/`

## Logged Artifacts

Each run writes `config.json`, `env_snapshot.json`, `run.log`, `metrics.csv`, `metrics.jsonl`, `metrics_selected_10k.csv`, `summary.json`, and checkpoints unless disabled by the launcher.

Selected HGR metrics include `hgr/episode_priority_mean`, `hgr/episode_priority_std`, `hgr/episode_priority_max`, `hgr/goal_priority_entropy_mean`, `hgr/goal_selected_rank_mean`, `hgr/candidate_td_error_mean`, `hgr/candidate_td_error_max`, `hgr/relabel_fraction`, `hgr/priority_clip_rate`, and `runtime/hgr_sampling_sec_per_update`.

## Validation Commands

Compile:

```bash
PYTHONPATH=metaworld_Drawer_Close_DDPG_HGR/code/src python3 -m py_compile \
  metaworld_Drawer_Close_DDPG_HGR/code/src/metacog_lite_v0/hgr.py \
  metaworld_Drawer_Close_DDPG_HGR/code/src/metacog_lite_v0/replay.py \
  metaworld_Drawer_Close_DDPG_HGR/code/src/metacog_lite_v0/trainer.py \
  metaworld_Drawer_Close_DDPG_HGR/code/src/train_drawer_close_lite_v0.py \
  metaworld_Drawer_Close_DDPG_HGR/code/src/test_hgr_fake_batch.py
```

Launcher syntax:

```bash
bash -n metaworld_Drawer_Close_DDPG_HGR/DDPG-HER-HGR-2M.sh
```

Dry-run command generation:

```bash
DRY_RUN=1 STEPS=1000 STEP_TAG=dry CUDA_DEVICE=0 metaworld_Drawer_Close_DDPG_HGR/DDPG-HER-HGR-2M.sh
```

Fake-batch HGR sampler test:

```bash
PYTHONPATH=metaworld_Drawer_Close_DDPG_HGR/code/src python3 metaworld_Drawer_Close_DDPG_HGR/code/src/test_hgr_fake_batch.py
```

Validate DrawerClose setup, including HGR sampling:

```bash
PYTHONPATH=metaworld_Drawer_Close_DDPG_HGR/code/src python3 metaworld_Drawer_Close_DDPG_HGR/code/src/train_drawer_close_lite_v0.py \
  --method hgr --weight_scope none --validate_only --device cpu
```

## Execution Plan

1. 1k dry run:

```bash
PYTHONPATH=metaworld_Drawer_Close_DDPG_HGR/code/src python3 metaworld_Drawer_Close_DDPG_HGR/code/src/train_drawer_close_lite_v0.py \
  --method hgr --weight_scope none --device cpu \
  --run_name drawer_close_ddpg_hgr_tdpriority_s1_1k_cpu \
  --output_root runs/dry_1k \
  --total_steps 1000 --start_steps 100 --warmup_steps 100 \
  --batch_size 64 --buffer_capacity 20000 --eval_freq 1000 --eval_episodes 2 \
  --checkpoint_every_steps 1000 --no_save_step_checkpoints
```

2. 50k smoke seed 1: run one explicit command with `--total_steps 50000`, or temporarily limit the launcher seed loop.

3. 200k pilot seeds 1-2: run two explicit commands with `STEPS=200000 STEP_TAG=200k`.

4. 2M full seeds 1-5:

```bash
STEPS=2000000 STEP_TAG=2m CUDA_DEVICE=0 metaworld_Drawer_Close_DDPG_HGR/DDPG-HER-HGR-2M.sh
```

## Paper Interpretation

If HGR is stronger than SABER on DrawerClose DDPG, the interpretation is that TD-error-based hindsight goal prioritization can be stronger in this task, while SABER targets a different intervention point: critic-loss allocation under matched HER relabeling.

If SABER is similar to or stronger than HGR, the interpretation is that SABER remains competitive with a TD-error-based hindsight goal-prioritization baseline while changing only critic loss allocation.

## Remaining Risks

- HGR does extra no-grad critic evaluations during sampling, so runtime can be materially slower than B0.
- `hgr_max_future_candidates=50` caps DrawerClose future candidates. Set it to `0` for full future candidate scoring if runtime allows.
- If priority sharpness is unstable, use `hgr_alpha_goal=0.6` or a small `hgr_uniform_mix` such as `0.05` as fallback ablations.

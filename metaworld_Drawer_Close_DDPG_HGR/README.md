# MetaWorld DrawerClose DDPG+HER+Full HGR

This directory contains an independent DrawerClose DDPG+HER implementation of
the full two-stage Hindsight Goal Ranking (HGR) baseline. It uses the same
completed-episode pair-priority, importance-sampling (IS), beta annealing,
replay-overwrite, and checkpoint semantics as
`experiment_FPP/ddpg_hgr_tdpriority_1m/`.

The existing DrawerClose DDPG HER/SABER/PER implementations are unchanged.

## HGR Intervention

For each completed episode of length `T`, the sampler stores every valid
transition/future-goal pair. A pair links transition slot `j` with a future
`next_achieved_goal`; an episode of horizon 200 has 20,100 valid pairs.

Sampling has two stages:

1. select a completed episode using its mean pair priority;
2. select a transition/future-goal pair inside that episode using pair priority.

Episode and pair distributions use TD-error priorities with
`alpha_episode=alpha_goal=0.6`. Newly created pairs receive the current global
maximum priority. After a critic update, only the sampled relabeled pairs are
updated using the pre-update absolute TD error exposed by the DDPG update.

For an HGR-relabeled sample, the critic receives the normalized product of the
episode-level and pair-level IS corrections. The maximum relabeled IS weight in
the minibatch is one. Actor weights and original desired-goal critic weights
remain one.

## Fixed DrawerClose Setup

- Environment: MetaWorld `drawer-close-v3`
- Reward: sparse, reward function `v2`
- Horizon: 200
- Backbone: single-critic DDPG
- Total steps: 2,000,000
- Seeds: 1, 2, 3, 4, 5
- Batch size / replay capacity: 256 / 300,000
- Future-goal HER probability: 0.8
- Random start / warmup: 10,000 / 5,000 steps
- Updates per environment step: 1
- Actor / critic learning rate: 3e-4 / 3e-4
- Gamma / tau: 0.98 / 0.005
- Exploration noise: 0.2
- Terminal convention: `success_terminates=false`,
  `timeout_bootstrap=true`

Full HGR settings:

- `hgr_priority_eps = 1e-6`
- `hgr_priority_init = 1.0`
- `hgr_priority_clip_max = 0.0` (disabled)
- `hgr_alpha_episode = hgr_alpha_goal = 0.6`
- `hgr_beta_episode_initial = hgr_beta_goal_initial = 0.4`
- `hgr_beta_episode_final = hgr_beta_goal_final = 1.0`
- `hgr_beta_anneal_updates = 0` (automatic full-run linear annealing)
- `hgr_uniform_mix = 0.0`
- Checkpoint schema: `hgr_episode_pair_priority_v1`

SABER support weighting, PER, and high replay ratio are disabled.

## Output Paths

New Full HGR runs are isolated from the legacy sampler-only results:

```text
runs/ddpg_hgr_tdpriority_original_pair_is_2m/
  DDPG-HER-HGR-original-pair-is-2m/
    drawer_close_ddpg_hgr_original_pair_is_s{1..5}_2m_cuda/
```

Logs are written below
`logs/ddpg_hgr_tdpriority_original_pair_is_2m/`.

## Legacy Result Boundary

The compact artifact at
`paper_artifacts/drawer_close_ddpg_her_hgr_2m/` was produced by the previous
two-stage sampler-only implementation. That implementation selected episodes
and future goals using TD error but did not maintain full pair-priority tables
or apply IS correction to the critic loss.

Those completed results must be labeled:

> HGR-style two-stage sampler without importance correction

They are not Full HGR results and are intentionally kept separate from new
`original_pair_is` outputs.

## Implementation

- Sampler: `code/src/metacog_lite_v0/hgr.py`
- Replay metadata: `code/src/metacog_lite_v0/replay.py`
- Training integration: `code/src/metacog_lite_v0/trainer.py`
- DrawerClose entrypoint: `code/src/train_drawer_close_lite_v0.py`
- Regression test: `code/src/test_hgr_fake_batch.py`
- Launcher: `DDPG-HER-HGR-2M.sh`

The run logs pair-table count and memory, episode/pair probabilities,
raw/normalized IS weights, beta progress, clipping/safety replacements, and HGR
sampling runtime. Checkpoints include replay state, pair tables, global maximum
priority, critic update count, and beta schedule state.

## Validation

```bash
PYTHONPATH=metaworld_Drawer_Close_DDPG_HGR/code/src \
  .venv/bin/python -m py_compile \
  metaworld_Drawer_Close_DDPG_HGR/code/src/metacog_lite_v0/config.py \
  metaworld_Drawer_Close_DDPG_HGR/code/src/metacog_lite_v0/hgr.py \
  metaworld_Drawer_Close_DDPG_HGR/code/src/metacog_lite_v0/replay.py \
  metaworld_Drawer_Close_DDPG_HGR/code/src/metacog_lite_v0/trainer.py \
  metaworld_Drawer_Close_DDPG_HGR/code/src/train_drawer_close_lite_v0.py \
  metaworld_Drawer_Close_DDPG_HGR/code/src/test_hgr_fake_batch.py

PYTHONPATH=metaworld_Drawer_Close_DDPG_HGR/code/src \
  .venv/bin/python \
  metaworld_Drawer_Close_DDPG_HGR/code/src/test_hgr_fake_batch.py

PYTHONPATH=metaworld_Drawer_Close_DDPG_HGR/code/src \
  .venv/bin/python \
  metaworld_Drawer_Close_DDPG_HGR/code/src/train_drawer_close_lite_v0.py \
  --method hgr --weight_scope none --validate_only --device cpu

bash -n metaworld_Drawer_Close_DDPG_HGR/DDPG-HER-HGR-2M.sh
DRY_RUN=1 metaworld_Drawer_Close_DDPG_HGR/DDPG-HER-HGR-2M.sh
```

## Execution

Full 2M experiment:

```bash
CUDA_DEVICE=0 metaworld_Drawer_Close_DDPG_HGR/DDPG-HER-HGR-2M.sh
```

Selected seeds or a shorter pilot:

```bash
SEED_LIST="1 2" STEPS=200000 STEP_TAG=pilot200k \
  CUDA_DEVICE=0 metaworld_Drawer_Close_DDPG_HGR/DDPG-HER-HGR-2M.sh
```

## Operational Notes

- At horizon 200, a completed episode owns 20,100 pair priorities.
- With a replay capacity of 300,000, the pair-priority arrays can use roughly
  121 MB per seed when the replay is full.
- Replay overwrite removes the stale episode's complete pair table.
- A checkpoint from the legacy transition-priority implementation is
  incompatible with `hgr_episode_pair_priority_v1`; start a new Full HGR run.

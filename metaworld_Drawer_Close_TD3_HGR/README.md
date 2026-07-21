# MetaWorld DrawerClose TD3+HER+Full HGR Pair-Priority + IS Baseline

This folder is an independent DrawerClose TD3+HER experiment snapshot using the same completed-episode pair-priority and importance-correction HGR protocol as the FPP HGR baseline. Only the task adapter and matched TD3-specific TD-error computation differ.

## What HGR Is

Hindsight Goal Ranking (HGR) is not a backbone RL algorithm. The backbone remains TD3 and HER remains active at ratio 0.8. The 80% HER branch uses the Full HGR protocol:

1. build every valid transition/future-goal pair for each completed episode;
2. sample episodes from mean pair priority and then sample pairs from pair priority;
3. apply the normalized product of episode and pair importance-sampling weights to both TD3 critic residuals;
4. keep actor weights and the 20% original-goal branch at one.

This is the same HGR algorithm used by the FPP full pair-priority baseline, adapted to the matched Drawer TD3 backbone.

## Held Fixed From DrawerClose TD3+HER

- Task: `drawer_close`
- Environment: `drawer-close-v3`
- Reward: sparse
- Episode horizon: 200
- Actor architecture: unchanged TD3 actor
- Critic architecture: unchanged twin-critic TD3 critics
- Target computation: clipped double-Q with target-policy smoothing
- Optimizers: Adam, actor/critic lr `3e-4`
- Gamma/tau: `0.98` / `0.005`
- Exploration noise: `0.2`
- Policy delay: `2`
- Target policy noise/clip: `0.2` / `0.5`
- Batch size: `256`
- Replay capacity: `300000`
- HER ratio: `0.8`
- Updates per env step: `1`
- Terminal convention: `success_terminates=false`, `timeout_bootstrap=true`
- HGR critic weights: normalized episode x pair importance weights on relabeled samples
- Actor weights and original-goal branch weights: all ones

SABER support lookup, SABER critic loss weighting, PER, and high replay-ratio are off for `method=hgr`.

## HGR-Specific Settings

Defaults:

- `hgr_priority_eps = 1e-6`
- `hgr_priority_init = 1.0`
- `hgr_priority_clip_max = 0.0` (disabled)
- `hgr_alpha_episode = 0.6`
- `hgr_alpha_goal = 0.6`
- `hgr_beta_episode_initial = hgr_beta_goal_initial = 0.4`
- both beta values anneal linearly to `1.0`
- `hgr_uniform_mix = 0.0`

Priority storage:

- each completed episode owns a triangular table for every valid transition/future-goal pair;
- new pair priorities use the current global maximum;
- episode priority is the mean of its pair priorities;
- only sampled pair entries are updated from pre-update TD errors;
- pair tables, beta progress, and global maximum priority are checkpointed.

Goal relabel convention follows the existing DrawerClose HER code path and uses future `next_achieved_goal` values from replay.

For TD3, pair priorities use the pre-update mean absolute TD error of the two critics against the standard clipped-double-Q target with target-policy smoothing. The same IS weight is applied to both critic residuals.

## Files

- Source root: `metaworld_Drawer_Close_TD3_HGR/`
- Trainer: `code/src/train_drawer_close_lite_v0.py`
- HGR sampler: `code/src/metacog_lite_v0/hgr.py`
- DrawerClose task spec: `code/src/metacog_lite_v0/task_drawer_close.py`
- Fake-batch test: `code/src/test_hgr_fake_batch.py`
- Launcher: `TD3-HER-HGR-2M.sh`
- Default results: `metaworld_Drawer_Close_TD3_HGR/runs/td3_hgr_pair_priority_is_2m/`
- Default logs: `metaworld_Drawer_Close_TD3_HGR/logs/td3_hgr_pair_priority_is_2m/`

## Logged Artifacts

Each run writes `config.json`, `env_snapshot.json`, `run.log`, `metrics.csv`, `metrics.jsonl`, `metrics_selected_10k.csv`, `summary.json`, and checkpoints unless disabled by the launcher.

Selected HGR metrics include `hgr/episode_priority_mean`, `hgr/episode_priority_std`, `hgr/episode_priority_max`, `hgr/goal_priority_entropy_mean`, `hgr/goal_selected_rank_mean`, `hgr/candidate_td_error_mean`, `hgr/candidate_td_error_max`, `hgr/relabel_fraction`, `hgr/priority_clip_rate`, and `runtime/hgr_sampling_sec_per_update`.

## Validation Commands

Compile:

```bash
PYTHONPATH=metaworld_Drawer_Close_TD3_HGR/code/src python3 -m py_compile \
  metaworld_Drawer_Close_TD3_HGR/code/src/metacog_lite_v0/hgr.py \
  metaworld_Drawer_Close_TD3_HGR/code/src/metacog_lite_v0/replay.py \
  metaworld_Drawer_Close_TD3_HGR/code/src/metacog_lite_v0/trainer.py \
  metaworld_Drawer_Close_TD3_HGR/code/src/train_drawer_close_lite_v0.py \
  metaworld_Drawer_Close_TD3_HGR/code/src/test_hgr_fake_batch.py
```

Launcher syntax:

```bash
bash -n metaworld_Drawer_Close_TD3_HGR/TD3-HER-HGR-2M.sh
```

Dry-run command generation:

```bash
DRY_RUN=1 STEPS=1000 STEP_TAG=dry CUDA_DEVICE=0 metaworld_Drawer_Close_TD3_HGR/TD3-HER-HGR-2M.sh
```

Fake-batch HGR sampler test:

```bash
PYTHONPATH=metaworld_Drawer_Close_TD3_HGR/code/src python3 metaworld_Drawer_Close_TD3_HGR/code/src/test_hgr_fake_batch.py
```

Validate DrawerClose setup, including HGR sampling:

```bash
PYTHONPATH=metaworld_Drawer_Close_TD3_HGR/code/src python3 metaworld_Drawer_Close_TD3_HGR/code/src/train_drawer_close_lite_v0.py \
  --method hgr --weight_scope none --validate_only --device cpu
```

## Execution Plan

1. 1k dry run:

```bash
PYTHONPATH=metaworld_Drawer_Close_TD3_HGR/code/src python3 metaworld_Drawer_Close_TD3_HGR/code/src/train_drawer_close_lite_v0.py \
  --method hgr --weight_scope none --device cpu \
  --run_name drawer_close_td3_hgr_pair_priority_is_s1_1k_cpu \
  --output_root runs/dry_1k \
  --total_steps 1000 --start_steps 100 --warmup_steps 100 \
  --batch_size 64 --buffer_capacity 20000 --eval_freq 1000 --eval_episodes 2 \
  --checkpoint_every_steps 1000 --no_save_step_checkpoints
```

2. 50k smoke seed 1: run one explicit command with `--total_steps 50000`, or temporarily limit the launcher seed loop.

3. 200k pilot seeds 1-2: run two explicit commands with `STEPS=200000 STEP_TAG=200k`.

4. 2M full seeds 1-5:

```bash
STEPS=2000000 STEP_TAG=2m CUDA_DEVICE=0 metaworld_Drawer_Close_TD3_HGR/TD3-HER-HGR-2M.sh
```


## Completed Result Summary

Completed on 2026-07-21 KST with five seeds, 2M environment steps per seed,
sparse `drawer-close-v3`, and 20 deterministic evaluation episodes every 10k
steps.

| Seed | Final Success | Best Success | AUC Success | Last-Quarter AUC | First Hit 90% |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.40 | 1.00 | 0.924 | 0.884 | 80k |
| 2 | 0.95 | 1.00 | 0.686 | 0.806 | 60k |
| 3 | 1.00 | 1.00 | 0.782 | 0.416 | 130k |
| 4 | 0.15 | 1.00 | 0.477 | 0.112 | 60k |
| 5 | 0.00 | 1.00 | 0.139 | 0.064 | 260k |
| **Mean** | **0.50** | **1.00** | **0.602** | **0.457** | **118k** |

All five seeds reached 90% success at least once, while two of five remained
at or above 90% at the final evaluation. Mean throughput was 11.81 steps/s per
seed, and the five-seed parallel block took approximately 47.1 hours.

### Matched DrawerClose TD3 Context

| Method | Final Success Seeds | Final Success | AUC Success | Mean Hit90 | Speed |
|---|---:|---:|---:|---:|---:|
| TD3+HER B0 | 1/5 | 0.35 | 0.329 | 76k | 41.70 steps/s |
| TD3+HER+B3 Critic-Only | 3/5 | **0.59** | 0.549 | 86k | 22.49 steps/s |
| TD3+HER+Full HGR | 2/5 | 0.50 | **0.602** | 118k | 11.81 steps/s |

Full HGR exceeds B3 Critic-Only in mean AUC by 0.053, but the paired
seed-bootstrap 95% interval for the AUC difference is wide and includes zero
(`[-0.264, 0.468]`), with positive differences on three of five seeds. B3 has
higher final success, one more seed above 90% at the final evaluation, and
approximately 1.9x higher throughput. The result therefore establishes Full
HGR as a strong but computationally expensive baseline without demonstrating
clear dominance over B3 across metrics.

## Paper Interpretation

If HGR is stronger than SABER on DrawerClose TD3, the interpretation is that TD-error-based hindsight goal prioritization can be stronger in this task, while SABER targets a different intervention point: critic-loss allocation under matched HER relabeling.

If SABER is similar to or stronger than HGR, the interpretation is that SABER remains competitive with a TD-error-based hindsight goal-prioritization baseline while changing only critic loss allocation.

## Remaining Risks

- Full HGR materializes completed-episode pair tables and evaluates their priority distributions during sampling, so runtime can be materially slower than B0.
- Horizon 200 creates 20,100 pair priorities for a full episode. Runtime and pair-table memory are therefore logged explicitly.
- Do not resume checkpoints from the superseded sampler-only implementation; the Full HGR checkpoint schema is `hgr_episode_pair_priority_v1`.

# MetaWorld DrawerOpen TD3+HER+Full HGR Pair-Priority + IS Baseline

This folder is an independent DrawerOpen TD3+HER experiment snapshot using the same completed-episode pair-priority and importance-correction HGR protocol as the FPP HGR baseline. Only the task adapter and matched TD3-specific TD-error computation differ.

## What HGR Is

Hindsight Goal Ranking (HGR) is not a backbone RL algorithm. The backbone remains TD3 and HER remains active at ratio 0.8. The 80% HER branch uses the Full HGR protocol:

1. build every valid transition/future-goal pair for each completed episode;
2. sample episodes from mean pair priority and then sample pairs from pair priority;
3. apply the normalized product of episode and pair importance-sampling weights to both TD3 critic residuals;
4. keep actor weights and the 20% original-goal branch at one.

This is the same HGR algorithm used by the FPP full pair-priority baseline, adapted to the matched Drawer TD3 backbone.

## Held Fixed From DrawerOpen TD3+HER

- Task: `drawer_open`
- Environment: `drawer-open-v3`
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

Goal relabel convention follows the existing DrawerOpen HER code path and uses future `next_achieved_goal` values from replay.

For TD3, pair priorities use the pre-update mean absolute TD error of the two critics against the standard clipped-double-Q target with target-policy smoothing. The same IS weight is applied to both critic residuals.

## Files

- Source root: `metaworld_Drawer_Open_TD3_HGR/`
- Trainer: `code/src/train_drawer_open_lite_v0.py`
- HGR sampler: `code/src/metacog_lite_v0/hgr.py`
- Replay change: `code/src/metacog_lite_v0/replay.py` returns the written replay slot from `add_transition()`
- Fake-batch test: `code/src/test_hgr_fake_batch.py`
- Launcher: `TD3-HER-HGR-2M.sh`
- Default results: `metaworld_Drawer_Open_TD3_HGR/runs/td3_hgr_pair_priority_is_2m/`
- Default logs: `metaworld_Drawer_Open_TD3_HGR/logs/td3_hgr_pair_priority_is_2m/`

The repository `.gitignore` already excludes checkpoints (`*.pt`), logs, `results*`, and `logs*`. This folder writes full artifacts under `runs/` by default so they can be excluded or curated separately before pushing.

## Logged Artifacts

Each run writes:

- `config.json`
- `env_snapshot.json`
- `run.log`
- `metrics.csv`
- `metrics.jsonl`
- `metrics_selected_10k.csv`
- `summary.json`
- `checkpoint.pt` and `best_checkpoint.pt` unless disabled by the launcher

Selected HGR metrics:

- `hgr/episode_priority_mean`
- `hgr/episode_priority_std`
- `hgr/episode_priority_max`
- `hgr/pair_table_count`
- `hgr/pair_priority_count`
- `hgr/pair_priority_memory_mb`
- `hgr/goal_priority_entropy_mean`
- `hgr/goal_selected_rank_mean`
- `hgr/candidate_td_error_mean`
- `hgr/candidate_td_error_max`
- `hgr/relabel_fraction`
- `hgr/episode_probability_mean`
- `hgr/pair_probability_mean`
- `hgr/is_weight_mean`
- `hgr/is_weight_min`
- `hgr/beta_episode`
- `hgr/beta_goal`
- `hgr/priority_clip_rate`
- `runtime/hgr_sampling_sec_per_update`

Selected performance/runtime metrics:

- final success
- best success
- AUC success
- last-quarter AUC success
- Hit90 reached flag and first Hit90 step
- steps/sec
- updates/sec
- runtime hours
- AUC/hour
- critic loss
- actor loss

## Completed Results: Full HGR 2M

The Full HGR run completed seeds 1-5 at 2M environment steps on 2026-07-19
KST. Evaluation used 20 deterministic episodes every 10k environment steps.
AUC was computed on the exact recorded grid without smoothing, interpolation,
or forward filling.

| Seed | Final success | Best success | AUC | Last-quarter AUC | Hit90 | Steps/s | Runtime |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | 0.00 | 0.000 | 0.000 | - | 11.66 | 47.63 h |
| 2 | 1.00 | 1.00 | 0.692 | 0.992 | 520k | 11.65 | 47.70 h |
| 3 | 0.90 | 1.00 | 0.532 | 0.997 | 930k | 11.64 | 47.73 h |
| 4 | 0.00 | 0.00 | 0.000 | 0.000 | - | 11.65 | 47.69 h |
| 5 | 1.00 | 1.00 | 0.628 | 0.996 | 760k | 11.64 | 47.75 h |
| Mean | 0.58 | 0.60 | 0.371 | 0.597 | 737k* | 11.65 | 47.70 h |

`*` Mean Hit90 is computed over the three seeds that reached 90% success;
coverage is 3/5 and the median is 760k. The AUC sample standard deviation is
0.343.

### Matched TD3 Comparison

| Experiment | Success seeds | Final success | AUC | Mean Hit90 | Steps/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER | 1/5 | 0.19 | 0.127 | 740k* | 41.55 |
| TD3+HER+SABER/B3-Critic-Only | 3/5 | 0.59 | 0.362 | 803k* | 21.57 |
| TD3+HER+Full HGR | 3/5 | 0.58 | 0.371 | 737k* | 11.65 |

For Full HGR minus HER, the paired mean AUC difference is +0.244 with a 95%
seed-bootstrap interval of [-0.253, 0.635]. For Full HGR minus SABER, the
paired mean AUC difference is +0.009 with a 95% interval of [-0.471, 0.489].
The five paired seeds therefore do not establish performance dominance between
Full HGR and SABER: Full HGR has the higher AUC on two seeds, while SABER has
the higher AUC on three.

The recorded training throughput is 11.65 steps/s for Full HGR and 21.57
steps/s for SABER, making SABER approximately 1.85x faster in these recorded
runs. This is an observed run-level comparison rather than a
hardware-normalized runtime claim. Full HGR changes HER pair sampling and
applies importance correction, whereas SABER keeps uniform replay and changes
only critic-loss allocation.

The committed result payload contains each seed's `metrics.csv` and
`summary.json`. Checkpoints, JSONL logs, and intermediate step checkpoints are
excluded.

## Validation Commands

Compile:

```bash
PYTHONPATH=metaworld_Drawer_Open_TD3_HGR/code/src python3 -m py_compile   metaworld_Drawer_Open_TD3_HGR/code/src/metacog_lite_v0/hgr.py   metaworld_Drawer_Open_TD3_HGR/code/src/metacog_lite_v0/replay.py   metaworld_Drawer_Open_TD3_HGR/code/src/metacog_lite_v0/trainer.py   metaworld_Drawer_Open_TD3_HGR/code/src/train_drawer_open_lite_v0.py   metaworld_Drawer_Open_TD3_HGR/code/src/test_hgr_fake_batch.py
```

Launcher syntax:

```bash
bash -n metaworld_Drawer_Open_TD3_HGR/TD3-HER-HGR-2M.sh
```

Dry-run command generation:

```bash
DRY_RUN=1 metaworld_Drawer_Open_TD3_HGR/TD3-HER-HGR-2M.sh
```

Fake-batch HGR sampler test:

```bash
PYTHONPATH=metaworld_Drawer_Open_TD3_HGR/code/src   python3 metaworld_Drawer_Open_TD3_HGR/code/src/test_hgr_fake_batch.py
```

Validate DrawerOpen setup, including HGR sampling:

```bash
PYTHONPATH=metaworld_Drawer_Open_TD3_HGR/code/src   python3 metaworld_Drawer_Open_TD3_HGR/code/src/train_drawer_open_lite_v0.py   --method hgr --weight_scope none --validate_only --device cpu
```

## Execution Plan

1. 1k dry run:

```bash
PYTHONPATH=metaworld_Drawer_Open_TD3_HGR/code/src python3 metaworld_Drawer_Open_TD3_HGR/code/src/train_drawer_open_lite_v0.py   --method hgr --weight_scope none --device cpu   --run_name drawer_open_td3_hgr_pair_priority_is_s1_1k_cpu   --output_root metaworld_Drawer_Open_TD3_HGR/runs/dry_1k   --total_steps 1000 --start_steps 100 --warmup_steps 100   --batch_size 64 --buffer_capacity 20000 --eval_freq 1000 --eval_episodes 2   --checkpoint_every_steps 1000 --no_save_step_checkpoints
```

2. 50k smoke seed 1:

```bash
STEPS=50000 STEP_TAG=50k CUDA_DEVICE=0 metaworld_Drawer_Open_TD3_HGR/TD3-HER-HGR-2M.sh
```

3. 200k pilot seeds 1-2: run the launcher after temporarily limiting the seed loop or launch two explicit commands with `STEPS=200000 STEP_TAG=200k`.

4. 2M full seeds 1-5:

```bash
STEPS=2000000 STEP_TAG=2m CUDA_DEVICE=0 metaworld_Drawer_Open_TD3_HGR/TD3-HER-HGR-2M.sh
```

## Paper Interpretation

If HGR is stronger than SABER on DrawerOpen TD3, the interpretation is that TD-error-based hindsight goal prioritization can be stronger in this task, while SABER targets a different intervention point: critic-loss allocation under matched HER relabeling.

If SABER is similar to or stronger than HGR, the interpretation is that SABER remains competitive with a TD-error-based hindsight goal-prioritization baseline while changing only critic loss allocation.

## Remaining Risks

- Full HGR materializes completed-episode pair tables and evaluates their priority distributions during sampling, so runtime can be materially slower than B0.
- Horizon 200 creates 20,100 pair priorities for a full episode. Runtime and pair-table memory are therefore logged explicitly.
- Do not resume checkpoints from the superseded sampler-only implementation; the Full HGR checkpoint schema is `hgr_episode_pair_priority_v1`.

# DDPG-HER-B3-Replay-Action-Query-2M

This isolated DrawerOpen DDPG experiment changes exactly one SABER Main
mechanism: the action used for support membership lookup.

- Main: `a_query = pi(observation, relabeled_goal)`
- Replay-action arm: `a_query = batch.action_exec`

The context still uses the sampled observation, HER-relabeled training goal,
and remaining horizon. The query action is used only to choose the action bin
whose membership is tested in the occupied local replay-action bins. The
DrawerOpen context construction, deterministic recent-tail scan, local
population thresholds, scanner formula, and normalized critic weighting remain
unchanged.

The sampled transition is **not removed from the local replay pool**. Therefore,
when the bounded local scan includes that transition, its replay action may
self-support the replay-action query. This is intentional and is the same
replay-query definition used by the FPP query ablation.

This experiment tests whether the current-policy query is necessary. It does
not change the actor objective or actor update, target action, Bellman target,
HER relabeling, reward recomputation, replay sampling, or minibatch membership.
Bellman regression remains `Q(x_i, a_replay_i)`, the actor weight is always one,
and SABER weights remain confined to the single DDPG critic loss.
`support_assignment_mode=support`; no strict count-matched assignment is used.

## Parity and provenance

`validate_config_parity.py` recursively compares all five generated production
configs against canonical DrawerOpen SABER Main. The only allowed differences
are:

- `support_query_mode`
- `run_name`
- `output_root`

The assignment RNG remains at the serialized default `30001` for every seed and
is not consumed by support assignment. Environment, Torch, NumPy, action-noise,
weight, and evaluation seeds retain the canonical mapping.

The recommended root launcher
`metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Replay-Action-Query-2M.sh` delegates to
`launch.sh` in this directory. All production argument construction is in this
directory (`launch.sh` and `run.py`), so it is included by `freeze_source.py` in
the frozen source manifest. The root file is a location-stable two-line
delegating entry point.

## Pre-production correctness sequence

From the repository root, run:

```bash
MAX_PARALLEL=1 CUDA_DEVICE=0 \
  bash metaworld_Drawer_Open_DDPG/replay_action_query_2m/run_sequence.sh
```

The sequence performs Python compilation, config parity, runner parity, all
isolated unit/integration tests, a 5k correctness run, the seed-1 20k smoke,
and the seeds-1--5 20k smoke, validating each generated run. It deliberately
stops before production. Short runs are correctness checks only and must never
be interpreted as performance evidence or included in the paper artifact.
Existing output directories are never overwritten.

The safe scheduler default is one process per selected GPU. Raising
`MAX_PARALLEL` is an explicit operator decision after checking GPU memory; the
launcher never starts all five seeds concurrently by default.

After every smoke gate passes, locally commit the verified source, record that
commit SHA, then create and verify the immutable source manifest:

```bash
.venv/bin/python \
  metaworld_Drawer_Open_DDPG/replay_action_query_2m/freeze_source.py --write
.venv/bin/python \
  metaworld_Drawer_Open_DDPG/replay_action_query_2m/freeze_source.py --verify
```

Do not push that commit without explicit user approval.

## Production

Only after the local source freeze is recorded:

```bash
MODE=main2m SEEDS="1 2 3 4 5" MAX_PARALLEL=1 CUDA_DEVICE=0 \
  bash metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Replay-Action-Query-2M.sh
```

Production results are collision-isolated under:

```text
metaworld_Drawer_Open_DDPG/results_2m/
  DDPG-HER-B3-Replay-Action-Query-2M/
```

Each run disables `checkpoint_step_*.pt` copies but retains `checkpoint.pt` at
completion. To inspect a generated cohort without making any performance
claim, run:

```bash
.venv/bin/python \
  metaworld_Drawer_Open_DDPG/replay_action_query_2m/validate_integration.py \
  --mode main2m
```

The integration gate checks the exact evaluation grid, replay/update counts,
single-critic DDPG checkpoint, config and RNG checkpoint state, replay-query
mode, support assignment, finite required metrics, normalized critic-weight
mean, and an exact replay-bin match rate of one on every finite post-warmup row.

## Analysis and compact artifact

HER and current-policy SABER Main seeds 1--5 are reused from the previously
verified compact DrawerOpen artifact; they are not retrained. Analysis uses only
`eval/env_success_mean` on the observed 10k--2M grid, with no smoothing,
interpolation, forward fill, or extrapolation. The primary contrast is Main
minus Replay-action Query. Ordered exhaustive paired bootstrap uses all
`5^5 = 3125` resamples, and exact two-sided sign flip uses all `2^5 = 32`
assignments.

The final compact artifact is written to:

```text
metaworld_Drawer_Open_DDPG/paper_artifacts/
  drawer_open_ddpg_saber_replay_action_query_2m/
```

It excludes checkpoints, full raw metrics, JSONL, logs, cache files, correctness
runs, and smoke runs. Results for all five seeds and every paired statistic are
reported regardless of direction. If the paired confidence interval includes
zero, the README and paper summary must not claim that an independent
current-policy-query effect has been established.

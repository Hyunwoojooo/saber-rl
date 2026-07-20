# DrawerOpen DDPG + HER + Full PER

This folder is the frozen paper experiment package for Meta-World `drawer-open-v3`.
It transfers the validated FPP Full PER mechanism to the matched Drawer
DDPG+HER backbone; it is not the earlier PER-style critic-loss control.

## Experimental Protocol

| Setting | Value |
|---|---:|
| Environment | `drawer-open-v3`, sparse goal reward |
| Backbone | DDPG |
| Budget / horizon | 2,000,000 steps / 200 |
| Seeds | 1, 2, 3, 4, 5 |
| Batch / replay capacity | 256 / 300,000 |
| Future HER probability | 0.8 |
| Updates per environment step | 1 |
| Actor / critic learning rate | 3e-4 / 3e-4 |
| PER alpha | 0.6 |
| PER beta | 0.4 to 1.0 over 2,000,000 steps |
| Priority epsilon | 1e-6 |
| Evaluation | 20 deterministic episodes every 10,000 steps |

Actor and critic networks, exploration, Bellman targets, HER reward
recomputation, terminal handling, optimization settings, and evaluation grid
match the corresponding Drawer HER/SABER backbone. Full PER changes replay-slot
sampling and applies importance-sampling (IS) correction to the critic only.
The frozen `agent.py`, `networks.py`, and `envs.py` files are byte-identical to
`metaworld_Drawer_Open_DDPG/code/src`. The local `replay.py` is byte-identical to the validated FPP
Full PER replay at `experiment_FPP/td3_her_per_full_ls512/code/src/metacog_lite_v0/replay.py`; the parity test enforces both contracts.

## Full PER Semantics

1. Replay slots are sampled with replacement using
   `P(i) proportional to (priority_i + eps)^alpha`.
2. Future-goal HER is applied after the replay slots have been sampled.
3. IS coefficients `(N * P(i))^(-beta)` are normalized by the sampled-batch
   maximum and then to unit minibatch mean, matching the validated FPP package.
4. The IS coefficients weight critic residuals only; actor weights remain one.
5. Priorities are updated after learning from the single-critic absolute TD error computed
   with the same Bellman target used by that optimizer step.
6. Duplicate replay slots use the maximum new priority, new slots start at the
   current maximum priority, and priority state is checkpointed for resume.

For DDPG, the priority update uses the absolute TD error from its single critic. SABER support lookup, support caches, PER-style loss weighting, and
high replay ratio are disabled.

## Validation

```bash
.venv/bin/python metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-PER-Full-2M/code/src/test_per_full.py
.venv/bin/python metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-PER-Full-2M/test_run_config_parity.py
bash -n metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-PER-Full-2M/run_all_seeds.sh
bash -n metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-PER-Full-2M/run_5seeds_parallel.sh
.venv/bin/python metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-PER-Full-2M/run.py --seed 1 --dry-run
.venv/bin/python metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-PER-Full-2M/run.py --seed 1 --device cpu --validate_only
```

A smoke run has its own `_smoke` run name and cannot overwrite the full run:

```bash
.venv/bin/python metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-PER-Full-2M/run.py --seed 1 --device cuda --smoke
```

Run one full seed, all seeds sequentially, or five seeds in parallel:

```bash
.venv/bin/python metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-PER-Full-2M/run.py --seed 1 --steps 2000000 --device cuda
bash metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-PER-Full-2M/run_all_seeds.sh --steps 2000000 --device cuda
GPU_IDS=0,1,2,3,4 bash metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-PER-Full-2M/run_5seeds_parallel.sh --steps 2000000
```

Runs are written to `metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-PER-Full-2M/runs/ddpg_her_per_full_2m/`. Each seed uses the
run name `drawer_open_ddpg_her_per_full_s{seed}_2000000` and produces the trainer's
`config.json`, `env_snapshot.json`, `metrics.csv`, `summary.json`, `run.log`, and
rolling `checkpoint.pt` artifacts. Full 2M training is intentionally not started
by the validation commands above.

# DrawerOpen TD3 + HER + Full PER

This folder is the frozen paper experiment package for Meta-World `drawer-open-v3`.
It transfers the validated FPP Full PER mechanism to the matched Drawer
TD3+HER backbone; it is not the earlier PER-style critic-loss control.

## Experimental Protocol

| Setting | Value |
|---|---:|
| Environment | `drawer-open-v3`, sparse goal reward |
| Backbone | TD3 |
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
`metaworld_Drawer_Open_TD3/B3_v4_512_s1_5/code/src`. The local `replay.py` is byte-identical to the validated FPP
Full PER replay at `experiment_FPP/td3_her_per_full_ls512/code/src/metacog_lite_v0/replay.py`; the parity test enforces both contracts.

## Full PER Semantics

1. Replay slots are sampled with replacement using
   `P(i) proportional to (priority_i + eps)^alpha`.
2. Future-goal HER is applied after the replay slots have been sampled.
3. IS coefficients `(N * P(i))^(-beta)` are normalized by the sampled-batch
   maximum and then to unit minibatch mean, matching the validated FPP package.
4. The IS coefficients weight critic residuals only; actor weights remain one.
5. Priorities are updated after learning from the mean absolute TD error of the two critics computed
   with the same Bellman target used by that optimizer step.
6. Duplicate replay slots use the maximum new priority, new slots start at the
   current maximum priority, and priority state is checkpointed for resume.

For TD3, the same IS coefficient is applied to both critic residuals and the priority update uses their mean absolute TD error. SABER support lookup, support caches, PER-style loss weighting, and
high replay ratio are disabled.

## Completed Results

The 5-seed, 2M-step experiment completed on 2026-07-22 KST.

| Seed | Final Success | Best Success | AUC Success | First Hit 90% | Speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.00 | 1.00 | 0.632 | 780k | 34.51 steps/s |
| 2 | 0.00 | 0.00 | 0.000 | - | 34.83 steps/s |
| 3 | 0.00 | 0.00 | 0.000 | - | 34.64 steps/s |
| 4 | 1.00 | 1.00 | 0.756 | 520k | 34.42 steps/s |
| 5 | 1.00 | 1.00 | 0.644 | 720k | 34.52 steps/s |
| Mean | 0.60 | 0.60 | 0.406 | 673k* | 34.58 steps/s |

`*` Mean First Hit 90% is computed over the three seeds that reached the
threshold. Full PER improves descriptively over TD3+HER B0 (AUC 0.127, Hit90
coverage 1/5) and has a higher mean AUC than SABER/B3-Critic-Only (0.406 versus
0.362) in this cohort. Full PER and SABER solve the same three seeds (1, 4, and
5), so the result does not remove the high seed variance of DrawerOpen TD3.

The committed result package is at
`../../results_2m/TD3-HER-PER-Full-2M/` and contains each seed's `metrics.csv`
and `summary.json`; checkpoints and verbose logs are intentionally excluded.

## Validation

```bash
.venv/bin/python metaworld_Drawer_Open_TD3/experiments/TD3-HER-PER-Full-2M/code/src/test_per_full.py
.venv/bin/python metaworld_Drawer_Open_TD3/experiments/TD3-HER-PER-Full-2M/test_run_config_parity.py
bash -n metaworld_Drawer_Open_TD3/experiments/TD3-HER-PER-Full-2M/run_all_seeds.sh
bash -n metaworld_Drawer_Open_TD3/experiments/TD3-HER-PER-Full-2M/run_5seeds_parallel.sh
.venv/bin/python metaworld_Drawer_Open_TD3/experiments/TD3-HER-PER-Full-2M/run.py --seed 1 --dry-run
.venv/bin/python metaworld_Drawer_Open_TD3/experiments/TD3-HER-PER-Full-2M/run.py --seed 1 --device cpu --validate_only
```

A smoke run has its own `_smoke` run name and cannot overwrite the full run:

```bash
.venv/bin/python metaworld_Drawer_Open_TD3/experiments/TD3-HER-PER-Full-2M/run.py --seed 1 --device cuda --smoke
```

Run one full seed, all seeds sequentially, or five seeds in parallel:

```bash
.venv/bin/python metaworld_Drawer_Open_TD3/experiments/TD3-HER-PER-Full-2M/run.py --seed 1 --steps 2000000 --device cuda
bash metaworld_Drawer_Open_TD3/experiments/TD3-HER-PER-Full-2M/run_all_seeds.sh --steps 2000000 --device cuda
GPU_IDS=0,1,2,3,4 bash metaworld_Drawer_Open_TD3/experiments/TD3-HER-PER-Full-2M/run_5seeds_parallel.sh --steps 2000000
```

Runs are written to `metaworld_Drawer_Open_TD3/experiments/TD3-HER-PER-Full-2M/runs/td3_her_per_full_2m/`. Each seed uses the
run name `drawer_open_td3_her_per_full_s{seed}_2000000` and produces the trainer's
`config.json`, `env_snapshot.json`, `metrics.csv`, `summary.json`, `run.log`, and
rolling `checkpoint.pt` artifacts. The validation commands above do not start a
new full run; the completed 2M results are preserved separately as described above.

# DrawerClose TD3 Attentive Experience Replay

This folder is an independent, frozen experiment snapshot for a matched no-HER
transfer of Attentive Experience Replay (AER) to sparse MetaWorld
`drawer-close-v3`.

## Experimental Arms

| Arm | Replay batch | Actor/critic learning rate | Purpose |
|---|---|---|---|
| `TD3-ER` | Uniform 256 | Fixed `3e-4` | No-HER TD3 control |
| `TD3-AnnealingER` | Uniform 256 | `lambda_t * 3e-4` | Learning-rate schedule control |
| `TD3-AER` | Uniform `ceil(lambda_t * 256)` candidates, raw-cosine top 256 | `lambda_t * 3e-4` | AER sampling plus its LR schedule |

All arms use `HER ratio = 0`. They also use unit actor and critic loss weights;
there is no PER priority, importance-sampling correction, SABER lookup, or loss
reweighting in this experiment.

The schedule is

```text
lambda_t = 1 + (2.5 - 1) * max(1 - t / ((2/3) * T), 0)
```

At the beginning AER selects 256 transitions from 640 candidates. For a 2M-step
run, `lambda_t` reaches 1 immediately after approximately 1.333M steps. The AER
query is the latest online `[observation, desired_goal]`; replay candidates use
their stored pre-action `[observation, desired_goal]`. Similarity is raw cosine.

## Fixed Protocol

- Environment: `drawer-close-v3`, sparse reward, reward function `v2`
- Horizon: 200; budget: 2M environment steps
- Seeds: 1-5; five CUDA processes per arm
- TD3: twin critics, delayed policy update 2, target noise/clip `0.2/0.5`
- UTD: 1; batch: 256; replay capacity: 300k
- Random starts: 10k; warmup: 5k
- `gamma=0.98`, `tau=0.005`, hidden layers 256-256
- Evaluation: deterministic 20 episodes every 10k steps
- Final evaluation: separate deterministic 100 episodes; not inserted into AUC
- Timeout transitions bootstrap; success does not terminate training episodes

## Completed Result Summary

All three arms completed seeds 1-5 for 2M environment steps. Each run contains
200 evaluations on the exact 10k-step grid and a separate deterministic
100-episode final evaluation. Values below are mean +/- sample standard
deviation across five seeds.

| Experiment | Status | AUC success | Last-quarter AUC | Final success (20 ep) | Fixed final success (100 ep) | Hit90 coverage / median | Speed |
|---|---|---:|---:|---:|---:|---:|---:|
| TD3-ER | Completed | **0.2512 +/- 0.0878** | **0.2026 +/- 0.0999** | 0.2200 +/- 0.2864 | 0.2040 +/- 0.2707 | 5/5 / 180k | 48.94 steps/s |
| TD3-AnnealingER | Completed | 0.1757 +/- 0.1374 | 0.0632 +/- 0.0805 | 0.0200 +/- 0.0447 | 0.0440 +/- 0.0984 | 5/5 / 70k | 48.89 steps/s |
| TD3-AER | Completed | 0.1690 +/- 0.1095 | 0.1581 +/- 0.1844 | **0.4000 +/- 0.5477** | **0.3860 +/- 0.5291** | 5/5 / 70k | 48.30 steps/s |

All 15 runs reached 1.0 success at least once, so best success alone does not
distinguish the methods. Hit90 is also transient on this task: TD3-AER reached
Hit90 in all seeds, but only seeds 4 and 5 retained at least 0.9 success in the
fixed 100-episode final evaluation.

### Seed-Level Results

| Method | Seed | AUC | Last-quarter AUC | Final (20 ep) | Fixed final (100 ep) | First Hit90 |
|---|---:|---:|---:|---:|---:|---:|
| TD3-ER | 1 | 0.2103 | 0.2660 | 0.20 | 0.17 | 740k |
| TD3-ER | 2 | 0.1196 | 0.1280 | 0.70 | 0.67 | 500k |
| TD3-ER | 3 | 0.3425 | 0.3270 | 0.00 | 0.00 | 160k |
| TD3-ER | 4 | 0.3023 | 0.0810 | 0.00 | 0.03 | 90k |
| TD3-ER | 5 | 0.2815 | 0.2110 | 0.20 | 0.15 | 180k |
| TD3-AnnealingER | 1 | 0.2141 | 0.0410 | 0.10 | 0.22 | 70k |
| TD3-AnnealingER | 2 | 0.0364 | 0.0450 | 0.00 | 0.00 | 320k |
| TD3-AnnealingER | 3 | 0.1237 | 0.0160 | 0.00 | 0.00 | 60k |
| TD3-AnnealingER | 4 | 0.1103 | 0.0095 | 0.00 | 0.00 | 20k |
| TD3-AnnealingER | 5 | 0.3941 | 0.2045 | 0.00 | 0.00 | 70k |
| TD3-AER | 1 | 0.1744 | 0.0100 | 0.00 | 0.00 | 350k |
| TD3-AER | 2 | 0.0121 | 0.0000 | 0.00 | 0.00 | 480k |
| TD3-AER | 3 | 0.1623 | 0.2875 | 0.00 | 0.00 | 10k |
| TD3-AER | 4 | 0.1751 | 0.0780 | 1.00 | 0.93 | 70k |
| TD3-AER | 5 | 0.3214 | 0.4150 | 1.00 | 1.00 | 50k |

### Paired AUC Comparisons

Intervals use 100,000 percentile-bootstrap resamples of paired seed
differences. With only five seeds, they are reported as uncertainty summaries
rather than definitive significance tests.

| Paired comparison | Mean delta AUC | 95% bootstrap interval | Positive seeds |
|---|---:|---:|---:|
| TD3-AER - TD3-ER | -0.0822 | [-0.1444, -0.0191] | 1/5 |
| TD3-AER - TD3-AnnealingER | -0.0067 | [-0.0498, 0.0387] | 2/5 |
| TD3-AnnealingER - TD3-ER | -0.0755 | [-0.1809, 0.0299] | 2/5 |

The primary AUC result does not support an AER improvement over uniform ER in
this DrawerClose transfer. AER and AnnealingER have similar mean AUC, so the
attentive top-k sampler does not show an additional AUC benefit over its
learning-rate schedule control. AER has the highest mean final success because
two seeds remain successful at 2M, but the very large standard deviation and
three collapsed seeds preclude a stability claim.

These no-HER arms are an external pipeline-level baseline. They must not be
presented as a matched post-HER comparison against TD3+HER or SABER.

## Run

```bash
bash metaworld_Drawer_Close_TD3_AER/TD3-ER-2M.sh
bash metaworld_Drawer_Close_TD3_AER/TD3-AnnealingER-2M.sh
bash metaworld_Drawer_Close_TD3_AER/TD3-AER-2M.sh
```

To run all arms sequentially:

```bash
bash metaworld_Drawer_Close_TD3_AER/run_all_3arms_2M.sh
```

Before a full run:

```bash
.venv/bin/python metaworld_Drawer_Close_TD3_AER/run.py --method td3_aer --validate-only
.venv/bin/python metaworld_Drawer_Close_TD3_AER/run.py --method td3_aer --seed 101 --smoke
```

## Logged AER Checks

`metrics.csv` records `aer/lambda`, effective learning rates, candidate count,
candidate and selected cosine statistics, similarity uplift, duplicate rates,
transition ages, original-success transition rates, selected task phases, AER
sampling time, and observed throughput. Each run also writes `summary.json`,
`final_eval_fixed.json`, `config.json`, `env_snapshot.json`, and checkpoints.

This is a matched DrawerClose transfer of AER, not an exact reproduction of the
original paper's environments. A comparison against `TD3+HER+SABER` is a
pipeline-level comparison: AER intentionally has no HER, while SABER is a
post-HER critic-loss allocation method.

## Committed Result Artifacts

For each of the 15 completed runs, the repository preserves `metrics.csv`,
`summary.json`, `final_eval_fixed.json`, `config.json`, and `env_snapshot.json`.
Checkpoints, `metrics.jsonl`, and `run.log` are intentionally excluded because
they are large runtime artifacts and are not required to reproduce the reported
tables.

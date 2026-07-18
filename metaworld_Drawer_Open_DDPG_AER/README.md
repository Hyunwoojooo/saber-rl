# DrawerOpen DDPG AER (no HER)

This folder is an isolated, matched transfer of Attentive Experience Replay
(AER) to the existing MetaWorld DrawerOpen DDPG backbone. It does not modify
the existing DrawerOpen DDPG/HER/SABER or TD3/AER source folders.

## Experimental block

This evaluates standalone AER rather than AER on top of HER. All arms use a
single-critic DDPG agent and set the HER ratio to zero.

| Arm | Replay sampling | Actor/critic learning rate |
| --- | --- | --- |
| DDPG-ER | Uniform 256 | Fixed 3e-4 |
| DDPG-AnnealingER | Uniform 256 | lambda(t) * 3e-4 |
| DDPG-AER | ceil(lambda(t) * 256) candidates, cosine top-256 | lambda(t) * 3e-4 |

The schedule is

```text
lambda(t) = 1 + (lambda0 - 1) * max(1 - t / (alpha * T), 0),
lambda0 = 2.5, alpha = 2/3.
```

For the 2M-step budget, lambda reaches one at about 1.333M steps. Similarity
is raw cosine similarity between the latest online policy input and replay
pre-action policy inputs. A policy input is the observation concatenated with
the original desired goal. Candidate sampling uses replacement. There is no
priority table, importance-sampling correction, SABER lookup, or non-unit loss
weight.

## Matched settings

- MetaWorld `drawer-open-v3`, sparse reward, horizon 200
- DDPG deterministic actor, one critic, and one actor update per critic update
- two 256-unit hidden layers
- batch 256, replay 300k, UTD 1
- 10k random starts and 5k warmup
- gamma 0.98, tau 0.005, exploration noise 0.2
- seeds 1-5 with fixed evaluation seeds
- evaluation every 10k steps with 20 episodes
- final fixed 100-episode evaluation
- success does not terminate training; time limits bootstrap

## Validation

```bash
.venv/bin/python metaworld_Drawer_Open_DDPG_AER/code/src/test_aer_sampler.py
.venv/bin/python metaworld_Drawer_Open_DDPG_AER/code/src/test_ddpg_aer_backbone.py
.venv/bin/python metaworld_Drawer_Open_DDPG_AER/run.py --arm aer --seed 1 --validate_only --device cpu
.venv/bin/python metaworld_Drawer_Open_DDPG_AER/run.py --arm aer --seed 101 --steps 400 --smoke --device cpu --output_root /tmp/drawer_open_ddpg_aer_smoke
bash -n metaworld_Drawer_Open_DDPG_AER/run_drawer_open_ddpg_aer_noher_2m.sh
```

The smoke-only configuration lowers warmup to 200 steps and batch size to 64
so the shortened run executes attentive sampling while `lambda(t) > 1`.

## Full experiment

```bash
bash metaworld_Drawer_Open_DDPG_AER/run_drawer_open_ddpg_aer_noher_2m.sh
```

The launcher runs five seeds in parallel within an arm and executes arms
sequentially in the default order AER, ER, and AnnealingER. Override `ARMS`,
`STEPS`, or `CUDA_DEVICE` through environment variables. Outputs are written
under `runs/`, logs under `logs/`, and launcher logs under
`orchestrator_logs/`.

## Completed Results

The 2M-step seeds 1-5 block completed for all three arms. Values below are the
mean and sample standard deviation over seeds on the exact 10k evaluation grid.

| Arm | Success AUC | Final success | Best success | Hit90 | Fixed 100-episode success |
| --- | ---: | ---: | ---: | ---: | ---: |
| DDPG-ER | 0.579 +/- 0.342 | 0.800 +/- 0.447 | 0.810 +/- 0.425 | 4/5, median 525k | 0.800 +/- 0.447 |
| DDPG-AnnealingER | 0.111 +/- 0.248 | 0.200 +/- 0.447 | 0.200 +/- 0.447 | 1/5, 910k | 0.200 +/- 0.447 |
| DDPG-AER | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0/5 | 0.000 +/- 0.000 |

The paired AER-minus-ER AUC difference is -0.5794 with an exact paired
bootstrap 95% interval of [-0.7899, -0.2842]. DDPG-ER exceeds
DDPG-AnnealingER by +0.4683 AUC with interval [0.1467, 0.7899] and positive
differences on all five paired seeds. These results associate the prescribed
AER learning-rate schedule with substantial degradation, while the independent
effect of attentive sampling remains unresolved without a fixed-LR AER arm.

The compact paper artifact is stored under
`paper_artifacts/drawer_open_ddpg_aer_noher_2m/`. It includes exact-grid
curves, aggregate and paired statistics, sanitized run records, provenance,
and a SHA256 manifest. Raw metrics, logs, and checkpoints are excluded.

## Interpretation

- AER minus ER measures the full AER protocol change.
- AER minus AnnealingER isolates attentive sampling from its LR schedule.
- HER+SABER versus standalone AER is an algorithm-level comparison because
  HER differs between the methods.
- Fixed-LR ER succeeds on 4/5 seeds, so AER failure is not an absolute task
  exploration floor; a fixed-LR AER arm is still needed to isolate sampling.

# DrawerClose DDPG AER (no HER)

This folder is an isolated, matched transfer of Attentive Experience Replay
(AER) to the existing MetaWorld DrawerClose DDPG backbone. It does not modify
the existing DrawerClose DDPG/HER/SABER or DrawerOpen AER source folders.

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

- MetaWorld `drawer-close-v3`, sparse reward, horizon 200
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
.venv/bin/python metaworld_Drawer_Close_DDPG_AER/code/src/test_aer_sampler.py
.venv/bin/python metaworld_Drawer_Close_DDPG_AER/code/src/test_ddpg_aer_backbone.py
.venv/bin/python metaworld_Drawer_Close_DDPG_AER/run.py --arm aer --seed 1 --validate_only --device cpu
.venv/bin/python metaworld_Drawer_Close_DDPG_AER/run.py --arm aer --seed 101 --steps 400 --smoke --device cpu --output_root /tmp/drawer_close_ddpg_aer_smoke
bash -n metaworld_Drawer_Close_DDPG_AER/run_drawer_close_ddpg_aer_noher_2m.sh
```

The smoke-only configuration lowers warmup to 200 steps and batch size to 64
so the shortened run executes attentive sampling while `lambda(t) > 1`.

## Full experiment

```bash
bash metaworld_Drawer_Close_DDPG_AER/run_drawer_close_ddpg_aer_noher_2m.sh
```

The launcher runs five seeds in parallel within an arm and executes arms
sequentially in the default order AER, ER, and AnnealingER. Override `ARMS`,
`STEPS`, or `CUDA_DEVICE` through environment variables. Outputs are written
under `runs/`, logs under `logs/`, and launcher logs under
`orchestrator_logs/`.

## Interpretation

- AER minus ER measures the full AER protocol change.
- AER minus AnnealingER isolates attentive sampling from its LR schedule.
- HER+SABER versus standalone AER is an algorithm-level comparison because
  HER differs between the methods.
- DrawerClose is prone to transient success and late collapse, so AUC and
  last-quarter AUC should be reported alongside final success.
- If all no-HER arms remain at the sparse-reward floor, the result is an
  exploration boundary rather than evidence of universal AER failure.

## Completed Results

All three arms completed 2M steps on seeds 1-5. Values below are the mean and
sample standard deviation over five seeds; Hit90 is the median among seeds
that reached 90% evaluation success.

| Arm | Final success | Best success | Success AUC | Hit90 |
| --- | ---: | ---: | ---: | ---: |
| DDPG-ER | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.150 +/- 0.054 | 5/5, median 280k |
| DDPG-AnnealingER | 0.330 +/- 0.391 | 1.000 +/- 0.000 | 0.199 +/- 0.122 | 5/5, median 380k |
| DDPG-AER | 0.200 +/- 0.447 | 1.000 +/- 0.000 | 0.194 +/- 0.067 | 5/5, median 160k |

All arms find transient successful policies, but late collapse is common.
AER reaches Hit90 earliest but does not improve mean AUC over AnnealingER.
All paired AUC intervals among the three arms include zero.

The GitHub-sized result artifact is stored at
`paper_artifacts/drawer_close_ddpg_aer_noher_2m/`. It contains exact-grid
curves, seed and aggregate tables, paired inference, sanitized run records,
source provenance, and a SHA256 manifest. Raw runs and checkpoints remain
excluded.

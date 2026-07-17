# DrawerOpen TD3 AER (no HER)

This folder is an isolated, matched transfer of Attentive Experience Replay
(AER) to the existing MetaWorld DrawerOpen TD3 backbone. It does not modify the
existing DrawerOpen TD3/HER/SABER source folders.

## Experimental question

The block evaluates standalone AER, not AER on top of HER. All three arms set
HER ratio to zero:

| Arm | Replay sampling | Actor/critic learning rate |
| --- | --- | --- |
| TD3-ER | Uniform 256 | Fixed 3e-4 |
| TD3-AnnealingER | Uniform 256 | lambda(t) * 3e-4 |
| TD3-AER | ceil(lambda(t) * 256) candidates, cosine top-256 | lambda(t) * 3e-4 |

lambda0 is 2.5 and alpha is 2/3. With a 2M-step budget, lambda reaches 1
at approximately 1.333M steps. AER uses raw cosine similarity between the
latest online policy input and replay pre-action policy inputs, where policy
input is observation concatenated with the original desired goal. Candidate
sampling uses replacement. No priority table, importance-sampling correction,
SABER lookup, or non-unit loss weight is used.

## Matched factors

- MetaWorld drawer-open-v3, sparse reward, horizon 200
- TD3 twin critics, target smoothing 0.2/0.5, policy delay 2
- two 256-unit hidden layers
- batch 256, replay 300k, UTD 1
- 10k random starts, 5k warmup
- gamma 0.98, tau 0.005, exploration noise 0.2
- seeds 1-5 and fixed evaluation seeds
- evaluation every 10k steps with 20 episodes
- final fixed 100-episode evaluation

## Validation

    python code/src/test_aer_sampler.py
    python run.py --arm aer --seed 1 --validate_only --device cpu
    python run.py --arm aer --seed 101 --steps 5000 --smoke --device cuda
    bash -n run_drawer_open_td3_aer_noher_2m.sh

## Full experiment

    ./run_drawer_open_td3_aer_noher_2m.sh

The default launcher runs five seeds in parallel per arm and runs the arms
sequentially as AER, ER, and AnnealingER. Set ARMS to override the order.
Outputs are under runs, logs under logs, and launcher state under
orchestrator_logs.

## Completed result summary

The full no-HER three-arm block completed at 2M steps for seeds 1-5. Values
below are mean and sample standard deviation over the exact 10k evaluation
grid; the final column is a separate fixed 100-episode evaluation.

| Arm | Success AUC | Final success | Best success | Hit90 | Fixed 100-episode success |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3-ER | 0.089 +/- 0.122 | 0.400 +/- 0.548 | 0.420 +/- 0.531 | 2/5, median 1.53M | 0.400 +/- 0.548 |
| TD3-AnnealingER | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0/5 | 0.000 +/- 0.000 |
| TD3-AER | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0/5 | 0.000 +/- 0.000 |

TD3-ER succeeds late on seeds 4 and 5, whereas TD3-AER and
TD3-AnnealingER remain at zero evaluation success on every seed. Thus,
standalone AER does not resolve the sparse-exploration floor in this matched
DrawerOpen TD3 transfer. Because AER and AnnealingER are both at the floor,
the result does not isolate attentive sampling as the independent cause; that
would require an AER-FixedLR arm.

The compact, self-verifying result package is under
`paper_artifacts/drawer_open_td3_aer_noher_2m/`. Checkpoints, full training
metrics, JSONL streams, and raw logs are intentionally excluded.

## Interpretation

- AER minus ER measures the end-to-end AER change.
- AER minus AnnealingER isolates attentive sampling from the prescribed
  learning-rate schedule.
- HER+SABER versus standalone AER is an algorithm-level comparison. Because
  HER differs, it is not evidence that one intervention mechanism is
  intrinsically superior.
- If all no-HER arms remain at the sparse-reward floor, the result identifies
  an exploration boundary and must not be presented as universal AER failure.

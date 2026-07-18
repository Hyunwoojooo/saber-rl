# FPP TD3 + HER + HGR

This folder is an isolated, matched TD3 transfer of the original two-stage
Hindsight Goal Ranking replay rule.  The TD3 actor-critic and ordinary HER
settings match the completed FetchPickAndPlace TD3+HER baseline.  HGR changes
only the 80% relabeled branch: it ranks completed episodes and then valid
transition/future-goal pairs using TD error, with the corresponding
importance-sampling (IS) correction on the critic loss.

This is a matched TD3 transfer, not an exact reproduction of the single-critic
HGR paper implementation.

## Frozen common configuration

| Setting | Value |
|---|---:|
| Environment | `FetchPickAndPlace-v4`, sparse reward |
| Horizon | 50 |
| Steps / seeds | 1,000,000 / 1--5 |
| Batch / replay | 256 / 300,000 |
| HER ratio | 0.8 |
| Start / warmup | 10,000 / 5,000 |
| UTD | 1 |
| Gamma / tau | 0.98 / 0.005 |
| Actor / critic LR | 3e-4 / 3e-4 |
| Hidden layers | 256, 256 |
| Exploration noise | 0.2 |
| TD3 policy delay | 2 |
| Target noise / clip | 0.2 / 0.5 |
| Evaluation | deterministic 20 episodes every 10k |
| Terminal convention | success nonterminal; timeout bootstraps |

SABER lookup/weights, proportional PER, and high replay ratio are disabled.

## HGR rule

For every completed episode, HGR stores all valid pairs
`(transition j, future goal i)` with `j < i`.  At FPP horizon 50 this gives
1,275 pair priorities per episode.

```text
P(episode)       proportional to mean(pair_priority)^0.6
P(pair|episode)  proportional to pair_priority^0.6
pair_priority    = reduced pre-update TD error + 1e-6
```

The episode and pair IS exponents both anneal linearly from 0.4 to 1.0 over
critic-update progress.  The product of the two normalized IS weights is
applied identically to both TD3 critic losses.  Actor weights remain one, and
the actor/targets retain the normal delayed TD3 update.

### Twin-critic priority convention

The primary run is frozen to:

```text
min_abs = min(abs(delta_q1), abs(delta_q2))
```

The value is captured before the optimizer step from the exact same smoothed
target used in that update; the target is never recomputed for priority
updates.  This preserves the local TD3 TD-error/PER-style convention.  The code
also supports `mean_abs`, but it is not part of this primary run and must use a
separately copied config, experiment ID, and output root if run as sensitivity.

## Validation

```bash
bash experiment_FPP/td3_hgr_tdpriority_1m/run_validate_seed1.sh
```

This runs the original HGR probability/IS/checkpoint tests, TD3-HGR integration
tests, runner protocol tests, and a real FetchPickAndPlace validation rollout
on CPU.

The launchers intentionally do not use the trainer's legacy `--smoke` flag,
because that flag changes batch, warmup, and replay settings.  Only total steps
and the isolated output root change.

```bash
# Exact-config 5k correctness run
DEVICE=cpu bash experiment_FPP/td3_hgr_tdpriority_1m/run_correctness_5k_seed1.sh

# Exact-config 20k smoke, seeds 1--5 in parallel
DEVICE=cuda:0 bash experiment_FPP/td3_hgr_tdpriority_1m/run_smoke_20k_seeds1_5_5parallel.sh

# Main 1M, seeds 1--5 in parallel
DEVICE=cuda:0 bash experiment_FPP/td3_hgr_tdpriority_1m/run_main_1m_seeds1_5_5parallel.sh
```

Every requested step count must be a multiple of the horizon.  This keeps
checkpoints at episode boundaries.  The wrapper refuses to overwrite an
existing run; use `--resume` only with a matching checkpoint.

## Outputs

Main outputs are isolated under:

```text
experiment_FPP/runs/td3_hgr_tdpriority_1m/
  fetch_pick_and_place_td3_hgr_tdpriority_s{seed}_1000000/
```

Each run retains `config.json`, `env_snapshot.json`, `metrics.csv`,
`metrics.jsonl`, `run.log`, rolling/final `checkpoint.pt`, best checkpoint, and
`summary.json`.  Checkpoints include both TD3 critics/targets, optimizer and
delay state, replay, HGR tables/beta progress, NumPy RNGs, and Torch CPU/CUDA
RNG states.

Use normalized trapezoidal success AUC on the exact 10k grid as the primary
endpoint.  Smoke performance must not be used to tune HGR or TD3 settings.

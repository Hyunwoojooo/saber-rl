# FPP SAC + HER + HGR

This folder is an isolated, matched SAC transfer of the original two-stage
Hindsight Goal Ranking replay rule. The SAC actor-critic and ordinary HER
settings match the completed FetchPickAndPlace SAC+HER baseline. HGR changes
only the 80% relabeled branch: it ranks completed episodes and then valid
transition/future-goal pairs using TD error, with the corresponding
importance-sampling (IS) correction on the critic loss.

This is a matched SAC transfer, not an exact reproduction of the original
single-critic HGR implementation.

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
| SAC alpha | 0.2 initial, automatic tuning |
| Target entropy | `-action_dim` (`-4` on FPP) |
| Evaluation | deterministic 20 episodes every 25k |
| Terminal convention | success nonterminal; timeout bootstraps |

SABER lookup/weights, proportional PER, and high replay ratio are disabled.
The 25k evaluation interval deliberately follows the completed FPP SAC+HER
baseline so the main comparison has the same evaluation grid.

## HGR rule

For every completed episode, HGR stores all valid pairs
`(transition j, future goal i)` with `j < i`. At FPP horizon 50 this gives
1,275 pair priorities per episode.

```text
P(episode)       proportional to mean(pair_priority)^0.6
P(pair|episode)  proportional to pair_priority^0.6
pair_priority    = reduced pre-update SAC TD error + 1e-6
```

The episode and pair IS exponents both anneal linearly from 0.4 to 1.0 over
995,001 critic updates. The product of the two normalized IS weights is
applied identically to both SAC critic losses. Actor and temperature losses
are unweighted.

### Twin-critic priority convention

The primary run is frozen to:

```text
min_abs = min(abs(delta_q1), abs(delta_q2))
```

Both residuals are captured before the optimizer step from the exact same
entropy-regularized SAC target used by the critic update. The target is not
recomputed for the priority update. The code supports `mean_abs` only as a
separately tagged sensitivity experiment; it must not overwrite this primary
run's config or output root.

## Validation and execution

```bash
# Unit/integration checks plus a real FPP validation rollout
bash experiment_FPP/sac_hgr_tdpriority_1m/run_validate_seed1.sh

# Exact-config 5k correctness run
DEVICE=cpu bash experiment_FPP/sac_hgr_tdpriority_1m/run_correctness_5k_seed1.sh

# Exact-config 25k smoke, seeds 1--5 in parallel
DEVICE=cuda:0 bash experiment_FPP/sac_hgr_tdpriority_1m/run_smoke_25k_seeds1_5_5parallel.sh

# Main 1M, seeds 1--5 in parallel
DEVICE=cuda:0 bash experiment_FPP/sac_hgr_tdpriority_1m/run_main_1m_seeds1_5_5parallel.sh
```

The launchers do not use the trainer's legacy `--smoke` flag because that flag
changes batch, warmup, and replay settings. Only total steps and the isolated
output root change. Every step count must be a multiple of the 50-step horizon.
The wrapper refuses to overwrite an existing run; resume requires an explicit
matching checkpoint.

## Outputs

Main outputs are isolated under:

```text
experiment_FPP/runs/sac_hgr_tdpriority_1m/
  fetch_pick_and_place_sac_hgr_tdpriority_s{seed}_1000000/
```

Each run retains the frozen config, environment snapshot, CSV/JSONL metrics,
log, summary, and rolling/final checkpoint. Checkpoints include both SAC
critics/targets, actor and optimizers, entropy temperature state, replay, HGR
tables/beta progress, NumPy RNGs, and Torch CPU/CUDA RNG states.

Use normalized trapezoidal success AUC on the exact 25k grid as the primary
endpoint. Smoke performance must not be used to tune SAC or HGR settings.

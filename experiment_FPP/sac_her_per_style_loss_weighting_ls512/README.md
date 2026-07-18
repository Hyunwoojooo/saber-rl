# FPP SAC + HER + PER-style Loss Weighting

This folder is an isolated, matched SAC transfer of the existing
FetchPickAndPlace PER-style loss-weighting baseline. The SAC actor-critic,
ordinary HER replay, and evaluation protocol match the completed FPP
SAC+HER baseline. Only the two critic losses receive batch-local weights
derived from the current pre-update SAC TD residuals.

This is **not full PER**. Replay indices remain uniformly sampled, priorities
are not stored in replay, and there is no PER importance-sampling beta
correction. It is also not HGR and does not construct HGR episode/pair tables.

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

The 25k evaluation interval follows the completed SAC+HER baseline. The
`ls512` tag is retained for comparison naming only; all SABER support
lookup, caches, and local search are disabled.

## PER-style rule

For each uniformly sampled HER batch, one entropy-regularized SAC target is
drawn and reused for both weighting and the actual critic update:

```text
delta1_i  = Q1(s_i, a_i) - y_i
delta2_i  = Q2(s_i, a_i) - y_i
td_i      = min(abs(delta1_i), abs(delta2_i))
priority  = (td_i + 1e-6)^0.6
p_norm    = priority / mean(priority)
w_raw     = clip(p_norm, 0, 2)
omega     = w_raw / mean(w_raw)
```

The same `omega` is applied to both critic MSE terms. The actor and entropy
temperature losses are unweighted. The cap of 2 applies to `w_raw`, before
the final mean-one normalization; the final `omega.max()` can therefore be
greater than 2.

The primary twin-critic convention is frozen to `min_abs`. The source
supports `mean_abs` only for a separately tagged sensitivity experiment;
it must not overwrite this primary config or output root.

## Validation and execution

```bash
# Formula/integration tests plus a real FPP validation rollout
bash experiment_FPP/sac_her_per_style_loss_weighting_ls512/run_validate_seed1.sh

# Exact-config 5k correctness run
DEVICE=cpu bash experiment_FPP/sac_her_per_style_loss_weighting_ls512/run_correctness_5k_seed1.sh

# Exact-config 25k smoke, seeds 1--5 in parallel
DEVICE=cuda:0 bash experiment_FPP/sac_her_per_style_loss_weighting_ls512/run_smoke_25k_seeds1_5_5parallel.sh

# Main 1M, seeds 1--5 in parallel
DEVICE=cuda:0 bash experiment_FPP/sac_her_per_style_loss_weighting_ls512/run_main_1m_seeds1_5_5parallel.sh
```

The launchers intentionally do not use the trainer's legacy `--smoke` flag,
because that flag changes batch size, warmup, and replay capacity. Only the
step count and isolated output root change. Every step count must be a
multiple of the 50-step horizon. The wrapper refuses to overwrite an
existing run. Resume requires an explicit existing checkpoint and rejects
mismatched methods, seeds, core hyperparameters, or non-episode-boundary
checkpoints.

## Outputs

Main outputs are isolated under:

```text
experiment_FPP/runs/sac_her_per_style_loss_weighting_ls512/
  fetch_pick_and_place_sac_her_per_style_loss_weighting_ls512_s{seed}_1000000/
```

Each run retains the frozen config, environment snapshot, CSV/JSONL metrics,
log, summary, and rolling/final checkpoint. Checkpoints include both SAC
critics/targets, actor and optimizers, entropy-temperature state, replay,
NumPy RNGs, and Torch CPU/CUDA RNG states. The HGR checkpoint field must be
`None`.

Use normalized trapezoidal success AUC on the exact 25k grid as the primary
endpoint. Correctness/smoke performance must not be used to tune the frozen
PER-style or SAC settings.

## Packaged results

The completed 1M-step runs for seeds 1--5 are packaged under
`experiment_FPP/paper_artifacts/fpp_sac_her_per_style_loss_weighting_ls512_1m`.
Regenerate the compact, validated artifact from the repository root with:

```bash
.venv/bin/python experiment_FPP/package_sac_her_per_style_results.py
```

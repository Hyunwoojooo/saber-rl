# FPP SAC + HER + Full PER

This experiment adds slot-level Prioritized Experience Replay to the matched
FetchPickAndPlace-v4 SAC+HER backbone.

## Matched configuration

| Setting | Value |
|---|---:|
| Environment | FetchPickAndPlace-v4, sparse reward |
| Steps | 1,000,000 |
| Seeds | 1, 2, 3, 4, 5 |
| Start / warmup steps | 10,000 / 5,000 |
| Batch / replay capacity | 256 / 300,000 |
| HER ratio | 0.8 |
| Updates per environment step | 1 |
| SAC alpha | 0.2 initial, automatically tuned |
| Target entropy | -action_dim (-4 for FPP) |
| Evaluation | 20 episodes every 25,000 steps |
| PER alpha | 0.6 |
| PER beta | 0.4 to 1.0 over 1,000,000 environment steps |
| Priority epsilon | 1e-6 |

The Gaussian actor, entropy-regularized stochastic target, automatic
temperature update, twin critics, environment, HER reward recomputation,
terminal convention, seed streams, and evaluation grid match the existing FPP
SAC+HER baseline. The local-search value 512 is kept only in run names for
comparison parity; SABER support lookup is disabled.

## Full-PER semantics

The priority update uses both SAC critics and the exact stochastic target
sample used by the same optimizer step:

    y_i = r_i + gamma * nonterminal_i * (min(Q1_target, Q2_target) - alpha * log_pi)
    priority_i = 0.5 * (abs(Q1_i - y_i) + abs(Q2_i - y_i)) + epsilon
    P(i) = priority_i^PER_alpha / sum_j priority_j^PER_alpha
    IS_i = (N * P(i))^(-PER_beta)

IS weights are normalized by the largest weight in the sampled minibatch and,
with `weighting_mode=batch_norm`, divided by their minibatch mean. They are
applied to both critic MSE terms. Actor and entropy-temperature losses remain
unweighted, although both still observe the prioritized minibatch state
distribution.

- Replay slots are priority-sampled before future-goal HER relabeling.
- New or overwritten slots receive the current maximum priority.
- Repeated slots in one minibatch are updated with the largest new priority.
- Priority arrays, maximum priority, NumPy sampling RNG, PyTorch CPU/CUDA RNG,
  SAC optimizer state, temperature state, and beta progress are checkpointed.
- Canonical evaluation and periodic checkpoints occur on 50-step FPP episode
  boundaries, which keeps resume behavior exact at those checkpoints.

## Validation and execution

Run the focused tests:

    .venv/bin/python experiment_FPP/sac_her_per_full_ls512/code/src/test_sac_per_full.py
    .venv/bin/python experiment_FPP/sac_her_per_full_ls512/test_run_config_parity.py

Print the canonical command:

    .venv/bin/python experiment_FPP/sac_her_per_full_ls512/run.py --seed 1 --dry-run

Validate the real environment wrapper:

    .venv/bin/python experiment_FPP/sac_her_per_full_ls512/run.py --seed 1 --device cpu --validate_only

Run a 1,000-step smoke:

    .venv/bin/python experiment_FPP/sac_her_per_full_ls512/run.py --seed 1 --device cuda --smoke

Run one full seed or all five seeds:

    .venv/bin/python experiment_FPP/sac_her_per_full_ls512/run.py --seed 1 --steps 1000000 --device cuda
    bash experiment_FPP/sac_her_per_full_ls512/run_all_seeds.sh --steps 1000000 --device cuda

Queue five parallel SAC seeds after the five TD3 full-PER seeds finish
successfully:

    tmux new-session -d -s fpp_sac_per_full_after_td3 \
      "bash experiment_FPP/sac_her_per_full_ls512/queue_after_td3_per_full.sh"

The queue validates all five TD3 `summary.json` files before launching SAC. If
any TD3 seed exits without a valid completion summary, or SAC output already
exists, it logs the error and does not launch SAC.

Resume an interrupted seed from its periodic checkpoint:

    .venv/bin/python experiment_FPP/sac_her_per_full_ls512/run.py --seed 1 --steps 1000000 --device cuda --resume --resume_path experiment_FPP/runs/sac_her_per_full_ls512/fetch_pick_and_place_sac_her_per_full_ls512_s1_1000000_ls512/checkpoint.pt

Results are written below `experiment_FPP/runs/sac_her_per_full_ls512`.

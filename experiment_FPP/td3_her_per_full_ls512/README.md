# FPP TD3 + HER + Full PER

This experiment adds slot-level Prioritized Experience Replay to the matched
FetchPickAndPlace-v4 TD3+HER backbone.

## Matched configuration

| Setting | Value |
|---|---:|
| Environment | FetchPickAndPlace-v4, sparse reward |
| Steps | 1,000,000 |
| Seeds | 1, 2, 3, 4, 5 |
| Batch / replay capacity | 256 / 300,000 |
| HER ratio | 0.8 |
| Updates per environment step | 1 |
| TD3 policy delay | 2 |
| TD3 target policy noise / clip | 0.2 / 0.5 |
| PER alpha | 0.6 |
| PER beta | 0.4 to 1.0 over 1,000,000 environment steps |
| Priority epsilon | 1e-6 |

The actor, twin critics, target smoothing, delayed actor update, environment,
HER reward recomputation, terminal convention, seed streams, and evaluation
grid match the existing FPP TD3+HER experiment. The local-search value 512 is
kept only in run names for comparison parity; SABER support lookup is disabled.

## Full-PER semantics

For a sampled transition, the priority update uses the mean absolute TD error
of the two TD3 critics computed from the same target used by that optimizer
step:

    priority_i = 0.5 * (abs(delta1_i) + abs(delta2_i)) + epsilon
    P(i) = priority_i^alpha / sum_j priority_j^alpha
    IS_i = (N * P(i))^(-beta)

IS weights are normalized by the largest weight in the sampled minibatch and,
with weighting_mode=batch_norm, divided by their minibatch mean. The resulting
weights are applied to both FPP TD3 critic MSE terms. Actor loss weights remain
one, although the actor still observes the prioritized minibatch state
distribution.

- Replay slots are priority-sampled before future-goal HER relabeling.
- New or overwritten slots receive the current maximum priority.
- Repeated slots in one minibatch are updated with the largest new priority.
- Priority arrays, maximum priority, sampling RNG, TD3 target-noise RNG, and
  beta progress are checkpointed for resume.

## Validation and execution

Run the focused unit tests:

    .venv/bin/python experiment_FPP/td3_her_per_full_ls512/code/src/test_td3_per_full.py
    .venv/bin/python experiment_FPP/td3_her_per_full_ls512/test_run_config_parity.py

Print the canonical command:

    .venv/bin/python experiment_FPP/td3_her_per_full_ls512/run.py --seed 1 --dry-run

Validate the real environment wrapper:

    .venv/bin/python experiment_FPP/td3_her_per_full_ls512/run.py --seed 1 --device cpu --validate_only

Run a 1,000-step smoke:

    .venv/bin/python experiment_FPP/td3_her_per_full_ls512/run.py --seed 1 --device cuda --smoke

Run one full seed or all five seeds:

    .venv/bin/python experiment_FPP/td3_her_per_full_ls512/run.py --seed 1 --steps 1000000 --device cuda
    bash experiment_FPP/td3_her_per_full_ls512/run_all_seeds.sh --steps 1000000 --device cuda

Resume an interrupted seed from its checkpoint:

    .venv/bin/python experiment_FPP/td3_her_per_full_ls512/run.py --seed 1 --steps 1000000 --device cuda --resume --resume_path experiment_FPP/runs/td3_her_per_full_ls512/fetch_pick_and_place_td3_her_per_full_ls512_s1_1000000_ls512/checkpoint.pt

Results are written below experiment_FPP/runs/td3_her_per_full_ls512.

# DrawerOpen DDPG Shuffled-Policy Query Ablation

This folder is a self-contained code snapshot for one DDPG + future-goal HER query ablation on Meta-World DrawerOpen. It follows the FPP support-query implementation and changes only support_query_mode.

## One-Factor Definition

- Environment: drawer-open-v3, sparse reward, reward function v2
- Backbone: DDPG
- Replay: uniform replay with future-goal HER probability 0.8
- Training: 2,000,000 environment steps, horizon 200, UTD 1
- Seeds: 1-5, launched as five CUDA processes
- Main method: b3_support_runtime_opt
- Weight scope: critic_only
- Weight normalization: minibatch mean one
- Fixed support settings: M_B=16, M_Q_min=4, N_local_min=4, local_search_max=512, five bins per action dimension
- Fixed query mode: shuffled_policy

The query action is a batch-deranged current-policy action from another minibatch row. This preserves the policy-action marginal distribution while removing sample-level state-goal/action alignment. HER relabeling, sampled transitions, rewards, Bellman targets, DDPG actor objective, network architecture, optimizer, and update budget remain unchanged.

The permutation has no fixed points and uses the checkpointed weight RNG, matching the FPP shuffled-policy control.

## Run

From the repository root:

    bash metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-B3-Shuffled-Policy-Query-2M/DDPG-HER-B3-Shuffled-Policy-Query-2M.sh

The runner uses CUDA device 0 by default. Override CUDA_DEVICE, STEPS, OUTPUT_ROOT, or LOG_ROOT through environment variables when needed.

## Validation

No training is started by these checks:

    PYTHONPATH=metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-B3-Shuffled-Policy-Query-2M/code/src .venv/bin/python metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-B3-Shuffled-Policy-Query-2M/code/src/test_support_query_modes.py
    .venv/bin/python metaworld_Drawer_Open_DDPG/experiments/DDPG-HER-B3-Shuffled-Policy-Query-2M/code/src/train_drawer_open_lite_v0.py --validate_only --device cpu --method b3_support_runtime_opt --weight_scope critic_only --weighting_mode batch_norm --support_query_mode shuffled_policy

The synthetic test verifies expected current-policy, replay-action, and shuffled-policy support masks, cache on/off parity, replay-query bin agreement, deterministic derangement behavior, and rejection of invalid modes.

## Logged Checks

The run records support_query_mode in config, CSV rows, checkpoints, and summary. It also records query-bin agreement with current-policy and replay-action bins, context eligibility, support-positive rate, query-bin presence given context, weight statistics, cache statistics, and scanner audit metrics.

For replay_action, query/query_bin_matches_replay_action_rate must be 1. For shuffled_policy, the sample permutation has no fixed points, although discretized action bins can still coincide.

## Outputs

Results are written below results_2m and logs below logs_2m inside this folder unless overridden. Each seed stores config.json, env_snapshot.json, metrics.csv, checkpoint.pt, and summary.json according to the shared Drawer training code.

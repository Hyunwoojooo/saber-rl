# FPP DDPG-HGR Original Pair-Priority Baseline

This folder isolates the FetchPickAndPlace DDPG+HER baseline with the two-stage
Hindsight Goal Ranking (HGR) replay rule from
[HGR](https://arxiv.org/abs/2110.15043). HGR is not an RL backbone. The DDPG
actor-critic remains matched to the local FPP DDPG+HER B0 experiment; HGR only
changes how the HER branch samples completed episodes and transition/future-goal
pairs, plus the corresponding importance-sampling correction.

## Matched DDPG+HER Configuration

| Setting | Value |
|---|---:|
| Environment | `FetchPickAndPlace-v4` |
| Reward | sparse |
| Horizon | 50 |
| Total steps | 1,000,000 |
| Seeds | 1, 2, 3, 4, 5 |
| Batch size | 256 |
| Replay capacity | 300,000 |
| HER ratio | 0.8 |
| Updates per step | 1 |
| Gamma / tau | 0.98 / 0.005 |
| Actor / critic LR | 3e-4 / 3e-4 |
| Hidden dimension | 256 |
| Exploration noise | 0.2 |
| Start / warmup | 10,000 / 5,000 |
| Evaluation | 20 episodes every 10,000 steps |
| Success terminates | false |
| Timeout bootstrap | true |

Actor/critic architecture, optimizers, target calculation, exploration, HER
reward recomputation, and terminal handling are unchanged. SABER lookup and
weights, PER, and high replay ratio are disabled. `weight_scope=none` refers to
SABER; HGR's own IS correction is carried separately in `hgr_is_weight`.

## HGR Pair Tables

For each completed episode `e`, the replay sampler creates every valid pair:

```text
P_e = {(j, i) | 0 <= j < i <= H}
p[e,j,i] = abs(TD(s_j, a_j, g_i)) + epsilon
episode_priority[e] = mean_(j,i) p[e,j,i]
```

At horizon 50 this is 1,275 float32 pair priorities per episode. In storage,
`next_achieved_goal[k]` is `g_(k+1)`, so pair-table positions use transition
slot `j` and goal-source slot `k >= j` while still representing `j < i`.

New pairs use the current global maximum priority. A critic update changes only
the sampled pair entries. Each table tracks its priority sum, so updating the
episode mean is O(1). Overwriting any transition removes the overwritten
episode's complete pair table; the new episode is finalized only when done.

## Two-Stage Sampling And IS

```text
P(e) proportional to episode_priority[e]^alpha_episode
P(j,i | e) proportional to p[e,j,i]^alpha_goal

w_e  = (1 / (N_episode * P(e)))^beta_episode
w_ji = (1 / (|P_e| * P(j,i | e)))^beta_goal
w    = w_e * w_ji
```

The 80% HER branch uses HGR episode/pair sampling. Its combined IS weights are
normalized by the maximum relabeled weight in the batch and applied only to the
critic MSE. The actor weights are always one. The 20% original-goal branch is
uniform replay with critic and actor weight one.

Fixed HGR settings:

- `alpha_episode = 0.6`
- `alpha_goal = 0.6`
- `beta_episode_initial = beta_goal_initial = 0.4`
- both beta values linearly anneal to `1.0` over critic-update progress
- `priority_eps = 1e-6`
- `priority_init = 1.0`
- `uniform_mix = 0`
- priority clipping disabled (`hgr_priority_clip_max = 0`)

The priority update reuses the raw per-sample absolute TD error computed inside
`DDPGAgent.update` before the critic optimizer step. It does not run online TD
evaluation over every future candidate.

Pair tables, global maximum priority, critic-update count, and beta schedule
progress are included in `checkpoint.pt` for exact resume behavior.

## Output

Full results are isolated from the superseded partial implementation:

```text
experiment_FPP/runs/ddpg_hgr_tdpriority_original_pair_is_1m/
  fetch_pick_and_place_ddpg_hgr_tdpriority_s{seed}_1000000/
```

The stopped legacy partial output under
`experiment_FPP/runs/ddpg_hgr_tdpriority_1m_optimized/` must not be resumed into
this experiment.

## Validation And Execution

```bash
# 1. Probability, IS, beta, overwrite, and checkpoint unit test
.venv/bin/python \
  experiment_FPP/ddpg_hgr_tdpriority_1m/code/src/test_hgr_tdpriority_sampling.py

# 2. Real FetchPickAndPlace wrapper validation
.venv/bin/python experiment_FPP/ddpg_hgr_tdpriority_1m/run.py \
  --seed 1 --steps 5000 --device cpu --validate_only

# 3. Seed-1 5k correctness smoke
bash experiment_FPP/ddpg_hgr_tdpriority_1m/run_fpp_ddpg_hgr_tdpriority_correctness_5k.sh

# 4. Seed-1 50k CUDA runtime smoke
bash experiment_FPP/ddpg_hgr_tdpriority_1m/run_fpp_ddpg_hgr_tdpriority_smoke.sh

# 5. Seeds 1-5, 1M full run
bash experiment_FPP/ddpg_hgr_tdpriority_1m/run_fpp_ddpg_hgr_tdpriority_1m_5parallel.sh
```

The 5k and 50k runs are correctness/runtime checks only. Do not tune alpha or
beta from their performance. A command-only check is available with
`run.py --dry-run`.

## Logged HGR Metrics

The normal summary/config/environment/log/metrics outputs include episode and
pair priority statistics, table count and memory, episode/pair probabilities,
raw and normalized IS weights, actor weight, beta values and progress, priority
safety/clip rates, relabel/fallback rates, and HGR sampling seconds per update.

## Paper Interpretation

If HGR beats SABER: goal prioritization can be stronger in FPP DDPG, while SABER
targets a different critic-loss allocation intervention under matched HER
relabeling.

If SABER matches or beats HGR: SABER remains competitive with a TD-error-based
hindsight goal-prioritization baseline while changing only critic-loss
allocation.

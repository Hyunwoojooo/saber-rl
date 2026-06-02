# FetchPush PER-Full prioritized replay baseline

This folder contains the FetchPush full Prioritized Experience Replay baseline.

## Method

- Replay buffer stores one slot-level priority per transition.
- New transitions are inserted with the current max priority.
- Batch indices are sampled by:

```text
P(i) = priority_i ** per_alpha / sum(priority ** per_alpha)
```

- HER relabeling is then applied to the sampled slots exactly as in the uniform replay path.
- Critic loss uses PER importance-sampling weights:

```text
IS_i = (N * P(i)) ** (-beta)
IS_i = IS_i / max_batch(IS)
Omega_critic_i = IS_i / mean_batch(IS)   # with weighting_mode=batch_norm
Omega_actor_i = 1
```

- After the critic/actor update, sampled slot priorities are updated from the latest TD-error:

```text
priority_i = |TD-error_i| + eps_PER
```

Because FetchPush uses HER, this baseline uses slot-level priorities under relabeled-goal training.

## Main command

From `C:\Users\SS\Projects\metacog-rl\metacog-rl`:

```powershell
cmd /c call version5.0\metacog-lite-v0\experiment_FetchPush\PER-Full-baseline\start_per_full_1m_cuda_parallel2.cmd
```

## Smoke command

```powershell
cmd /c call version5.0\metacog-lite-v0\experiment_FetchPush\PER-Full-baseline\start_per_full_100k_cuda_smoke.cmd
```


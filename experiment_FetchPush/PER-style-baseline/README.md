# FetchPush PER-style loss-weighting baseline

This folder contains the FetchPush PER-style diagnostic baseline.

## Method

- Replay sampling remains uniform, identical to B0/DDPG+HER.
- TD-error is computed on the sampled batch.
- Critic loss weights are computed from batch-local PER-like priorities:

```text
priority_i = (|TD-error_i| + eps_PER) ** per_alpha
w_i = clip(priority_i / mean_batch(priority), 0, U_PER_max)
Omega_critic_i = w_i / mean_batch(w)
Omega_actor_i = 1
```

This is not full Prioritized Experience Replay because replay sampling probabilities are not changed.

## Main command

From `C:\Users\SS\Projects\metacog-rl\metacog-rl`:

```powershell
cmd /c call version5.0\metacog-lite-v0\experiment_FetchPush\PER-style-baseline\start_per_style_1m_cuda_parallel2.cmd
```

## Smoke command

```powershell
cmd /c call version5.0\metacog-lite-v0\experiment_FetchPush\PER-style-baseline\start_per_style_100k_cuda_smoke.cmd
```


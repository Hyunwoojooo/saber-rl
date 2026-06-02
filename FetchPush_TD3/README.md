# FetchPush TD3+HER experiment snapshots

This directory contains TD3 versions of the FetchPush comparison matrix.

## Variants

- `TD3-HER-B0`: TD3 + HER baseline.
- `TD3-HER-B3-Critic-Only`: TD3 + HER + SABER/B3 critic-only local support weighting.
- `TD3-HER-PER-Style`: TD3 + HER + PER-style batch-local TD-error critic loss weighting.
- `TD3-HER-PER-Full`: TD3 + HER + full prioritized replay sampling with IS critic weighting.

All variants keep the FetchPush task, HER relabeling, sparse reward, replay size, batch size, evaluation protocol, and 1M-step training budget aligned with the DDPG FetchPush experiments.

## TD3 defaults

```text
policy_delay = 2
target_policy_noise = 0.2
target_noise_clip = 0.5
```

TD3 uses twin critics. SABER and PER critic weights are applied to both critic losses. Actor loss remains unweighted for the critic-only comparison variants.

## Run

Each folder has:

```powershell
cmd /c call start_1m_cuda_parallel2.cmd
cmd /c call start_100k_cuda_smoke.cmd
```

CUDA parallelism is fixed to 2 by default.

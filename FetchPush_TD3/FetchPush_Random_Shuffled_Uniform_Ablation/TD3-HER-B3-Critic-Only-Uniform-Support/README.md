# TD3-HER-B3-Critic-Only

FetchPush TD3 + HER + SABER/B3 critic-only local support weighting.

This uses the same FetchPush SABER runtime path as the DDPG comparison snapshot:

```text
method = b3_support_only
weight_scope = critic_only
local_search_max = 512
support_lookup_optimization = true
support_scan_mode = sticky_random_v1
support_numba_scan_enabled = true
support_batch_query_coalescing = false
```

Run from this folder:

```powershell
cmd /c call start_1m_cuda_parallel2.cmd
```

Smoke:

```powershell
cmd /c call start_100k_cuda_smoke.cmd
```

# TD3-HER-PER-Full

FetchPush TD3 + HER + full Prioritized Experience Replay.

Replay sampling uses slot-level priorities. HER relabeling is applied after priority sampling. Critic loss uses PER importance-sampling weights, and priorities are updated from the latest TD3 critic TD-error.

Run from this folder:

```powershell
cmd /c call start_1m_cuda_parallel2.cmd
```

Smoke:

```powershell
cmd /c call start_100k_cuda_smoke.cmd
```

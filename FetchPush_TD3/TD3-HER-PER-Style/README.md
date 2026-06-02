# TD3-HER-PER-Style

FetchPush TD3 + HER + PER-style critic loss weighting.

Replay sampling remains uniform. TD-error is computed on the sampled batch, converted to a PER-like weight, and applied to critic loss only.

Run from this folder:

```powershell
cmd /c call start_1m_cuda_parallel2.cmd
```

Smoke:

```powershell
cmd /c call start_100k_cuda_smoke.cmd
```

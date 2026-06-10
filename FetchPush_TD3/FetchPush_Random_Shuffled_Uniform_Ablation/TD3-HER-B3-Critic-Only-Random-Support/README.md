# TD3-HER-B3-Critic-Only-Random-Support

FetchPush TD3 + HER random-support negative-control ablation.

This variant first computes the B3 support mask, then preserves only the number
of high-weight samples and randomly reassigns those high weights within the
batch. In other words, the high-weight density is support-conditioned, but the
sample identity is randomized.

```text
method = b3_support_random
weight_scope = critic_only
local_search_max = 512
support_lookup_optimization = true
support_scan_mode = sticky_random_v1
support_numba_scan_enabled = true
support_batch_query_coalescing = false
```

## Completed Results

Completed 1M-step FetchPush runs used 5 seeds, sparse rewards, HER, CUDA, and
20-episode evaluations every 10k environment steps.

| Method | Final success | Best success | Success AUC | Hit90 step | Speed | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only-Random-Support | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.654 +/- 0.041 | 450k +/- 32k | 34.65 steps/s | 8.02 h |

Seed-level summary:

| Seed | Final success | Best success | AUC | Hit90 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.00 | 1.00 | 0.594 | 490k |
| 2 | 1.00 | 1.00 | 0.658 | 450k |
| 3 | 1.00 | 1.00 | 0.703 | 430k |
| 4 | 1.00 | 1.00 | 0.640 | 470k |
| 5 | 1.00 | 1.00 | 0.675 | 410k |

Interpretation:

- Random-Support is not a pure random baseline: it keeps the B3-derived
  high-weight count and randomizes which batch samples receive those weights.
- The result is strong, suggesting that support-conditioned reweighting density
  and stochastic critic reweighting already provide a useful training signal.
- Aligned B3-Critic-Only remains slightly better than Random-Support on the main
  FetchPush TD3 comparison, but the gap is modest.

Run from this folder:

```powershell
cmd /c call start_1m_cuda_parallel2.cmd
```

Smoke:

```powershell
cmd /c call start_100k_cuda_smoke.cmd
```

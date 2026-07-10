# FPP TD3 HER vs HER+SABER Runtime Profile

- Throughput slowdown: `66.59%`
- Relative training overhead: `199.31%`

| Component | HER ms/env-step | SABER ms/env-step | Delta | SABER training share |
|---|---:|---:|---:|---:|
| Interaction | 1.0002 | 1.0705 | +0.0704 | 4.44% |
| Replay + HER | 1.9252 | 2.6514 | +0.7261 | 10.99% |
| Batch preparation | 0.0961 | 0.1081 | +0.0120 | 0.45% |
| Support and weighting | 0.0000 | 15.0276 | +15.0276 | 62.27% |
| Critic update | 2.9450 | 2.8319 | -0.1131 | 11.73% |
| Actor + target update | 1.3836 | 1.2923 | -0.0913 | 5.35% |
| Eval/logging amortized | 0.0800 | 0.0851 | +0.0051 | 0.35% |
| Total training loop | 8.0630 | 24.1334 | +16.0704 | 100.00% |

This is a hardware-specific implementation profile. It does not establish wall-clock superiority,
and the 100k measurements must not be linearly extrapolated to a 1M-step runtime claim.

# FPP B3 v4 ls512 1M Analysis

Generated: 2026-05-30T00:05:27

## Main Comparison

| method | n | final | best | AUC | tail10 | tail50 | runtime_h | steps/s | lookup_tail50_ms | numba_tail50_ms | scanned_tail50 | psi_tail50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 5 | 0.940 | 1.000 | 0.539 | 0.940 | 0.830 | 4.27 | 65.02 | 0.00 | nan | 0.0 | 0.000 |
| B3_v4_1024 | 5 | 0.920 | 1.000 | 0.592 | 0.960 | 0.902 | 7.03 | 39.51 | 10.54 | 4.32 | 674.3 | 0.441 |
| B3_v4_512 | 5 | 0.960 | 1.000 | 0.609 | 0.969 | 0.918 | 6.66 | 41.72 | 8.78 | 2.95 | 388.4 | 0.339 |

## B3_v4_512 Per Seed

| seed | final | best | AUC | tail10 | tail50 | first_hit_0.90 | runtime_h | steps/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.950 | 1.000 | 0.656 | 0.965 | 0.943 | 480000 | 6.61 | 42.02 |
| 2 | 0.950 | 1.000 | 0.615 | 0.975 | 0.931 | 540000 | 6.69 | 41.52 |
| 3 | 1.000 | 1.000 | 0.565 | 0.970 | 0.860 | 630000 | 6.69 | 41.54 |
| 4 | 1.000 | 1.000 | 0.632 | 0.970 | 0.964 | 480000 | 6.65 | 41.74 |
| 5 | 0.900 | 1.000 | 0.578 | 0.965 | 0.893 | 610000 | 6.65 | 41.76 |

## Runtime/support tail50 for B3_v4_512

- lookup sec/update: 0.0088
- numba support scan sec/update: 0.0030
- actor critic update sec/update: 0.0068
- full context scan rate: 0.6046
- predicate early exit rate: 0.3394
- support cache hit rate: 0.0195
- local scan cache hit rate: 0.9933
- mean scanned candidates: 388.3767
- local support count: 34.8488
- mean w_P: 1.3394
- ESS normalized: 0.8896
- local scan truncated rate: 0.9566

## Phase support tail50 for B3_v4_512

- approach: 0.3424
- grasp: 0.4880
- lift: 0.3590
- place: 0.2424

## Interpretation

- ls512 improved final success and AUC against the prior B3 v4 1024 run while reducing runtime.
- Compared with B0, ls512 has better final success and AUC, but remains substantially slower.
- Lowering local_search_max to 512 did not damage support enough to hurt FPP performance in this run; support rate and local support counts remain above thresholds.
- The main residual cost is still support lookup plus actor/critic update overhead, although lookup is lower than the 1024 setting.

# FPP B3 v4 ls256 500k Screening

Generated: 2026-06-01

## Method Summary

| method | n | final | best | AUC | tail10 | tail50 | runtime_h | steps/s | lookup_ms | numba_ms | psi_tail50 | place_tail50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B3_v4_ls256_500k | 3 | 0.650 +/- 0.122 | 0.867 | 0.333 | 0.670 | 0.333 | 2.71 | 51.33 | 7.44 | 2.05 | 0.368 | 0.463 |

## Per Seed

| seed | final | best | AUC | tail10 | tail50 | first_hit_0.90 | runtime_h | steps/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.500 | 0.800 | 0.321 | 0.585 | 0.320 |  | 2.73 | 50.81 |
| 2 | 0.650 | 0.950 | 0.319 | 0.700 | 0.320 | 490000 | 2.71 | 51.28 |
| 3 | 0.800 | 0.850 | 0.359 | 0.725 | 0.360 |  | 2.68 | 51.90 |

## Runtime/Support Tail50

- lookup ms/update: 7.44
- numba support scan ms/update: 2.05
- scanned candidates: 191.2
- psi_q_rank: 0.368
- mean w_P: 1.368
- anneal scale: nan
- effective lambda_support: nan
- place support: 0.463

## Note

This is the local_search_max=256 screening run without quality gate.

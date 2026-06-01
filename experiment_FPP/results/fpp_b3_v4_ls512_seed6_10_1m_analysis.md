# FPP B3 v4 ls512 Seed 6-10 1M

Generated: 2026-06-01

## Method Summary

| method | n | final | best | AUC | tail10 | tail50 | runtime_h | steps/s | lookup_ms | numba_ms | psi_tail50 | place_tail50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B3_v4_ls512_seed6_10 | 5 | 0.760 +/- 0.381 | 0.860 | 0.477 | 0.781 | 0.726 | 6.72 | 41.37 | 9.22 | 2.77 | 0.361 | 0.300 |

## Per Seed

| seed | final | best | AUC | tail10 | tail50 | first_hit_0.90 | runtime_h | steps/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.950 | 1.000 | 0.515 | 0.975 | 0.797 | 700000 | 6.69 | 41.51 |
| 7 | 0.950 | 1.000 | 0.577 | 0.980 | 0.892 | 600000 | 6.60 | 42.08 |
| 8 | 1.000 | 1.000 | 0.645 | 0.965 | 0.957 | 470000 | 6.60 | 42.11 |
| 9 | 0.000 | 0.300 | 0.071 | 0.035 | 0.084 |  | 7.08 | 39.24 |
| 10 | 0.900 | 1.000 | 0.576 | 0.950 | 0.901 | 570000 | 6.62 | 41.93 |

## Runtime/Support Tail50

- lookup ms/update: 9.22
- numba support scan ms/update: 2.77
- scanned candidates: 379.4
- psi_q_rank: 0.361
- mean w_P: 1.361
- anneal scale: nan
- effective lambda_support: nan
- place support: 0.300

## Note

Additional seeds for the main ls512 v4 setting; seed9 is retained as a real failed/outlier run.

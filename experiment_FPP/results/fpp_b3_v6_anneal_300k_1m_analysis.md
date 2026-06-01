# FPP B3 v6 Anneal 300k 1M

Generated: 2026-06-01

## Method Summary

| method | n | final | best | AUC | tail10 | tail50 | runtime_h | steps/s | lookup_ms | numba_ms | psi_tail50 | place_tail50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B3_v6_anneal_300k | 5 | 0.930 +/- 0.093 | 0.990 | 0.532 | 0.917 | 0.817 | 6.69 | 41.53 | 8.38 | 2.78 | 0.378 | 0.317 |

## Per Seed

| seed | final | best | AUC | tail10 | tail50 | first_hit_0.90 | runtime_h | steps/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.950 | 1.000 | 0.617 | 0.930 | 0.914 | 500000 | 6.60 | 42.12 |
| 2 | 1.000 | 1.000 | 0.583 | 0.960 | 0.904 | 590000 | 6.71 | 41.39 |
| 3 | 1.000 | 1.000 | 0.564 | 0.945 | 0.872 | 610000 | 6.62 | 41.93 |
| 4 | 0.750 | 0.950 | 0.295 | 0.760 | 0.497 | 940000 | 6.91 | 40.20 |
| 5 | 0.950 | 1.000 | 0.602 | 0.990 | 0.896 | 580000 | 6.61 | 42.03 |

## Runtime/Support Tail50

- lookup ms/update: 8.38
- numba support scan ms/update: 2.78
- scanned candidates: 371.4
- psi_q_rank: 0.378
- mean w_P: 1.378
- anneal scale: 1.000
- effective lambda_support: 1.000
- place support: 0.317

## Note

This variant keeps quality gate disabled and ramps support weight from 0 at 300k to full strength at 500k.

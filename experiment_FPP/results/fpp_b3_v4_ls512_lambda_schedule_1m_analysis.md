# FPP B3 v4 ls512 Lambda Schedule 1M

Generated: 2026-06-01

## Method Summary

| method | n | final | best | AUC | tail10 | tail50 | runtime_h | steps/s | lookup_ms | numba_ms | psi_tail50 | place_tail50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B3_v4_ls512_lambda_schedule | 5 | 0.950 +/- 0.077 | 1.000 | 0.611 | 0.948 | 0.918 | 6.67 | 41.65 | 8.69 | 2.84 | 0.340 | 0.245 |

## Per Seed

| seed | final | best | AUC | tail10 | tail50 | first_hit_0.90 | runtime_h | steps/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 0.662 | 0.955 | 0.946 | 470000 | 6.63 | 41.89 |
| 2 | 0.800 | 1.000 | 0.600 | 0.920 | 0.911 | 590000 | 6.71 | 41.42 |
| 3 | 0.950 | 1.000 | 0.565 | 0.960 | 0.865 | 630000 | 6.69 | 41.54 |
| 4 | 1.000 | 1.000 | 0.633 | 0.955 | 0.946 | 500000 | 6.66 | 41.73 |
| 5 | 1.000 | 1.000 | 0.595 | 0.950 | 0.924 | 530000 | 6.66 | 41.69 |

## Runtime/Support Tail50

- lookup ms/update: 8.69
- numba support scan ms/update: 2.84
- scanned candidates: 387.5
- psi_q_rank: 0.340
- mean w_P: 1.170
- anneal scale: nan
- effective lambda_support: nan
- place support: 0.245

## Note

This variant changes only lambda_support over training: 1.0 until 300k, linear decay to 0.5 by 700k, then hold.

# FPP B3 v4 Quality Floor 0.25 ls512 1M Analysis

Generated: 2026-05-30T10:08:36

## Main Comparison

| method | n | final | best | AUC | tail10 | tail50 | hit0.90 | runtime_h | steps/s | lookup_ms | quality_ms | scanned | psi | score | wP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 5 | 0.940 | 1.000 | 0.539 | 0.940 | 0.830 | 642000 | 4.27 | 65.02 | 0.00 | nan | 0.0 | 0.000 | nan | 1.000 |
| B3_v4_1024 | 5 | 0.920 | 1.000 | 0.592 | 0.960 | 0.902 | 576000 | 7.03 | 39.51 | 10.54 | nan | 674.3 | 0.441 | nan | 1.441 |
| B3_v4_512 | 5 | 0.960 | 1.000 | 0.609 | 0.969 | 0.918 | 548000 | 6.66 | 41.72 | 8.78 | nan | 388.4 | 0.339 | nan | 1.339 |
| B3_v4_quality_floor025 | 5 | 0.980 | 1.000 | 0.581 | 0.964 | 0.877 | 566000 | 7.16 | 38.82 | 10.31 | 3.71 | 501.2 | 0.347 | 0.229 | 1.229 |

## Quality Floor 0.25 Per Seed

| seed | final | best | AUC | tail10 | tail50 | hit0.90 | runtime_h | steps/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 0.561 | 0.950 | 0.840 | 630000 | 7.18 | 38.69 |
| 2 | 1.000 | 1.000 | 0.654 | 0.975 | 0.944 | 460000 | 7.14 | 38.92 |
| 3 | 1.000 | 1.000 | 0.567 | 0.950 | 0.864 | 600000 | 7.17 | 38.75 |
| 4 | 1.000 | 1.000 | 0.555 | 0.980 | 0.848 | 550000 | 7.17 | 38.76 |
| 5 | 0.900 | 1.000 | 0.570 | 0.965 | 0.891 | 590000 | 7.13 | 38.98 |

## Delta vs B3 v4 ls512

- final success: 0.9800 vs 0.9600; delta 0.0200
- AUC: 0.5814 vs 0.6091; delta -0.0277
- tail10: 0.9640 vs 0.9690; delta -0.0050
- tail50: 0.8774 vs 0.9182; delta -0.0408
- runtime hours: 7.1556 vs 6.6587; delta 0.4968
- steps/s: 38.8200 vs 41.7170; delta -2.8970
- first hit 0.90: 566000.0000 vs 548000.0000; delta 18000.0000
- lookup tail50 sec: 0.0103 vs 0.0088; delta 0.0015
- quality scan tail50 sec: 0.0037 vs nan; delta nan
- support score tail50: 0.2288 vs nan; delta nan
- mean w_P tail50: 1.2288 vs 1.3394; delta -0.1106

## Quality Diagnostics Tail50

- support_quality_mean: 0.1898
- support_score_mean: 0.2288
- same_bin_match_count: 2.2199
- q_hand: 0.1093
- q_grip: 0.1213
- q_lift: 0.1113
- q_place: 0.2299
- q_near_stable: 0.1321

## Interpretation

- Quality floor 0.25 improved final success to 0.98 mean, with four seeds reaching 1.0 and one seed ending at 0.9.
- AUC dropped against plain B3 v4 ls512, which means early/mid learning was slower despite strong final performance.
- Runtime also regressed versus plain B3 v4 ls512 because quality scan adds exact same-bin progress computation.
- This supports keeping quality gating as a final-performance/stability candidate, but not as the default runtime-efficient setting without further schedule/floor tuning.

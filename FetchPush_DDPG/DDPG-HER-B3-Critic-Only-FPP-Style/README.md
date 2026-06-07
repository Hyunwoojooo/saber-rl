# DDPG-HER-B3-Critic-Only-FPP-Style

FetchPush DDPG + HER with SABER B3 support-only critic loss weighting. The
support thresholds follow the FetchPickAndPlace B3-Critic-Only setting:
M_B=16, M_Q_min=4, N_local_min=4, N_gap_min=128. The executed FetchPush DDPG-B3
run uses `local_search_max=256`.

Run through `FetchPush_DDPG/run_all_ddpg_fetchpush_5parallel.sh` or the queue
script. Results are written under each experiment folder's `results/` directory.

## Reported 1M FetchPush Result

These values are the replacement DDPG-HER-B3 result from the rerun supplied on
2026-06-07. They supersede the earlier local DDPG-B3 summary for this variant.
Raw historical run artifacts remain under `results/`.

| Metric | Mean +/- Std |
| --- | ---: |
| Final Success | 0.990 +/- 0.022 |
| Best Success | 1.000 +/- 0.000 |
| AUC | 0.748 +/- 0.038 |
| Final-window Success | 0.984 +/- 0.015 |
| Hit90 | 344k +/- 50k |
| Speed | 32.04 +/- 0.45 steps/s |
| Runtime | 8.67 +/- 0.12 h |

Per seed:

| Seed | Final Success | Best Success | AUC | Final-window | Hit90 | Runtime | Speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1 | 1.000 | 1.000 | 0.7195 | 0.960 | 360k | 8.61 h | 32.28 |
| s2 | 0.950 | 1.000 | 0.8033 | 0.990 | 270k | 8.61 h | 32.27 |
| s3 | 1.000 | 1.000 | 0.7095 | 1.000 | 400k | 8.85 h | 31.39 |
| s4 | 1.000 | 1.000 | 0.7390 | 0.990 | 370k | 8.74 h | 31.78 |
| s5 | 1.000 | 1.000 | 0.7690 | 0.980 | 320k | 8.55 h | 32.50 |

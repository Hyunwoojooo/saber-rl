# FetchSlide Agent Family Results

Generated: 2026-06-04 KST

## Scope

The main comparison rows use completed FetchSlide 1M-step runs with seeds 1-5. `DDPG+HER+SABER` is the no-quality B3 critic-only runtime-optimized v5 control with `local_search_max=512`; quality-gate, anneal, and positive-loss variants are intentionally excluded from this comparison.

Additional `DDPG+HER+SABER local_search_max=1024` records are included in a separate incomplete-run section. Those runs did not reach 1M steps, so they should not be interpreted as final-performance comparisons.

## Run Mapping

| Label | Run definition | Example path |
|---|---|---|
| DDPG+HER | B0 DDPG+HER | `experiment/FetchSlide/b0/fetch_slide_b0_s1_1000000_lsnone` |
| DDPG+HER+SABER | B3 critic-only runtime opt v5 no-quality, ls512 | `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_no_quality_s1_1000000_ls512_after_quality` |
| DDPG+HER+SABER ls1024, incomplete | Earlier B3 critic-only/runtime-optimization attempts with `local_search_max=1024` | `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_optimization/fetch_slide_b3_critic_only_runtime_optimization_s1_1000000_ls1024` |
| DDPG+HER+PER-Style | PER-style weighting baseline | `experiment_FetchSlide/experiment/FetchSlide/per_style/fetch_slide_per_style_s1_1000000_lsnone` |
| TD3+HER | TD3 baseline | `experiment_FetchSlide/experiment/FetchSlide/td3/fetch_slide_td3_s1_1000000_lsnone` |
| TD3+SABER | TD3 + SABER critic-only, ls512 | `experiment_FetchSlide/experiment/FetchSlide/td3_saber/fetch_slide_td3_saber_s1_1000000_ls512` |
| SAC+HER | SAC baseline | `experiment_FetchSlide/experiment/FetchSlide/sac/fetch_slide_sac_s1_1000000_lsnone` |
| SAC+SABER | SAC + SABER critic-only, ls512 | `experiment_FetchSlide/experiment/FetchSlide/sac_saber/fetch_slide_sac_saber_s1_1000000_ls512` |

## Core Outcome Metrics

| Method | n | Final success | Best success | Success AUC | Last-quarter AUC | Final return | Best return | Drop best-final | Final-window success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DDPG+HER | 5 | 0.600 +/- 0.138 | 0.740 +/- 0.037 | 0.313 +/- 0.012 | 0.519 +/- 0.023 | -38.94 +/- 3.68 | -36.19 +/- 3.26 | 0.140 +/- 0.132 | 0.572 +/- 0.067 |
| DDPG+HER+SABER | 5 | 0.520 +/- 0.093 | 0.740 +/- 0.097 | 0.313 +/- 0.030 | 0.476 +/- 0.060 | -42.06 +/- 1.74 | -37.56 +/- 2.62 | 0.220 +/- 0.121 | 0.508 +/- 0.111 |
| DDPG+HER+PER-Style | 5 | 0.600 +/- 0.114 | 0.750 +/- 0.055 | 0.346 +/- 0.060 | 0.520 +/- 0.083 | -40.40 +/- 3.41 | -36.41 +/- 3.11 | 0.150 +/- 0.100 | 0.574 +/- 0.093 |
| TD3+HER | 5 | 0.630 +/- 0.163 | 0.770 +/- 0.112 | 0.281 +/- 0.059 | 0.594 +/- 0.108 | -37.54 +/- 4.62 | -34.61 +/- 2.26 | 0.140 +/- 0.107 | 0.662 +/- 0.096 |
| TD3+SABER | 5 | 0.690 +/- 0.073 | 0.830 +/- 0.068 | 0.323 +/- 0.039 | 0.627 +/- 0.064 | -36.52 +/- 2.71 | -32.90 +/- 1.04 | 0.140 +/- 0.132 | 0.688 +/- 0.034 |
| SAC+HER | 5 | 0.820 +/- 0.103 | 0.950 +/- 0.045 | 0.500 +/- 0.027 | 0.775 +/- 0.044 | -32.35 +/- 3.63 | -29.39 +/- 2.48 | 0.130 +/- 0.125 | 0.820 +/- 0.055 |
| SAC+SABER | 5 | 0.840 +/- 0.073 | 0.950 +/- 0.032 | 0.535 +/- 0.020 | 0.806 +/- 0.023 | -32.69 +/- 2.58 | -30.01 +/- 3.59 | 0.110 +/- 0.080 | 0.834 +/- 0.039 |

## Runtime Metrics

| Method | Steps/sec | Wall-clock hours/run |
|---|---:|---:|
| DDPG+HER | 119.08 +/- 0.13 | 2.33 +/- 0.00 |
| DDPG+HER+SABER | 26.85 +/- 0.29 | 10.35 +/- 0.11 |
| DDPG+HER+PER-Style | 99.17 +/- 0.08 | 2.80 +/- 0.00 |
| TD3+HER | 111.43 +/- 0.29 | 2.49 +/- 0.01 |
| TD3+SABER | 36.43 +/- 0.26 | 7.62 +/- 0.05 |
| SAC+HER | 66.78 +/- 0.06 | 4.16 +/- 0.00 |
| SAC+SABER | 30.57 +/- 0.17 | 9.09 +/- 0.05 |

## Sample Efficiency

First evaluation step where success rate reaches the threshold. `n/5` means how many seeds reached that threshold.

| Method | >=0.25 | >=0.50 | >=0.75 | >=0.90 |
|---|---:|---:|---:|---:|
| DDPG+HER | 315k +/- 25k (5/5) | 595k +/- 48k (5/5) | 908k +/- 12k (3/5) | NA (0/5) |
| DDPG+HER+SABER | 240k +/- 58k (5/5) | 570k +/- 97k (5/5) | 758k +/- 136k (3/5) | NA (0/5) |
| DDPG+HER+PER-Style | 230k +/- 78k (5/5) | 525k +/- 63k (5/5) | 692k +/- 77k (3/5) | NA (0/5) |
| TD3+HER | 435k +/- 122k (5/5) | 695k +/- 135k (5/5) | 838k +/- 91k (4/5) | NA (0/5) |
| TD3+SABER | 380k +/- 29k (5/5) | 580k +/- 104k (5/5) | 895k +/- 70k (5/5) | 950k +/- 0k (1/5) |
| SAC+HER | 230k +/- 81k (5/5) | 355k +/- 40k (5/5) | 515k +/- 37k (5/5) | 900k +/- 0k (3/5) |
| SAC+SABER | 220k +/- 58k (5/5) | 340k +/- 60k (5/5) | 455k +/- 66k (5/5) | 850k +/- 133k (4/5) |

## Incomplete DDPG+HER+SABER ls1024 Records

These `local_search_max=1024` runs were launched with 1M-step directory names, but the available metrics stop before 1M and no completed-run `summary.json` exists. They are useful for runtime/context, not for final success comparison against the completed 1M rows above.

| Run set | n | Last step | Last eval success | Best success | AUC so far | Steps/sec | Hours/run | psi_q_rank | mean_w_P | lookup time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Runtime opt ls1024 mirror | 5 | 58k +/- 0k | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 14.27 +/- 0.07 | 1.13 +/- 0.00 | 0.188 +/- 0.023 | 1.188 +/- 0.023 | 0.079 +/- 0.002 |
| Runtime opt ls1024 old root | 5 | 144k +/- 1k | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 12.19 +/- 0.06 | 3.28 +/- 0.00 | 0.207 +/- 0.025 | 1.207 +/- 0.025 | 0.075 +/- 0.006 |
| Original B3 critic-only ls1024 | 5 | 480k +/- 3k | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 7.66 +/- 0.04 | 17.41 +/- 0.00 | 0.239 +/- 0.020 | 1.239 +/- 0.020 | 0.144 +/- 0.007 |

### ls1024 Per-Seed Completion Status

| Run set | Seed | Last step | Last eval success | Steps/sec | Hours | psi_q_rank | lookup time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Runtime opt ls1024 mirror | 1 | 58k | 0.000 | 14.32 | 1.13 | 0.172 | 0.082 |
| Runtime opt ls1024 mirror | 2 | 58k | 0.000 | 14.36 | 1.13 | 0.230 | 0.078 |
| Runtime opt ls1024 mirror | 3 | 58k | 0.000 | 14.25 | 1.13 | 0.191 | 0.078 |
| Runtime opt ls1024 mirror | 4 | 57k | 0.000 | 14.14 | 1.13 | 0.168 | 0.079 |
| Runtime opt ls1024 mirror | 5 | 58k | 0.000 | 14.27 | 1.13 | 0.180 | 0.079 |
| Runtime opt ls1024 old root | 1 | 144k | 0.000 | 12.26 | 3.27 | 0.207 | 0.083 |
| Runtime opt ls1024 old root | 2 | 144k | 0.000 | 12.23 | 3.27 | 0.230 | 0.080 |
| Runtime opt ls1024 old root | 3 | 144k | 0.000 | 12.17 | 3.28 | 0.238 | 0.074 |
| Runtime opt ls1024 old root | 4 | 143k | 0.000 | 12.10 | 3.28 | 0.176 | 0.070 |
| Runtime opt ls1024 old root | 5 | 144k | 0.000 | 12.21 | 3.28 | 0.184 | 0.066 |
| Original B3 critic-only ls1024 | 1 | 483k | 0.000 | 7.70 | 17.41 | 0.203 | 0.136 |
| Original B3 critic-only ls1024 | 2 | 483k | 0.000 | 7.71 | 17.41 | 0.234 | 0.158 |
| Original B3 critic-only ls1024 | 3 | 476k | 0.000 | 7.60 | 17.41 | 0.250 | 0.142 |
| Original B3 critic-only ls1024 | 4 | 478k | 0.000 | 7.63 | 17.41 | 0.258 | 0.140 |
| Original B3 critic-only ls1024 | 5 | 479k | 0.000 | 7.65 | 17.40 | 0.250 | 0.145 |

## SABER / PER / SAC Diagnostics

| Method | psi_q_rank | mean_w_P | support_score | lookup time | numba | pred early-exit | PER w_max | PER ESS | PER entropy | SAC alpha | SAC log_prob |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DDPG+HER | 0.000 +/- 0.000 | 1.000 +/- 0.000 | NA | 0.000 +/- 0.000 | NA | NA | NA | NA | NA | NA | NA |
| DDPG+HER+SABER | 0.176 +/- 0.025 | 1.176 +/- 0.025 | 0.176 +/- 0.025 | 0.014 +/- 0.001 | 1.000 +/- 0.000 | 0.177 +/- 0.025 | NA | NA | NA | NA | NA |
| DDPG+HER+PER-Style | 0.000 +/- 0.000 | 0.929 +/- 0.013 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 2.000 +/- 0.000 | 164.459 +/- 8.149 | 5.393 +/- 0.011 | NA | NA |
| TD3+HER | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | NA | NA | NA | NA | NA |
| TD3+SABER | 0.160 +/- 0.013 | 1.160 +/- 0.013 | 0.160 +/- 0.013 | 0.013 +/- 0.003 | 1.000 +/- 0.000 | 0.160 +/- 0.013 | NA | NA | NA | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| SAC+HER | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | NA | NA | NA | 0.010 +/- 0.001 | 3.828 +/- 0.285 |
| SAC+SABER | 0.137 +/- 0.021 | 1.137 +/- 0.021 | 0.137 +/- 0.021 | 0.015 +/- 0.003 | 1.000 +/- 0.000 | 0.137 +/- 0.021 | NA | NA | NA | 0.012 +/- 0.003 | 4.045 +/- 0.334 |

## Per-Seed Results

| Method | Seed | Final success | Best success | AUC | Last-quarter AUC | Final return | Steps/sec | Hours | hit 0.25 | hit 0.50 | hit 0.75 | hit 0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DDPG+HER | 1 | 0.400 | 0.750 | 0.332 | 0.550 | -43.10 | 119.20 | 2.33 | 275k | 525k | 925k | NA |
| DDPG+HER | 2 | 0.550 | 0.700 | 0.301 | 0.505 | -41.95 | 118.89 | 2.34 | 325k | 650k | NA | NA |
| DDPG+HER | 3 | 0.700 | 0.700 | 0.299 | 0.485 | -36.80 | 119.06 | 2.33 | 300k | 625k | NA | NA |
| DDPG+HER | 4 | 0.800 | 0.800 | 0.317 | 0.535 | -32.95 | 119.24 | 2.33 | 325k | 625k | 900k | NA |
| DDPG+HER | 5 | 0.550 | 0.750 | 0.315 | 0.520 | -39.90 | 119.00 | 2.33 | 350k | 550k | 900k | NA |
| DDPG+HER+SABER | 1 | 0.450 | 0.800 | 0.313 | 0.412 | -43.55 | 27.18 | 10.22 | 150k | 550k | 650k | NA |
| DDPG+HER+SABER | 2 | 0.600 | 0.850 | 0.368 | 0.585 | -40.75 | 26.85 | 10.35 | 300k | 575k | 950k | NA |
| DDPG+HER+SABER | 3 | 0.500 | 0.800 | 0.282 | 0.433 | -41.70 | 26.98 | 10.30 | 300k | 675k | 675k | NA |
| DDPG+HER+SABER | 4 | 0.650 | 0.650 | 0.306 | 0.465 | -39.80 | 26.30 | 10.56 | 200k | 400k | NA | NA |
| DDPG+HER+SABER | 5 | 0.400 | 0.600 | 0.294 | 0.485 | -44.50 | 26.92 | 10.32 | 250k | 650k | NA | NA |
| DDPG+HER+PER-Style | 1 | 0.700 | 0.700 | 0.331 | 0.468 | -39.30 | 99.27 | 2.80 | 275k | 525k | NA | NA |
| DDPG+HER+PER-Style | 2 | 0.550 | 0.750 | 0.415 | 0.650 | -43.85 | 99.02 | 2.81 | 275k | 525k | 625k | NA |
| DDPG+HER+PER-Style | 3 | 0.400 | 0.700 | 0.313 | 0.482 | -43.85 | 99.21 | 2.80 | 275k | 525k | NA | NA |
| DDPG+HER+PER-Style | 4 | 0.700 | 0.850 | 0.414 | 0.580 | -34.65 | 99.20 | 2.80 | 75k | 425k | 650k | NA |
| DDPG+HER+PER-Style | 5 | 0.650 | 0.750 | 0.260 | 0.420 | -40.35 | 99.14 | 2.80 | 250k | 625k | 800k | NA |
| TD3+HER | 1 | 0.750 | 0.800 | 0.346 | 0.720 | -34.65 | 111.87 | 2.48 | 350k | 475k | 700k | NA |
| TD3+HER | 2 | 0.400 | 0.550 | 0.181 | 0.400 | -44.75 | 111.27 | 2.50 | 650k | 900k | NA | NA |
| TD3+HER | 3 | 0.500 | 0.800 | 0.256 | 0.588 | -38.35 | 111.16 | 2.50 | 475k | 725k | 875k | NA |
| TD3+HER | 4 | 0.850 | 0.850 | 0.330 | 0.665 | -30.95 | 111.67 | 2.49 | 400k | 675k | 825k | NA |
| TD3+HER | 5 | 0.650 | 0.850 | 0.294 | 0.597 | -39.00 | 111.20 | 2.50 | 300k | 700k | 950k | NA |
| TD3+SABER | 1 | 0.600 | 0.950 | 0.318 | 0.702 | -39.10 | 36.46 | 7.62 | 400k | 650k | 825k | 950k |
| TD3+SABER | 2 | 0.750 | 0.750 | 0.253 | 0.510 | -33.90 | 36.17 | 7.68 | 375k | 650k | 975k | NA |
| TD3+SABER | 3 | 0.650 | 0.850 | 0.352 | 0.652 | -38.55 | 36.65 | 7.58 | 425k | 525k | 925k | NA |
| TD3+SABER | 4 | 0.800 | 0.800 | 0.367 | 0.652 | -32.60 | 36.77 | 7.55 | 350k | 400k | 800k | NA |
| TD3+SABER | 5 | 0.650 | 0.800 | 0.325 | 0.620 | -38.45 | 36.12 | 7.69 | 350k | 675k | 950k | NA |
| SAC+HER | 1 | 0.900 | 0.900 | 0.503 | 0.810 | -26.65 | 66.72 | 4.16 | 275k | 300k | 475k | NA |
| SAC+HER | 2 | 0.750 | 1.000 | 0.496 | 0.780 | -34.20 | 66.80 | 4.16 | 300k | 400k | 575k | 900k |
| SAC+HER | 3 | 0.900 | 0.900 | 0.468 | 0.702 | -33.65 | 66.86 | 4.15 | 275k | 325k | 525k | NA |
| SAC+HER | 4 | 0.650 | 0.950 | 0.487 | 0.757 | -37.15 | 66.70 | 4.16 | 75k | 400k | 475k | 900k |
| SAC+HER | 5 | 0.900 | 1.000 | 0.549 | 0.827 | -30.10 | 66.84 | 4.16 | 225k | 350k | 525k | 900k |
| SAC+SABER | 1 | 0.850 | 1.000 | 0.525 | 0.825 | -31.60 | 30.87 | 9.00 | 275k | 450k | 475k | 950k |
| SAC+SABER | 2 | 0.900 | 0.950 | 0.569 | 0.818 | -32.15 | 30.63 | 9.07 | 250k | 275k | 425k | 625k |
| SAC+SABER | 3 | 0.700 | 0.950 | 0.544 | 0.762 | -37.75 | 30.40 | 9.14 | 150k | 325k | 350k | 875k |
| SAC+SABER | 4 | 0.850 | 0.900 | 0.529 | 0.820 | -30.60 | 30.51 | 9.10 | 275k | 300k | 475k | NA |
| SAC+SABER | 5 | 0.900 | 0.950 | 0.510 | 0.805 | -31.35 | 30.46 | 9.12 | 150k | 350k | 550k | 950k |

## Short Read

- SAC family is the strongest completed set. SAC+SABER has the best mean final success, success AUC, and last-quarter AUC, while SAC+HER is close and faster.
- TD3+SABER improves TD3+HER on mean final success, best success, AUC, and last-quarter AUC, but costs about 3x wall-clock time.
- DDPG+HER+SABER no-quality ls512 does not improve over DDPG+HER in this FetchSlide set; it is slower and has lower final/late success.
- Earlier DDPG+HER+SABER ls1024 attempts did not finish 1M steps. They mainly show why ls512 became the practical main setting: lookup time and wall-clock cost were much higher, especially for the original B3 critic-only path.
- DDPG+HER+PER-Style improves DDPG success AUC over B0 and roughly matches final success, with modest runtime overhead compared with B0.

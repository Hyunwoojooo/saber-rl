# FetchPickAndPlace Main Experiment Results

Generated: 2026-06-04 KST

Scope: completed FetchPickAndPlace runs at 1,000,000 environment steps, seeds 1..5. Metrics are parsed from each run's final evaluation line in `run.log` and supporting diagnostics from local raw `metrics.csv`; the committed run folders include `metrics_selected_10k.csv` and `metrics_final_row.json` instead of full raw metrics.

Important mapping notes:
- DDPG+HER: `B0` (`b0_ddpg_her`). Clean DDPG+HER baseline; support/PER weighting disabled. Run names use ls1024, but local search is inactive for this baseline.
- DDPG+HER+SABER: `b3_critic_only_runtime_opt_v4_ls512` (`b3_support_runtime_opt`). Main DDPG+HER SABER comparison used here: critic-only v4 runtime-optimized support weighting, local_search_max=512. Seeds 1..5 are used for parity; seeds 6..10 also exist and are not included.
- DDPG+HER+PER-style: `per_style_loss_weighting_ls512` (`per_style_loss_weighting`). TD-error priority-derived critic loss weighting with uniform replay sampling; this is not full PER.
- TD3+HER: `td3_her_baseline_ls512` (`td3_her`). TD3 twin critics, delayed policy update, target policy smoothing; no support weighting.
- TD3+HER+SABER: `td3_saber_critic_only_v4_ls512` (`td3_saber_runtime_opt`). TD3 + critic-only SABER support weighting, matched to v4 ls512 settings.
- SAC+HER: `sac_her_baseline_ls512` (`sac`). SAC + HER baseline with uniform critic weighting.
- SAC+HER+SABER: `sac_saber_critic_only_v4_ls512` (`sac_saber`). SAC + HER + critic-only SABER support weighting, v4 exact support lookup.
- DDPG+HER+PER-full: `ddpg_her_per_full_ls512` (`ddpg_her_per_full`). Full prioritized replay sampling with importance-sampling critic weights; beta anneals 0.4 to 1.0.

## Aggregate Outcomes

| Experiment | n | Final success | Best success | Final return | Best return | AUC success | First hit >=0.90 | Failed final seeds | Finish window |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| DDPG+HER | 5 | 0.940 +/- 0.042 | 1.000 +/- 0.000 | -13.76 +/- 3.25 | -9.94 +/- 2.17 | 0.539 +/- 0.125 | 600000 | - | 2026-05-27 20:52:01 - 2026-05-27 20:52:52 |
| DDPG+HER+SABER | 5 | 0.960 +/- 0.042 | 1.000 +/- 0.000 | -10.81 +/- 2.06 | -8.72 +/- 0.70 | 0.609 +/- 0.038 | 540000 | - | 2026-05-29 23:53:51 - 2026-05-29 23:58:38 |
| DDPG+HER+PER-style | 5 | 0.930 +/- 0.104 | 0.990 +/- 0.022 | -13.67 +/- 6.34 | -10.98 +/- 3.20 | 0.568 +/- 0.069 | 630000 | - | 2026-06-02 16:06:59 - 2026-06-02 16:07:56 |
| TD3+HER | 5 | 0.840 +/- 0.188 | 0.880 +/- 0.144 | -18.51 +/- 8.45 | -16.27 +/- 5.44 | 0.372 +/- 0.032 | 980000 | - | 2026-06-03 09:26:56 - 2026-06-03 09:28:05 |
| TD3+HER+SABER | 5 | 0.970 +/- 0.045 | 0.990 +/- 0.022 | -10.85 +/- 2.17 | -10.34 +/- 2.50 | 0.442 +/- 0.035 | 840000 | - | 2026-06-03 04:54:57 - 2026-06-03 05:08:27 |
| SAC+HER | 5 | 0.790 +/- 0.288 | 0.910 +/- 0.124 | -17.23 +/- 11.64 | -13.75 +/- 6.88 | 0.463 +/- 0.077 | 800000 | 5 | 2026-06-03 16:37:34 - 2026-06-03 16:39:22 |
| SAC+HER+SABER | 5 | 0.990 +/- 0.022 | 1.000 +/- 0.000 | -8.80 +/- 0.67 | -8.46 +/- 0.42 | 0.565 +/- 0.030 | 750000 | - | 2026-06-04 02:09:25 - 2026-06-04 02:15:50 |
| DDPG+HER+PER-full | 5 | 0.780 +/- 0.438 | 0.840 +/- 0.358 | -18.94 +/- 17.46 | -16.51 +/- 15.75 | 0.460 +/- 0.238 | 640000 | 5 | 2026-06-04 07:04:25 - 2026-06-04 07:06:16 |

Definitions:
- Final success/return: last evaluation at step 1,000,000.
- Best success: maximum evaluation success within the run; ties break by env success and return.
- Best return: maximum evaluation return within the run.
- AUC success and first-hit thresholds are read from `metrics.csv` when available.
- `+/-` is sample standard deviation across seeds.

## Seed-Level Final Results

| Experiment | s1 | s2 | s3 | s4 | s5 |
|---|---:|---:|---:|---:|---:|
| DDPG+HER | 0.90 / -13.45 | 0.95 / -16.05 | 0.95 / -14.00 | 0.90 / -16.80 | 1.00 / -8.50 |
| DDPG+HER+SABER | 0.95 / -10.65 | 0.95 / -10.15 | 1.00 / -11.60 | 1.00 / -8.00 | 0.90 / -13.65 |
| DDPG+HER+PER-style | 0.75 / -24.75 | 1.00 / -9.10 | 1.00 / -10.55 | 0.95 / -11.10 | 0.95 / -12.85 |
| TD3+HER | 0.55 / -30.10 | 0.95 / -14.45 | 1.00 / -10.60 | 0.95 / -12.70 | 0.75 / -24.70 |
| TD3+HER+SABER | 1.00 / -10.25 | 1.00 / -8.85 | 0.95 / -10.65 | 0.90 / -14.55 | 1.00 / -9.95 |
| SAC+HER | 1.00 / -8.55 | 1.00 / -9.20 | 0.50 / -28.95 | 1.00 / -8.50 | 0.45 / -30.95 |
| SAC+HER+SABER | 0.95 / -9.80 | 1.00 / -8.80 | 1.00 / -8.45 | 1.00 / -8.00 | 1.00 / -8.95 |
| DDPG+HER+PER-full | 1.00 / -11.20 | 1.00 / -11.20 | 0.90 / -13.70 | 1.00 / -8.60 | 0.00 / -50.00 |

Each seed cell is `final success / final return`.

## Final Training And Weight Diagnostics

| Experiment | loss_q | loss_pi | mean w_P | psiQ rate | fallback rate | support available | PER priority ESS | PER IS weight mean | TD/PER weight ESS | runtime steps/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DDPG+HER | 0.711 +/- 0.604 | -8.890 +/- 2.615 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | NA +/- NA | NA +/- NA | NA +/- NA | 65.03 +/- 0.10 |
| DDPG+HER+SABER | 0.443 +/- 0.128 | -10.742 +/- 3.359 | 1.330 +/- 0.052 | 0.330 +/- 0.052 | 0.670 +/- 0.052 | 0.330 +/- 0.052 | NA +/- NA | NA +/- NA | NA +/- NA | 41.72 +/- 0.20 |
| DDPG+HER+PER-style | 1.230 +/- 1.415 | -12.763 +/- 6.795 | 0.950 +/- 0.016 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 179.4 +/- 10.7 | 0.950 +/- 0.016 | 198.6 +/- 5.8 | 61.08 +/- 0.09 |
| TD3+HER | 3.163 +/- 0.625 | 4.344 +/- 1.060 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | NA +/- NA | NA +/- NA | NA +/- NA | 65.33 +/- 0.13 |
| TD3+HER+SABER | 2.395 +/- 1.523 | 1.758 +/- 2.433 | 1.284 +/- 0.051 | 0.284 +/- 0.051 | 0.716 +/- 0.051 | 0.284 +/- 0.051 | NA +/- NA | NA +/- NA | NA +/- NA | 41.25 +/- 0.59 |
| SAC+HER | 0.486 +/- 0.116 | 3.066 +/- 1.403 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | NA +/- NA | NA +/- NA | NA +/- NA | 38.93 +/- 0.07 |
| SAC+HER+SABER | 0.279 +/- 0.112 | 1.369 +/- 0.341 | 1.298 +/- 0.031 | 0.298 +/- 0.031 | 0.702 +/- 0.031 | 0.298 +/- 0.031 | NA +/- NA | NA +/- NA | NA +/- NA | 29.10 +/- 0.12 |
| DDPG+HER+PER-full | 900.937 +/- 2013.999 | -682.010 +/- 1499.337 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 86.7 +/- 33.1 | 0.248 +/- 0.008 | 145.6 +/- 5.9 | 58.02 +/- 0.14 |

Notes:
- Uniform baselines intentionally show `mean w_P` near 1, `psiQ rate` near 0, and `fallback rate` near 1 because SABER lookup is disabled.
- PER-style uses TD-error-derived critic loss weights while replay sampling remains uniform, so PER sampling ESS fields can be `NA`.
- PER-full uses prioritized replay sampling plus normalized importance-sampling weights; `PER priority ESS` and `PER IS weight mean` are the relevant diagnostics.

## Per-Experiment Details

### DDPG+HER

- Result root: `experiment_FPP/runs/B0`
- Method: `b0_ddpg_her`
- Note: Clean DDPG+HER baseline; support/PER weighting disabled. Run names use ls1024, but local search is inactive for this baseline.

| Seed | Final step | Final success | Final env success | Final return | Best success | Best return | AUC success | First hit >=0.90 | Finish time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1000000 | 0.900 | 0.900 | -13.45 | 1.000 | -9.65 | 0.640 | 510000 | 2026-05-27 20:52:24 |
| 2 | 1000000 | 0.950 | 0.950 | -16.05 | 1.000 | -12.60 | 0.537 | 600000 | 2026-05-27 20:52:49 |
| 3 | 1000000 | 0.950 | 0.950 | -14.00 | 1.000 | -8.70 | 0.579 | 610000 | 2026-05-27 20:52:52 |
| 4 | 1000000 | 0.900 | 0.900 | -16.80 | 1.000 | -11.55 | 0.325 | 920000 | 2026-05-27 20:52:11 |
| 5 | 1000000 | 1.000 | 1.000 | -8.50 | 1.000 | -7.20 | 0.612 | 570000 | 2026-05-27 20:52:01 |

### DDPG+HER+SABER

- Result root: `experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512`
- Method: `b3_support_runtime_opt`
- Note: Main DDPG+HER SABER comparison used here: critic-only v4 runtime-optimized support weighting, local_search_max=512. Seeds 1..5 are used for parity; seeds 6..10 also exist and are not included.

| Seed | Final step | Final success | Final env success | Final return | Best success | Best return | AUC success | First hit >=0.90 | Finish time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1000000 | 0.950 | 0.950 | -10.65 | 1.000 | -9.05 | 0.656 | 480000 | 2026-05-29 23:53:51 |
| 2 | 1000000 | 0.950 | 0.950 | -10.15 | 1.000 | -9.15 | 0.615 | 540000 | 2026-05-29 23:58:38 |
| 3 | 1000000 | 1.000 | 1.000 | -11.60 | 1.000 | -9.45 | 0.565 | 630000 | 2026-05-29 23:58:28 |
| 4 | 1000000 | 1.000 | 1.000 | -8.00 | 1.000 | -8.00 | 0.632 | 480000 | 2026-05-29 23:56:32 |
| 5 | 1000000 | 0.900 | 0.900 | -13.65 | 1.000 | -7.95 | 0.578 | 610000 | 2026-05-29 23:56:20 |

### DDPG+HER+PER-style

- Result root: `experiment_FPP/runs/per_style_loss_weighting_ls512`
- Method: `per_style_loss_weighting`
- Note: TD-error priority-derived critic loss weighting with uniform replay sampling; this is not full PER.

| Seed | Final step | Final success | Final env success | Final return | Best success | Best return | AUC success | First hit >=0.90 | Finish time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1000000 | 0.750 | 0.750 | -24.75 | 0.950 | -16.65 | 0.461 | 890000 | 2026-06-02 16:07:56 |
| 2 | 1000000 | 1.000 | 1.000 | -9.10 | 1.000 | -8.85 | 0.596 | 640000 | 2026-06-02 16:07:55 |
| 3 | 1000000 | 1.000 | 1.000 | -10.55 | 1.000 | -10.05 | 0.604 | 530000 | 2026-06-02 16:07:55 |
| 4 | 1000000 | 0.950 | 0.950 | -11.10 | 1.000 | -9.55 | 0.637 | 510000 | 2026-06-02 16:06:59 |
| 5 | 1000000 | 0.950 | 0.950 | -12.85 | 1.000 | -9.80 | 0.544 | 630000 | 2026-06-02 16:07:38 |

### TD3+HER

- Result root: `experiment_FPP/runs/td3_her_baseline_ls512`
- Method: `td3_her`
- Note: TD3 twin critics, delayed policy update, target policy smoothing; no support weighting.

| Seed | Final step | Final success | Final env success | Final return | Best success | Best return | AUC success | First hit >=0.90 | Finish time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1000000 | 0.550 | 0.550 | -30.10 | 0.700 | -19.70 | 0.346 | NA | 2026-06-03 09:27:56 |
| 2 | 1000000 | 0.950 | 0.950 | -14.45 | 0.950 | -14.45 | 0.373 | 1000000 | 2026-06-03 09:27:06 |
| 3 | 1000000 | 1.000 | 1.000 | -10.60 | 1.000 | -10.60 | 0.394 | 980000 | 2026-06-03 09:27:26 |
| 4 | 1000000 | 0.950 | 0.950 | -12.70 | 1.000 | -12.70 | 0.411 | 970000 | 2026-06-03 09:28:05 |
| 5 | 1000000 | 0.750 | 0.750 | -24.70 | 0.750 | -23.90 | 0.334 | NA | 2026-06-03 09:26:56 |

### TD3+HER+SABER

- Result root: `experiment_FPP/runs/td3_saber_critic_only_v4_ls512`
- Method: `td3_saber_runtime_opt`
- Note: TD3 + critic-only SABER support weighting, matched to v4 ls512 settings.

| Seed | Final step | Final success | Final env success | Final return | Best success | Best return | AUC success | First hit >=0.90 | Finish time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1000000 | 1.000 | 1.000 | -10.25 | 1.000 | -10.25 | 0.430 | 840000 | 2026-06-03 05:06:47 |
| 2 | 1000000 | 1.000 | 1.000 | -8.85 | 1.000 | -8.20 | 0.457 | 860000 | 2026-06-03 04:54:57 |
| 3 | 1000000 | 0.950 | 0.950 | -10.65 | 1.000 | -8.75 | 0.496 | 630000 | 2026-06-03 04:59:04 |
| 4 | 1000000 | 0.900 | 0.900 | -14.55 | 0.950 | -14.55 | 0.407 | 630000 | 2026-06-03 05:08:27 |
| 5 | 1000000 | 1.000 | 1.000 | -9.95 | 1.000 | -9.95 | 0.422 | 910000 | 2026-06-03 05:04:29 |

### SAC+HER

- Result root: `experiment_FPP/runs/sac_her_baseline_ls512`
- Method: `sac`
- Note: SAC + HER baseline with uniform critic weighting.

| Seed | Final step | Final success | Final env success | Final return | Best success | Best return | AUC success | First hit >=0.90 | Finish time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1000000 | 1.000 | 1.000 | -8.55 | 1.000 | -8.55 | 0.546 | 775000 | 2026-06-03 16:37:52 |
| 2 | 1000000 | 1.000 | 1.000 | -9.20 | 1.000 | -9.20 | 0.480 | 925000 | 2026-06-03 16:39:22 |
| 3 | 1000000 | 0.500 | 0.500 | -28.95 | 0.750 | -22.05 | 0.396 | NA | 2026-06-03 16:37:34 |
| 4 | 1000000 | 1.000 | 1.000 | -8.50 | 1.000 | -8.50 | 0.522 | 800000 | 2026-06-03 16:38:25 |
| 5 | 1000000 | 0.450 | 0.450 | -30.95 | 0.800 | -20.45 | 0.371 | NA | 2026-06-03 16:39:17 |

### SAC+HER+SABER

- Result root: `experiment_FPP/runs/sac_saber_critic_only_v4_ls512`
- Method: `sac_saber`
- Note: SAC + HER + critic-only SABER support weighting, v4 exact support lookup.

| Seed | Final step | Final success | Final env success | Final return | Best success | Best return | AUC success | First hit >=0.90 | Finish time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1000000 | 0.950 | 0.950 | -9.80 | 1.000 | -8.10 | 0.619 | 650000 | 2026-06-04 02:09:25 |
| 2 | 1000000 | 1.000 | 1.000 | -8.80 | 1.000 | -8.80 | 0.557 | 775000 | 2026-06-04 02:13:16 |
| 3 | 1000000 | 1.000 | 1.000 | -8.45 | 1.000 | -8.45 | 0.553 | 750000 | 2026-06-04 02:15:50 |
| 4 | 1000000 | 1.000 | 1.000 | -8.00 | 1.000 | -8.00 | 0.555 | 750000 | 2026-06-04 02:14:49 |
| 5 | 1000000 | 1.000 | 1.000 | -8.95 | 1.000 | -8.95 | 0.543 | 800000 | 2026-06-04 02:12:44 |

### DDPG+HER+PER-full

- Result root: `experiment_FPP/runs/ddpg_her_per_full_ls512`
- Method: `ddpg_her_per_full`
- Note: Full prioritized replay sampling with importance-sampling critic weights; beta anneals 0.4 to 1.0.

| Seed | Final step | Final success | Final env success | Final return | Best success | Best return | AUC success | First hit >=0.90 | Finish time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1000000 | 1.000 | 1.000 | -11.20 | 1.000 | -10.65 | 0.484 | 770000 | 2026-06-04 07:06:16 |
| 2 | 1000000 | 1.000 | 1.000 | -11.20 | 1.000 | -9.65 | 0.520 | 730000 | 2026-06-04 07:05:06 |
| 3 | 1000000 | 0.900 | 0.900 | -13.70 | 1.000 | -9.30 | 0.604 | 550000 | 2026-06-04 07:05:48 |
| 4 | 1000000 | 1.000 | 1.000 | -8.60 | 1.000 | -8.30 | 0.643 | 460000 | 2026-06-04 07:04:25 |
| 5 | 1000000 | 0.000 | 0.000 | -50.00 | 0.200 | -44.65 | 0.049 | NA | 2026-06-04 07:05:26 |

## Readout

- SAC+HER+SABER is the strongest run set here: mean final success 0.990 with all seeds at or near 1.0.
- TD3+HER+SABER clearly improves over TD3+HER: mean final success 0.970 vs 0.840 and mean final return -10.85 vs -18.51.
- DDPG+HER+SABER improves over DDPG+HER and DDPG+HER+PER-style, but does not match the TD3/SAC families in final return.
- DDPG+HER+PER-full does not explain SABER gains: it finishes at mean final success 0.780 because seed 5 collapses to 0.0 final success.
- PER-style is weaker than clean DDPG+HER on final success in this set, consistent with the earlier observation that loss weighting alone is not the SABER mechanism.

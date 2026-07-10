# SAC Common Evaluation Grid Reanalysis

All metrics in this report are CPU-only reanalysis results. No RL training is run by this script.

The shared evaluation grid is `50k, 100k, ..., 1M` (20 exact points). No interpolation, forward filling, smoothing, or extrapolation is used.

## Artifact Validation

Missing artifacts: none.

| Method | Seed | Run | Metrics | Summary |
|---|---:|---|---|---|
| HER | 1 | `fetch_pick_and_place_sac_her_baseline_ls512_s1_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512/fetch_pick_and_place_sac_her_baseline_ls512_s1_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512/fetch_pick_and_place_sac_her_baseline_ls512_s1_1000000_ls512/summary.json` |
| HER | 2 | `fetch_pick_and_place_sac_her_baseline_ls512_s2_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512/fetch_pick_and_place_sac_her_baseline_ls512_s2_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512/fetch_pick_and_place_sac_her_baseline_ls512_s2_1000000_ls512/summary.json` |
| HER | 3 | `fetch_pick_and_place_sac_her_baseline_ls512_s3_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512/fetch_pick_and_place_sac_her_baseline_ls512_s3_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512/fetch_pick_and_place_sac_her_baseline_ls512_s3_1000000_ls512/summary.json` |
| HER | 4 | `fetch_pick_and_place_sac_her_baseline_ls512_s4_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512/fetch_pick_and_place_sac_her_baseline_ls512_s4_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512/fetch_pick_and_place_sac_her_baseline_ls512_s4_1000000_ls512/summary.json` |
| HER | 5 | `fetch_pick_and_place_sac_her_baseline_ls512_s5_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512/fetch_pick_and_place_sac_her_baseline_ls512_s5_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_baseline_ls512/fetch_pick_and_place_sac_her_baseline_ls512_s5_1000000_ls512/summary.json` |
| HER+SABER | 1 | `fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s1_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_saber_critic_only_v4_ls512/fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s1_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_saber_critic_only_v4_ls512/fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s1_1000000_ls512/summary.json` |
| HER+SABER | 2 | `fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s2_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_saber_critic_only_v4_ls512/fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s2_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_saber_critic_only_v4_ls512/fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s2_1000000_ls512/summary.json` |
| HER+SABER | 3 | `fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s3_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_saber_critic_only_v4_ls512/fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s3_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_saber_critic_only_v4_ls512/fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s3_1000000_ls512/summary.json` |
| HER+SABER | 4 | `fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s4_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_saber_critic_only_v4_ls512/fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s4_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_saber_critic_only_v4_ls512/fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s4_1000000_ls512/summary.json` |
| HER+SABER | 5 | `fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s5_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_saber_critic_only_v4_ls512/fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s5_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_saber_critic_only_v4_ls512/fetch_pick_and_place_sac_saber_critic_only_v4_ls512_s5_1000000_ls512/summary.json` |
| Random-Support | 1 | `fetch_pick_and_place_sac_support_random_critic_only_ls512_s1_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_support_random_critic_only_ls512/fetch_pick_and_place_sac_support_random_critic_only_ls512_s1_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_support_random_critic_only_ls512/fetch_pick_and_place_sac_support_random_critic_only_ls512_s1_1000000_ls512/summary.json` |
| Random-Support | 2 | `fetch_pick_and_place_sac_support_random_critic_only_ls512_s2_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_support_random_critic_only_ls512/fetch_pick_and_place_sac_support_random_critic_only_ls512_s2_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_support_random_critic_only_ls512/fetch_pick_and_place_sac_support_random_critic_only_ls512_s2_1000000_ls512/summary.json` |
| Random-Support | 3 | `fetch_pick_and_place_sac_support_random_critic_only_ls512_s3_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_support_random_critic_only_ls512/fetch_pick_and_place_sac_support_random_critic_only_ls512_s3_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_support_random_critic_only_ls512/fetch_pick_and_place_sac_support_random_critic_only_ls512_s3_1000000_ls512/summary.json` |
| Random-Support | 4 | `fetch_pick_and_place_sac_support_random_critic_only_ls512_s4_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_support_random_critic_only_ls512/fetch_pick_and_place_sac_support_random_critic_only_ls512_s4_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_support_random_critic_only_ls512/fetch_pick_and_place_sac_support_random_critic_only_ls512_s4_1000000_ls512/summary.json` |
| Random-Support | 5 | `fetch_pick_and_place_sac_support_random_critic_only_ls512_s5_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_support_random_critic_only_ls512/fetch_pick_and_place_sac_support_random_critic_only_ls512_s5_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_support_random_critic_only_ls512/fetch_pick_and_place_sac_support_random_critic_only_ls512_s5_1000000_ls512/summary.json` |
| Random-Fixed-Count | 1 | `fetch_pick_and_place_sac_random_fixed_count_k76_critic_only_ls512_s1_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_random_fixed_count_k76_critic_only_ls512/fetch_pick_and_place_sac_random_fixed_count_k76_critic_only_ls512_s1_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_random_fixed_count_k76_critic_only_ls512/fetch_pick_and_place_sac_random_fixed_count_k76_critic_only_ls512_s1_1000000_ls512/summary.json` |
| Random-Fixed-Count | 2 | `fetch_pick_and_place_sac_random_fixed_count_k76_critic_only_ls512_s2_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_random_fixed_count_k76_critic_only_ls512/fetch_pick_and_place_sac_random_fixed_count_k76_critic_only_ls512_s2_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_random_fixed_count_k76_critic_only_ls512/fetch_pick_and_place_sac_random_fixed_count_k76_critic_only_ls512_s2_1000000_ls512/summary.json` |
| Random-Fixed-Count | 3 | `fetch_pick_and_place_sac_random_fixed_count_k76_critic_only_ls512_s3_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_random_fixed_count_k76_critic_only_ls512/fetch_pick_and_place_sac_random_fixed_count_k76_critic_only_ls512_s3_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_random_fixed_count_k76_critic_only_ls512/fetch_pick_and_place_sac_random_fixed_count_k76_critic_only_ls512_s3_1000000_ls512/summary.json` |
| Random-Fixed-Count | 4 | `fetch_pick_and_place_sac_random_fixed_count_k76_critic_only_ls512_s4_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_random_fixed_count_k76_critic_only_ls512/fetch_pick_and_place_sac_random_fixed_count_k76_critic_only_ls512_s4_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_random_fixed_count_k76_critic_only_ls512/fetch_pick_and_place_sac_random_fixed_count_k76_critic_only_ls512_s4_1000000_ls512/summary.json` |
| Random-Fixed-Count | 5 | `fetch_pick_and_place_sac_random_fixed_count_k76_critic_only_ls512_s5_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_random_fixed_count_k76_critic_only_ls512/fetch_pick_and_place_sac_random_fixed_count_k76_critic_only_ls512_s5_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_random_fixed_count_k76_critic_only_ls512/fetch_pick_and_place_sac_random_fixed_count_k76_critic_only_ls512_s5_1000000_ls512/summary.json` |
| HER high-RR x4 | 1 | `fetch_pick_and_place_sac_her_high_rr_x4_s1_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_high_replay_ratio_x4/fetch_pick_and_place_sac_her_high_rr_x4_s1_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_high_replay_ratio_x4/fetch_pick_and_place_sac_her_high_rr_x4_s1_1000000_ls512/summary.json` |
| HER high-RR x4 | 2 | `fetch_pick_and_place_sac_her_high_rr_x4_s2_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_high_replay_ratio_x4/fetch_pick_and_place_sac_her_high_rr_x4_s2_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_high_replay_ratio_x4/fetch_pick_and_place_sac_her_high_rr_x4_s2_1000000_ls512/summary.json` |
| HER high-RR x4 | 3 | `fetch_pick_and_place_sac_her_high_rr_x4_s3_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_high_replay_ratio_x4/fetch_pick_and_place_sac_her_high_rr_x4_s3_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_high_replay_ratio_x4/fetch_pick_and_place_sac_her_high_rr_x4_s3_1000000_ls512/summary.json` |
| HER high-RR x4 | 4 | `fetch_pick_and_place_sac_her_high_rr_x4_s4_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_high_replay_ratio_x4/fetch_pick_and_place_sac_her_high_rr_x4_s4_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_high_replay_ratio_x4/fetch_pick_and_place_sac_her_high_rr_x4_s4_1000000_ls512/summary.json` |
| HER high-RR x4 | 5 | `fetch_pick_and_place_sac_her_high_rr_x4_s5_1000000_ls512` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_high_replay_ratio_x4/fetch_pick_and_place_sac_her_high_rr_x4_s5_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/sac_her_high_replay_ratio_x4/fetch_pick_and_place_sac_her_high_rr_x4_s5_1000000_ls512/summary.json` |
| REDQ-HER TQB | 1 | `fetch_pick_and_place_redq_her_utd4_n10_m2_tqbound_ls512_s1_1000000` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/redq_her_utd4_n10_m2_tqbound_ls512_1m/fetch_pick_and_place_redq_her_utd4_n10_m2_tqbound_ls512_s1_1000000/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/redq_her_utd4_n10_m2_tqbound_ls512_1m/fetch_pick_and_place_redq_her_utd4_n10_m2_tqbound_ls512_s1_1000000/summary.json` |
| REDQ-HER TQB | 2 | `fetch_pick_and_place_redq_her_utd4_n10_m2_tqbound_ls512_s2_1000000` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/redq_her_utd4_n10_m2_tqbound_ls512_1m/fetch_pick_and_place_redq_her_utd4_n10_m2_tqbound_ls512_s2_1000000/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/redq_her_utd4_n10_m2_tqbound_ls512_1m/fetch_pick_and_place_redq_her_utd4_n10_m2_tqbound_ls512_s2_1000000/summary.json` |
| REDQ-HER TQB | 3 | `fetch_pick_and_place_redq_her_utd4_n10_m2_tqbound_ls512_s3_1000000` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/redq_her_utd4_n10_m2_tqbound_ls512_1m/fetch_pick_and_place_redq_her_utd4_n10_m2_tqbound_ls512_s3_1000000/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/redq_her_utd4_n10_m2_tqbound_ls512_1m/fetch_pick_and_place_redq_her_utd4_n10_m2_tqbound_ls512_s3_1000000/summary.json` |
| REDQ-HER TQB | 4 | `fetch_pick_and_place_redq_her_utd4_n10_m2_tqbound_ls512_s4_1000000` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/redq_her_utd4_n10_m2_tqbound_ls512_1m/fetch_pick_and_place_redq_her_utd4_n10_m2_tqbound_ls512_s4_1000000/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/redq_her_utd4_n10_m2_tqbound_ls512_1m/fetch_pick_and_place_redq_her_utd4_n10_m2_tqbound_ls512_s4_1000000/summary.json` |
| REDQ-HER TQB | 5 | `fetch_pick_and_place_redq_her_utd4_n10_m2_tqbound_ls512_s5_1000000` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/redq_her_utd4_n10_m2_tqbound_ls512_1m/fetch_pick_and_place_redq_her_utd4_n10_m2_tqbound_ls512_s5_1000000/metrics_selected_10k.csv` | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/redq_her_utd4_n10_m2_tqbound_ls512_1m/fetch_pick_and_place_redq_her_utd4_n10_m2_tqbound_ls512_s5_1000000/summary.json` |
| SABER diagnostic | 1 | `fetch_pick_and_place_sac_saber_diagnostic_ls512_s1_1000000_ls512` | `experiment_FPP/runs/fpp_sac_saber_diagnostic_ls512_1m/fetch_pick_and_place_sac_saber_diagnostic_ls512_s1_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/runs/fpp_sac_saber_diagnostic_ls512_1m/fetch_pick_and_place_sac_saber_diagnostic_ls512_s1_1000000_ls512/summary.json` |
| SABER diagnostic | 2 | `fetch_pick_and_place_sac_saber_diagnostic_ls512_s2_1000000_ls512` | `experiment_FPP/runs/fpp_sac_saber_diagnostic_ls512_1m/fetch_pick_and_place_sac_saber_diagnostic_ls512_s2_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/runs/fpp_sac_saber_diagnostic_ls512_1m/fetch_pick_and_place_sac_saber_diagnostic_ls512_s2_1000000_ls512/summary.json` |
| SABER diagnostic | 3 | `fetch_pick_and_place_sac_saber_diagnostic_ls512_s3_1000000_ls512` | `experiment_FPP/runs/fpp_sac_saber_diagnostic_ls512_1m/fetch_pick_and_place_sac_saber_diagnostic_ls512_s3_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/runs/fpp_sac_saber_diagnostic_ls512_1m/fetch_pick_and_place_sac_saber_diagnostic_ls512_s3_1000000_ls512/summary.json` |
| SABER diagnostic | 4 | `fetch_pick_and_place_sac_saber_diagnostic_ls512_s4_1000000_ls512` | `experiment_FPP/runs/fpp_sac_saber_diagnostic_ls512_1m/fetch_pick_and_place_sac_saber_diagnostic_ls512_s4_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/runs/fpp_sac_saber_diagnostic_ls512_1m/fetch_pick_and_place_sac_saber_diagnostic_ls512_s4_1000000_ls512/summary.json` |
| SABER diagnostic | 5 | `fetch_pick_and_place_sac_saber_diagnostic_ls512_s5_1000000_ls512` | `experiment_FPP/runs/fpp_sac_saber_diagnostic_ls512_1m/fetch_pick_and_place_sac_saber_diagnostic_ls512_s5_1000000_ls512/metrics_selected_10k.csv` | `experiment_FPP/runs/fpp_sac_saber_diagnostic_ls512_1m/fetch_pick_and_place_sac_saber_diagnostic_ls512_s5_1000000_ls512/summary.json` |

## Native Versus Common-Grid AUC

| Method | Seed | Native AUC | Common-grid AUC | Native - Common | Final success | Hit90 |
|---|---:|---:|---:|---:|---:|---:|
| HER | 1 | 0.546154 | 0.559211 | -0.013057 | 1.000000 | 800000 |
| HER | 2 | 0.480128 | 0.469737 | 0.010391 | 1.000000 | 950000 |
| HER | 3 | 0.395513 | 0.406579 | -0.011066 | 0.500000 |  |
| HER | 4 | 0.522436 | 0.540789 | -0.018354 | 1.000000 | 800000 |
| HER | 5 | 0.370513 | 0.378947 | -0.008435 | 0.450000 |  |
| HER+SABER | 1 | 0.618590 | 0.636842 | -0.018252 | 0.950000 | 650000 |
| HER+SABER | 2 | 0.557051 | 0.567105 | -0.010054 | 1.000000 | 800000 |
| HER+SABER | 3 | 0.553205 | 0.573684 | -0.020479 | 1.000000 | 750000 |
| HER+SABER | 4 | 0.555128 | 0.575000 | -0.019872 | 1.000000 | 750000 |
| HER+SABER | 5 | 0.542949 | 0.564474 | -0.021525 | 1.000000 | 800000 |
| Random-Support | 1 | 0.408838 | 0.435526 | -0.026688 | 0.500000 |  |
| Random-Support | 2 | 0.514646 | 0.536842 | -0.022196 | 0.950000 | 700000 |
| Random-Support | 3 | 0.425758 | 0.439474 | -0.013716 | 0.550000 |  |
| Random-Support | 4 | 0.558081 | 0.575000 | -0.016919 | 1.000000 | 750000 |
| Random-Support | 5 | 0.411616 | 0.423684 | -0.012068 | 0.450000 |  |
| Random-Fixed-Count | 1 | 0.505808 | 0.511842 | -0.006034 | 1.000000 | 900000 |
| Random-Fixed-Count | 2 | 0.553788 | 0.582895 | -0.029107 | 1.000000 | 750000 |
| Random-Fixed-Count | 3 | 0.442172 | 0.451316 | -0.009144 | 1.000000 | 1000000 |
| Random-Fixed-Count | 4 | 0.557071 | 0.588158 | -0.031087 | 1.000000 | 800000 |
| Random-Fixed-Count | 5 | 0.466414 | 0.493421 | -0.027007 | 1.000000 | 900000 |
| HER high-RR x4 | 1 | 0.874747 | 0.906579 | -0.031831 | 0.950000 | 200000 |
| HER high-RR x4 | 2 | 0.619192 | 0.643421 | -0.024229 | 1.000000 | 650000 |
| HER high-RR x4 | 3 | 0.740657 | 0.782895 | -0.042238 | 1.000000 | 450000 |
| HER high-RR x4 | 4 | 0.765404 | 0.788158 | -0.022754 | 0.950000 | 450000 |
| HER high-RR x4 | 5 | 0.695707 | 0.718421 | -0.022714 | 0.900000 | 550000 |
| REDQ-HER TQB | 1 | 0.755808 | 0.780263 | -0.024455 | 1.000000 | 450000 |
| REDQ-HER TQB | 2 | 0.867929 | 0.903947 | -0.036018 | 1.000000 | 250000 |
| REDQ-HER TQB | 3 | 0.883586 | 0.909211 | -0.025625 | 1.000000 | 250000 |
| REDQ-HER TQB | 4 | 0.843434 | 0.877632 | -0.034197 | 1.000000 | 300000 |
| REDQ-HER TQB | 5 | 0.840657 | 0.861842 | -0.021186 | 1.000000 | 300000 |
| SABER diagnostic | 1 | 0.613636 | 0.636842 | -0.023206 | 0.950000 | 650000 |
| SABER diagnostic | 2 | 0.546212 | 0.567105 | -0.020893 | 1.000000 | 800000 |
| SABER diagnostic | 3 | 0.543687 | 0.573684 | -0.029997 | 1.000000 | 750000 |
| SABER diagnostic | 4 | 0.544444 | 0.575000 | -0.030556 | 1.000000 | 750000 |
| SABER diagnostic | 5 | 0.545202 | 0.564474 | -0.019272 | 1.000000 | 800000 |

## Aggregate

Standard deviations are sample standard deviations (`ddof=1`).

| Method | Seeds | Common AUC mean +/- std | Native AUC mean +/- std | Final mean +/- std | Hit90 reached | Median Hit90 |
|---|---|---:|---:|---:|---:|---:|
| HER | 1;2;3;4;5 | 0.471053 +/- 0.079494 | 0.462949 +/- 0.077216 | 0.790000 +/- 0.288097 | 3/5 | 800000 |
| HER+SABER | 1;2;3;4;5 | 0.583421 +/- 0.030186 | 0.565385 +/- 0.030237 | 0.990000 +/- 0.022361 | 5/5 | 750000 |
| Random-Support | 1;2;3;4;5 | 0.482105 +/- 0.068967 | 0.463788 +/- 0.068311 | 0.690000 +/- 0.263154 | 2/5 | 725000 |
| Random-Fixed-Count | 1;2;3;4;5 | 0.525526 +/- 0.059032 | 0.505051 +/- 0.051304 | 1.000000 +/- 0.000000 | 5/5 | 900000 |
| HER high-RR x4 | 1;2;3;4;5 | 0.767895 +/- 0.097224 | 0.739141 +/- 0.094015 | 0.960000 +/- 0.041833 | 5/5 | 450000 |
| REDQ-HER TQB | 1;2;3;4;5 | 0.866579 +/- 0.051983 | 0.838283 +/- 0.049410 | 1.000000 +/- 0.000000 | 5/5 | 300000 |
| SABER diagnostic | 1;2;3;4;5 | 0.583421 +/- 0.030186 | 0.558636 +/- 0.030760 | 0.990000 +/- 0.022361 | 5/5 | 750000 |

Native AUC ranking: REDQ-HER TQB > HER high-RR x4 > HER+SABER > SABER diagnostic > Random-Fixed-Count > Random-Support > HER.

Common-grid AUC ranking: REDQ-HER TQB > HER high-RR x4 > HER+SABER > SABER diagnostic > Random-Fixed-Count > Random-Support > HER.

Method ranking changed after common-grid reanalysis: no.

## Paired AUC Differences

The intervals below are 95% paired percentile bootstrap CIs from 10,000 resamples with RNG seed 27027. Seeds are resampled as paired AUC differences; methods are not resampled independently.

| Comparison | Mean paired difference | 95% CI |
|---|---:|---:|
| SABER - HER | 0.112368 | [0.064211, 0.160526] |
| SABER - Random-Support | 0.101316 | [0.034211, 0.163684] |
| SABER - Random-Fixed-Count | 0.057895 | [0.002632, 0.113158] |
| SABER - high-RR x4 | -0.184474 | [-0.235263, -0.122895] |
| REDQ-HER TQB - SABER | 0.283158 | [0.212632, 0.329474] |

These intervals are reported directly; no broader statistical significance claim is made.

## Reproducibility

Analysis command: `python experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/common_grid_50k/code/analyze_sac_common_grid.py --repo-root . --artifact-root experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m --include-diagnostic --output-dir experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/common_grid_50k/results`

Test command: `python -m unittest experiment_FPP.shared.test_analyze_sac_common_grid`

Generated files: `per_seed_common_grid.csv`, `aggregate_common_grid.csv`, `paired_seed_differences.csv`, `paired_bootstrap_ci.csv`, `grid_validation.json`, and `report.md`.

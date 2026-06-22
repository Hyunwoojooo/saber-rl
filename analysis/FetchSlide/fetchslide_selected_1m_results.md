# FetchSlide Selected 1M Experiment Results

Generated from local `metrics.csv`, `config.json`, and `summary.json` files on 2026-06-01 KST.
Values are reported as seed mean +/- sample standard deviation over 5 seeds unless noted otherwise.
For final-step duplicated rows, the row with nonzero/final `eval/*` values was selected; this avoids the training-log row that carries default zero eval fields.

## Experiments Included
| ID | Experiment | Seeds | Method | local_search_max | quality_mode | quality_apply | anneal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | FetchSlide B0 baseline 1M | 1,2,3,4,5 | b0 | 0 | n/a | n/a | n/a |
| V5-NoQ-LS1024 | FetchSlide B3 critic-only runtime opt v5 no-quality ls1024 1M | 1,2,3,4,5 | b3_critic_only_runtime_opt_v5 | 1024 | n/a | n/a | n/a |
| V5-NoQ-LS512 | FetchSlide B3 critic-only runtime opt v5 no-quality ls512 1M | 1,2,3,4,5 | b3_critic_only_runtime_opt_v5 | 512 | none | False | False |
| V5-NoQ-LS256 | FetchSlide B3 critic-only runtime opt v5 no-quality ls256 1M | 1,2,3,4,5 | b3_critic_only_runtime_opt_v5 | 256 | none | False | False |
| V5-Quality-LS512 | FetchSlide B3 critic-only runtime opt v5 quality apply floor0.25 ls512 1M | 1,2,3,4,5 | b3_critic_only_runtime_opt_v5 | 512 | fetch_slide_phase_soft | True | False |
| V6-Anneal-LS512 | FetchSlide B3 critic-only runtime opt v6 anneal 300k-500k ls512 1M | 1,2,3,4,5 | b3_critic_only_runtime_opt_v6_anneal_300k | 512 | none | False | True |

## Source Directories
### B0: FetchSlide B0 baseline 1M
- `experiment/FetchSlide/b0/fetch_slide_b0_s1_1000000_lsnone`
- `experiment/FetchSlide/b0/fetch_slide_b0_s2_1000000_lsnone`
- `experiment/FetchSlide/b0/fetch_slide_b0_s3_1000000_lsnone`
- `experiment/FetchSlide/b0/fetch_slide_b0_s4_1000000_lsnone`
- `experiment/FetchSlide/b0/fetch_slide_b0_s5_1000000_lsnone`

### V5-NoQ-LS1024: FetchSlide B3 critic-only runtime opt v5 no-quality ls1024 1M
- `experiment_FetchSlide/experiment/FetchSlideRuntimeOptV5Main/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_s1_1000000_ls1024`
- `experiment_FetchSlide/experiment/FetchSlideRuntimeOptV5Main/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_s2_1000000_ls1024`
- `experiment_FetchSlide/experiment/FetchSlideRuntimeOptV5Main/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_s3_1000000_ls1024`
- `experiment_FetchSlide/experiment/FetchSlideRuntimeOptV5Main/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_s4_1000000_ls1024`
- `experiment_FetchSlide/experiment/FetchSlideRuntimeOptV5Main/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_s5_1000000_ls1024`

### V5-NoQ-LS512: FetchSlide B3 critic-only runtime opt v5 no-quality ls512 1M
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_no_quality_s1_1000000_ls512_after_quality`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_no_quality_s2_1000000_ls512_after_quality`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_no_quality_s3_1000000_ls512_after_quality`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_no_quality_s4_1000000_ls512_after_quality`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_no_quality_s5_1000000_ls512_after_quality`

### V5-NoQ-LS256: FetchSlide B3 critic-only runtime opt v5 no-quality ls256 1M
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_no_quality_s1_1000000_ls256`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_no_quality_s2_1000000_ls256`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_no_quality_s3_1000000_ls256`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_no_quality_s4_1000000_ls256`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_no_quality_s5_1000000_ls256`

### V5-Quality-LS512: FetchSlide B3 critic-only runtime opt v5 quality apply floor0.25 ls512 1M
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_quality_apply_floor025_s1_1000000_ls512_verified`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_quality_apply_floor025_s2_1000000_ls512_verified`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_quality_apply_floor025_s3_1000000_ls512_verified`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_quality_apply_floor025_s4_1000000_ls512_verified`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5/fetch_slide_b3_critic_only_runtime_opt_v5_quality_apply_floor025_s5_1000000_ls512_verified`

### V6-Anneal-LS512: FetchSlide B3 critic-only runtime opt v6 anneal 300k-500k ls512 1M
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v6_anneal_300k/fetch_slide_b3_critic_only_runtime_opt_v6_anneal_300k_s1_1000000_ls512`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v6_anneal_300k/fetch_slide_b3_critic_only_runtime_opt_v6_anneal_300k_s2_1000000_ls512`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v6_anneal_300k/fetch_slide_b3_critic_only_runtime_opt_v6_anneal_300k_s3_1000000_ls512`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v6_anneal_300k/fetch_slide_b3_critic_only_runtime_opt_v6_anneal_300k_s4_1000000_ls512`
- `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v6_anneal_300k/fetch_slide_b3_critic_only_runtime_opt_v6_anneal_300k_s5_1000000_ls512`

## Key Summary
| Experiment | eval/success_mean | eval/final_success | eval/best_success | eval/AUC_success | runtime/steps_per_sec | runtime/total_sec | time/local_context_lookup | lite/mean_w_P | lite/psi_q_rank_rate | lite/q_candidate_fallback_rate | lite/mean_num_scanned_candidates | runtime/support_lambda_effective | runtime/support_lookup_enabled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 0.60000 +/- 0.15411 | 0.52000 +/- 0.14405 | 0.74000 +/- 0.04183 | 0.31256 +/- 0.01356 | 119.09 +/- 0.14869 | 8,397.2 +/- 10.486 | 0 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 | n/a | n/a |
| V5-NoQ-LS1024 | 0.59000 +/- 0.11402 | 0.53000 +/- 0.11511 | 0.73000 +/- 0.04472 | 0.30103 +/- 0.02824 | 22.901 +/- 0.19975 | 43,668.2 +/- 380.84 | 0.01807 +/- 0.00214 | 1.2461 +/- 0.02376 | 0.24609 +/- 0.02376 | 0.75391 +/- 0.02376 | 1,018.7 +/- 1.4755 | n/a | n/a |
| V5-NoQ-LS512 | 0.52000 +/- 0.10368 | 0.44000 +/- 0.09618 | 0.74000 +/- 0.10840 | 0.31256 +/- 0.03319 | 26.847 +/- 0.32895 | 37,252.4 +/- 461.25 | 0.01387 +/- 0.00103 | 1.1758 +/- 0.02748 | 0.17578 +/- 0.02748 | 0.82422 +/- 0.02748 | 510.58 +/- 1.3207 | 1.0000 +/- 0 | 1.0000 +/- 0 |
| V5-NoQ-LS256 | 0.44000 +/- 0.12942 | 0.43000 +/- 0.13038 | 0.74000 +/- 0.06519 | 0.32769 +/- 0.02881 | 29.452 +/- 0.11270 | 33,953.8 +/- 129.54 | 0.01000 +/- 0.00126 | 1.1023 +/- 0.00699 | 0.10234 +/- 0.00699 | 0.89766 +/- 0.00699 | 255.48 +/- 0.62675 | 1.0000 +/- 0 | 1.0000 +/- 0 |
| V5-Quality-LS512 | 0.51000 +/- 0.17103 | 0.48000 +/- 0.16808 | 0.69000 +/- 0.09618 | 0.30244 +/- 0.04446 | 35.975 +/- 0.19792 | 27,798.0 +/- 153.08 | 0.01267 +/- 0.00158 | 1.0827 +/- 0.01685 | 0.17266 +/- 0.02928 | 0.82734 +/- 0.02928 | 510.70 +/- 1.4980 | 1.0000 +/- 0 | 1.0000 +/- 0 |
| V6-Anneal-LS512 | 0.53000 +/- 0.14405 | 0.48000 +/- 0.14832 | 0.78000 +/- 0.05701 | 0.31218 +/- 0.03278 | 62.490 +/- 0.24904 | 16,002.8 +/- 63.787 | 0 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |

## Per-Seed Key Results
| Experiment | Seed | final_success | best_success | AUC_success | steps/sec | total_sec | total_h | local_lookup_s | mean_w_P | psi_q_rank | support_lambda | lookup_enabled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 1 | 0.40000 | 0.75000 | 0.33205 | 119.21 | 8,388.5 | 2.3301 | 0 | 1.0000 | 0 | n/a | n/a |
| B0 | 2 | 0.55000 | 0.70000 | 0.30064 | 118.89 | 8,410.9 | 2.3364 | 0 | 1.0000 | 0 | n/a | n/a |
| B0 | 3 | 0.70000 | 0.70000 | 0.29872 | 119.07 | 8,398.5 | 2.3329 | 0 | 1.0000 | 0 | n/a | n/a |
| B0 | 4 | 0.80000 | 0.80000 | 0.31667 | 119.26 | 8,385.3 | 2.3292 | 0 | 1.0000 | 0 | n/a | n/a |
| B0 | 5 | 0.55000 | 0.75000 | 0.31474 | 119.01 | 8,403.0 | 2.3342 | 0 | 1.0000 | 0 | n/a | n/a |
| V5-NoQ-LS1024 | 1 | 0.70000 | 0.70000 | 0.27308 | 22.632 | 44,185.0 | 12.274 | 0.01548 | 1.2188 | 0.21875 | n/a | n/a |
| V5-NoQ-LS1024 | 2 | 0.50000 | 0.80000 | 0.33718 | 23.177 | 43,146.3 | 11.985 | 0.01911 | 1.2461 | 0.24609 | n/a | n/a |
| V5-NoQ-LS1024 | 3 | 0.60000 | 0.75000 | 0.28718 | 22.947 | 43,579.3 | 12.105 | 0.01901 | 1.2266 | 0.22656 | n/a | n/a |
| V5-NoQ-LS1024 | 4 | 0.70000 | 0.70000 | 0.32500 | 22.813 | 43,834.6 | 12.176 | 0.01620 | 1.2734 | 0.27344 | n/a | n/a |
| V5-NoQ-LS1024 | 5 | 0.45000 | 0.70000 | 0.28269 | 22.938 | 43,596.0 | 12.110 | 0.02054 | 1.2656 | 0.26562 | n/a | n/a |
| V5-NoQ-LS512 | 1 | 0.45000 | 0.80000 | 0.31346 | 27.185 | 36,785.0 | 10.218 | 0.01539 | 1.1953 | 0.19531 | 1.0000 | 1.0000 |
| V5-NoQ-LS512 | 2 | 0.60000 | 0.85000 | 0.36795 | 26.847 | 37,247.6 | 10.347 | 0.01251 | 1.2070 | 0.20703 | 1.0000 | 1.0000 |
| V5-NoQ-LS512 | 3 | 0.50000 | 0.80000 | 0.28205 | 26.981 | 37,063.3 | 10.295 | 0.01393 | 1.1367 | 0.13672 | 1.0000 | 1.0000 |
| V5-NoQ-LS512 | 4 | 0.65000 | 0.65000 | 0.30577 | 26.303 | 38,017.9 | 10.561 | 0.01362 | 1.1758 | 0.17578 | 1.0000 | 1.0000 |
| V5-NoQ-LS512 | 5 | 0.40000 | 0.60000 | 0.29359 | 26.919 | 37,148.4 | 10.319 | 0.01390 | 1.1641 | 0.16406 | 1.0000 | 1.0000 |
| V5-NoQ-LS256 | 1 | 0.50000 | 0.70000 | 0.35641 | 29.363 | 34,056.8 | 9.4602 | 0.01115 | 1.1055 | 0.10547 | 1.0000 | 1.0000 |
| V5-NoQ-LS256 | 2 | 0.40000 | 0.80000 | 0.33590 | 29.458 | 33,946.5 | 9.4296 | 0.00915 | 1.1055 | 0.10547 | 1.0000 | 1.0000 |
| V5-NoQ-LS256 | 3 | 0.25000 | 0.65000 | 0.34038 | 29.354 | 34,066.4 | 9.4629 | 0.01041 | 1.0898 | 0.08984 | 1.0000 | 1.0000 |
| V5-NoQ-LS256 | 4 | 0.60000 | 0.75000 | 0.32564 | 29.634 | 33,744.6 | 9.3735 | 0.00824 | 1.1055 | 0.10547 | 1.0000 | 1.0000 |
| V5-NoQ-LS256 | 5 | 0.45000 | 0.80000 | 0.28013 | 29.451 | 33,954.7 | 9.4319 | 0.01104 | 1.1055 | 0.10547 | 1.0000 | 1.0000 |
| V5-Quality-LS512 | 1 | 0.70000 | 0.80000 | 0.34615 | 36.057 | 27,734.2 | 7.7039 | 0.01061 | 1.0945 | 0.18750 | 1.0000 | 1.0000 |
| V5-Quality-LS512 | 2 | 0.25000 | 0.70000 | 0.32756 | 36.210 | 27,616.5 | 7.6713 | 0.01347 | 1.0984 | 0.20312 | 1.0000 | 1.0000 |
| V5-Quality-LS512 | 3 | 0.60000 | 0.65000 | 0.29615 | 35.725 | 27,991.8 | 7.7755 | 0.01175 | 1.0554 | 0.12500 | 1.0000 | 1.0000 |
| V5-Quality-LS512 | 4 | 0.45000 | 0.75000 | 0.31218 | 35.819 | 27,918.2 | 7.7550 | 0.01277 | 1.0811 | 0.17188 | 1.0000 | 1.0000 |
| V5-Quality-LS512 | 5 | 0.55000 | 0.55000 | 0.23013 | 36.063 | 27,729.2 | 7.7026 | 0.01474 | 1.0842 | 0.17578 | 1.0000 | 1.0000 |
| V6-Anneal-LS512 | 1 | 0.50000 | 0.80000 | 0.33205 | 62.211 | 16,074.5 | 4.4651 | 0 | 1.0000 | 0 | 0 | 0 |
| V6-Anneal-LS512 | 2 | 0.35000 | 0.80000 | 0.34167 | 62.793 | 15,925.4 | 4.4237 | 0 | 1.0000 | 0 | 0 | 0 |
| V6-Anneal-LS512 | 3 | 0.50000 | 0.70000 | 0.26795 | 62.640 | 15,964.2 | 4.4345 | 0 | 1.0000 | 0 | 0 | 0 |
| V6-Anneal-LS512 | 4 | 0.55000 | 0.85000 | 0.33269 | 62.546 | 15,988.2 | 4.4412 | 0 | 1.0000 | 0 | 0 | 0 |
| V6-Anneal-LS512 | 5 | 0.75000 | 0.75000 | 0.28654 | 62.261 | 16,061.5 | 4.4615 | 0 | 1.0000 | 0 | 0 | 0 |

## All Numeric Metrics
Each table reports final-step seed mean +/- sample standard deviation. `n/a` means the metric was not present for that experiment or was nonnumeric.

### Core / Outcome Metrics
| Metric | B0 | V5-NoQ-LS1024 | V5-NoQ-LS512 | V5-NoQ-LS256 | V5-Quality-LS512 | V6-Anneal-LS512 |
| --- | --- | --- | --- | --- | --- | --- |
| step | 1,000,000 +/- 0 | 1,000,000 +/- 0 | 1,000,000 +/- 0 | 1,000,000 +/- 0 | 1,000,000 +/- 0 | 1,000,000 +/- 0 |
| episode | 20,000.0 +/- 0 | 20,000.0 +/- 0 | 20,000.0 +/- 0 | 20,000.0 +/- 0 | 20,000.0 +/- 0 | 20,000.0 +/- 0 |
| train/episode_return | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| train/episode_success | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| train/episode_len | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| train/replay_size | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| train/her_relabel_ratio | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| loss/critic | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| loss/actor | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| critic/q_policy_mean | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| critic/q_target_mean | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |

### Weight / Support Metrics
| Metric | B0 | V5-NoQ-LS1024 | V5-NoQ-LS512 | V5-NoQ-LS256 | V5-Quality-LS512 | V6-Anneal-LS512 |
| --- | --- | --- | --- | --- | --- | --- |
| lite/mean_w_P | 1.0000 +/- 0 | 1.2461 +/- 0.02376 | 1.1758 +/- 0.02748 | 1.1023 +/- 0.00699 | 1.0827 +/- 0.01685 | 1.0000 +/- 0 |
| lite/max_w_P | 1.0000 +/- 0 | 2.0000 +/- 0 | 2.0000 +/- 0 | 2.0000 +/- 0 | 2.0000 +/- 0 | 1.0000 +/- 0 |
| lite/std_w_P | 0 +/- 0 | 0.43003 +/- 0.01405 | 0.37923 +/- 0.02410 | 0.30292 +/- 0.00948 | 0.21558 +/- 0.02453 | 0 +/- 0 |
| lite/entropy_w_P | 5.5452 +/- 0 | 5.4916 +/- 0.00215 | 5.5002 +/- 0.00432 | 5.5139 +/- 0.00167 | 5.5282 +/- 0.00328 | 5.5452 +/- 0 |
| lite/ESS_w | 256.00 +/- 0 | 228.76 +/- 0.66432 | 231.89 +/- 1.7642 | 238.02 +/- 0.83090 | 246.20 +/- 1.8064 | 256.00 +/- 0 |
| lite/ESS_w_normalized | 1.0000 +/- 0 | 0.89361 +/- 0.00259 | 0.90583 +/- 0.00689 | 0.92978 +/- 0.00325 | 0.96170 +/- 0.00706 | 1.0000 +/- 0 |
| lite/support_available_rate | 0 +/- 0 | 0.24609 +/- 0.02376 | 0.17578 +/- 0.02748 | 0.10234 +/- 0.00699 | 0.17266 +/- 0.02928 | 0 +/- 0 |
| lite/psi_q_rank_rate | 0 +/- 0 | 0.24609 +/- 0.02376 | 0.17578 +/- 0.02748 | 0.10234 +/- 0.00699 | 0.17266 +/- 0.02928 | 0 +/- 0 |
| lite/q_candidate_fallback_rate | 1.0000 +/- 0 | 0.75391 +/- 0.02376 | 0.82422 +/- 0.02748 | 0.89766 +/- 0.00699 | 0.82734 +/- 0.02928 | 1.0000 +/- 0 |
| lite/local_scan_truncated_rate | 0 +/- 0 | 0.98438 +/- 0.00276 | 0.99375 +/- 0.00445 | 0.99375 +/- 0.00524 | 0.99453 +/- 0.00445 | 0 +/- 0 |
| lite/local_support_count_mean | 0 +/- 0 | 25.654 +/- 3.6142 | 14.067 +/- 1.3575 | 8.7133 +/- 0.53387 | 20.027 +/- 2.9326 | 0 +/- 0 |
| lite/local_support_count_p50 | 0 +/- 0 | 9.8000 +/- 1.3038 | 5.2000 +/- 0.83666 | 3.6000 +/- 0.54772 | 6.8000 +/- 2.1679 | 0 +/- 0 |
| lite/local_support_count_p90 | 0 +/- 0 | 72.800 +/- 13.447 | 41.800 +/- 5.2512 | 25.100 +/- 1.2450 | 56.800 +/- 5.6635 | 0 +/- 0 |
| lite/mean_len_A_Q_replay_local | 0 +/- 0 | 25.654 +/- 3.6142 | 14.067 +/- 1.3575 | 8.7133 +/- 0.53387 | 20.027 +/- 2.9326 | 0 +/- 0 |
| lite/mean_len_A_Q_sup | 0 +/- 0 | 9.2891 +/- 0.56630 | 7.0352 +/- 0.51191 | 5.4461 +/- 0.48154 | 7.9133 +/- 0.85619 | 0 +/- 0 |
| lite/mean_num_coarse_candidates | 0 +/- 0 | 17,851.2 +/- 1,123.0 | 17,998.8 +/- 1,206.6 | 17,947.9 +/- 1,228.6 | 18,008.8 +/- 1,806.1 | 0 +/- 0 |
| lite/mean_num_scanned_candidates | 0 +/- 0 | 1,018.7 +/- 1.4755 | 510.58 +/- 1.3207 | 255.48 +/- 0.62675 | 510.70 +/- 1.4980 | 0 +/- 0 |
| lite/support_quality_mean | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.05747 +/- 0.01074 | 0 +/- 0 |
| lite/support_quality_p50 | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| lite/support_quality_p90 | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.07141 +/- 0.07126 | 0 +/- 0 |
| lite/support_quality_positive_rate | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.12344 +/- 0.02600 | 0 +/- 0 |
| lite/support_score_mean | n/a | n/a | 0.17578 +/- 0.02748 | 0.10234 +/- 0.00699 | 0.08271 +/- 0.01685 | 0 +/- 0 |
| lite/support_score_p50 | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| lite/support_score_p90 | n/a | n/a | 1.0000 +/- 0 | 0.80000 +/- 0.44721 | 0.29515 +/- 0.04536 | 0 +/- 0 |
| lite/same_bin_match_count_mean | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.45234 +/- 0.12468 | 0 +/- 0 |

### Timing Metrics
| Metric | B0 | V5-NoQ-LS1024 | V5-NoQ-LS512 | V5-NoQ-LS256 | V5-Quality-LS512 | V6-Anneal-LS512 |
| --- | --- | --- | --- | --- | --- | --- |
| time/local_context_lookup | 0 +/- 0 | 0.01807 +/- 0.00214 | 0.01387 +/- 0.00103 | 0.01000 +/- 0.00126 | 0.01267 +/- 0.00158 | 0 +/- 0 |
| time/candidate_construction | 0 +/- 0 | 4.957e-04 +/- 4.405e-05 | 5.062e-04 +/- 3.214e-05 | 4.037e-04 +/- 4.684e-05 | 6.580e-04 +/- 6.478e-05 | 0 +/- 0 |
| time/weight_compute | 0 +/- 0 | 9.976e-06 +/- 1.949e-06 | 7.379e-06 +/- 1.136e-06 | 8.078e-06 +/- 9.644e-07 | 7.166e-06 +/- 1.275e-06 | 0 +/- 0 |
| time/actor_critic_update | 0.00287 +/- 8.628e-04 | 0.00263 +/- 3.709e-04 | 0.00264 +/- 5.064e-04 | 0.00274 +/- 4.764e-04 | 0.00279 +/- 0.00108 | 0.00256 +/- 7.205e-04 |
| time/q_candidate_eval | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| time/q_policy_eval | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| time/numba_support_scan | n/a | 0.01205 +/- 0.00139 | 0.00633 +/- 6.638e-04 | 0.00339 +/- 2.217e-04 | 0.00667 +/- 0.00119 | 0 +/- 0 |
| time/support_quality_scan | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.00667 +/- 0.00119 | 0 +/- 0 |

### Runtime Optimization Metrics
| Metric | B0 | V5-NoQ-LS1024 | V5-NoQ-LS512 | V5-NoQ-LS256 | V5-Quality-LS512 | V6-Anneal-LS512 |
| --- | --- | --- | --- | --- | --- | --- |
| runtime/steps_per_sec | 119.09 +/- 0.14869 | 22.901 +/- 0.19975 | 26.847 +/- 0.32895 | 29.452 +/- 0.11270 | 35.975 +/- 0.19792 | 62.490 +/- 0.24904 |
| runtime/total_sec | 8,397.2 +/- 10.486 | 43,668.2 +/- 380.84 | 37,252.4 +/- 461.25 | 33,953.8 +/- 129.54 | 27,798.0 +/- 153.08 | 16,002.8 +/- 63.787 |
| runtime/support_cache_hit_rate | n/a | 0.00391 +/- 0.00391 | 0.00625 +/- 0.00654 | 7.813e-04 +/- 0.00175 | 0.00547 +/- 0.00654 | 0 +/- 0 |
| runtime/support_cache_miss_rate | n/a | 0.99609 +/- 0.00391 | 0.99375 +/- 0.00654 | 0.99922 +/- 0.00175 | 0.99453 +/- 0.00654 | 0 +/- 0 |
| runtime/batch_query_unique_rate | n/a | 1.0000 +/- 0 | 1.0000 +/- 0 | 1.0000 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 |
| runtime/batch_query_coalescing_saving_rate | n/a | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| runtime/local_scan_cache_hit_rate | n/a | 0.99294 +/- 0.00174 | 0.99214 +/- 5.182e-05 | 0.99375 +/- 0.00214 | 0.99293 +/- 0.00177 | 0 +/- 0 |
| runtime/local_scan_cache_miss_rate | n/a | 0.00706 +/- 0.00174 | 0.00786 +/- 5.182e-05 | 0.00625 +/- 0.00214 | 0.00707 +/- 0.00177 | 0 +/- 0 |
| runtime/numba_support_scan_enabled | n/a | 1.0000 +/- 0 | 1.0000 +/- 0 | 1.0000 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 |
| runtime/numba_support_scan_fallback_rate | n/a | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| runtime/numba_support_queries | n/a | 255.00 +/- 1.0000 | 254.40 +/- 1.6733 | 255.80 +/- 0.44721 | 254.60 +/- 1.6733 | 0 +/- 0 |
| runtime/numba_scanner_warmed | n/a | 1.0000 +/- 0 | 1.0000 +/- 0 | 1.0000 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 |
| runtime/predicate_early_exit_enabled | n/a | 1.0000 +/- 0 | 1.0000 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| runtime/predicate_early_exit_rate | n/a | 0.24705 +/- 0.02372 | 0.17689 +/- 0.02768 | 0.10243 +/- 0.00704 | 0 +/- 0 | 0 +/- 0 |
| runtime/predicate_positive_early_exit_rate | n/a | 0.24705 +/- 0.02372 | 0.17689 +/- 0.02768 | 0.10243 +/- 0.00704 | 0 +/- 0 | 0 +/- 0 |
| runtime/predicate_negative_early_exit_rate | n/a | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| runtime/predicate_candidates_examined_mean | n/a | 874.88 +/- 18.645 | 459.08 +/- 10.615 | 241.90 +/- 2.5529 | 513.54 +/- 4.6073 | 0 +/- 0 |
| runtime/predicate_scan_fraction_mean | n/a | 0.85547 +/- 0.01887 | 0.89352 +/- 0.02187 | 0.94610 +/- 0.00944 | 1.0000 +/- 0 | 0 +/- 0 |
| runtime/support_diagnostics_exact_rate | n/a | 0.75391 +/- 0.02376 | 0.82422 +/- 0.02748 | 0.89766 +/- 0.00699 | 1.0000 +/- 0 | 0 +/- 0 |
| runtime/support_diagnostics_approx_rate | n/a | 0.24609 +/- 0.02376 | 0.17578 +/- 0.02748 | 0.10234 +/- 0.00699 | 0 +/- 0 | 0 +/- 0 |
| runtime/predicate_audit_interval | n/a | 100.00 +/- 0 | 100.00 +/- 0 | 100.00 +/- 0 | 100.00 +/- 0 | 0 +/- 0 |
| runtime/predicate_audit_ran | n/a | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| runtime/predicate_audit_has_run | n/a | 1.0000 +/- 0 | 1.0000 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| runtime/predicate_audit_age_updates | n/a | 1.0000 +/- 0 | 1.0000 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| runtime/support_lambda_effective | n/a | n/a | 1.0000 +/- 0 | 1.0000 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 |
| runtime/support_lookup_enabled | n/a | n/a | 1.0000 +/- 0 | 1.0000 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 |
| runtime/support_lookup_skipped | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 1.0000 +/- 0 |
| runtime/support_quality_enabled | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 |
| runtime/support_quality_apply_to_weight | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 |
| runtime/support_quality_numba_enabled | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 |
| runtime/support_quality_numba_fallback_rate | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| runtime/support_anneal_start_step | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 300,000 +/- 0 |
| runtime/support_anneal_end_step | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 500,000 +/- 0 |
| runtime/support_anneal_progress | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 1.0000 +/- 0 |

### Audit Metrics
| Metric | B0 | V5-NoQ-LS1024 | V5-NoQ-LS512 | V5-NoQ-LS256 | V5-Quality-LS512 | V6-Anneal-LS512 |
| --- | --- | --- | --- | --- | --- | --- |
| audit/support_semantic_match_rate | n/a | 1.0000 +/- 0 | 1.0000 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| audit/local_count_exact_match_rate | n/a | 0.74922 +/- 0.02967 | 0.82812 +/- 0.02620 | 0.87344 +/- 0.02321 | 0 +/- 0 | 0 +/- 0 |
| audit/local_count_abs_delta_mean | n/a | 16.697 +/- 2.9485 | 5.9016 +/- 0.64323 | 2.2609 +/- 0.60321 | 0 +/- 0 | 0 +/- 0 |
| audit/psi_q_rank_rate | n/a | 0.25938 +/- 0.03083 | 0.18359 +/- 0.02576 | 0.13984 +/- 0.02522 | 0 +/- 0 | 0 +/- 0 |
| audit/psi_A_policy_rate | n/a | 0.25938 +/- 0.03083 | 0.18359 +/- 0.02576 | 0.13984 +/- 0.02522 | 0 +/- 0 | 0 +/- 0 |
| audit/q_candidate_fallback_rate | n/a | 0.74062 +/- 0.03083 | 0.81641 +/- 0.02576 | 0.86016 +/- 0.02522 | 0 +/- 0 | 0 +/- 0 |
| audit/mean_len_A_Q_replay_local | n/a | 42.551 +/- 3.1807 | 20.950 +/- 1.7335 | 10.693 +/- 0.58434 | 0 +/- 0 | 0 +/- 0 |
| audit/mean_len_A_Q_sup | n/a | 10.503 +/- 0.52444 | 8.0461 +/- 0.58927 | 6.4688 +/- 0.22195 | 0 +/- 0 | 0 +/- 0 |
| audit/mean_num_scanned_candidates | n/a | 1,017.8 +/- 3.7983 | 510.96 +/- 0.79868 | 255.90 +/- 0.16134 | 0 +/- 0 | 0 +/- 0 |
| audit/local_scan_truncated_rate | n/a | 0.98672 +/- 0.00761 | 0.99687 +/- 0.00175 | 0.99766 +/- 0.00214 | 0 +/- 0 | 0 +/- 0 |

### Evaluation Metrics
| Metric | B0 | V5-NoQ-LS1024 | V5-NoQ-LS512 | V5-NoQ-LS256 | V5-Quality-LS512 | V6-Anneal-LS512 |
| --- | --- | --- | --- | --- | --- | --- |
| eval/success_mean | 0.60000 +/- 0.15411 | 0.59000 +/- 0.11402 | 0.52000 +/- 0.10368 | 0.44000 +/- 0.12942 | 0.51000 +/- 0.17103 | 0.53000 +/- 0.14405 |
| eval/goal_success_mean | 0.60000 +/- 0.15411 | 0.59000 +/- 0.11402 | 0.52000 +/- 0.10368 | 0.44000 +/- 0.12942 | 0.51000 +/- 0.17103 | 0.53000 +/- 0.14405 |
| eval/env_success_mean | 0.60000 +/- 0.15411 | 0.59000 +/- 0.11402 | 0.52000 +/- 0.10368 | 0.44000 +/- 0.12942 | 0.51000 +/- 0.17103 | 0.53000 +/- 0.14405 |
| eval/return_mean | -38.940 +/- 4.1154 | -40.730 +/- 2.6355 | -42.060 +/- 1.9447 | -41.270 +/- 3.6032 | -42.030 +/- 3.8319 | -41.280 +/- 2.5929 |
| eval/final_success | 0.52000 +/- 0.14405 | 0.53000 +/- 0.11511 | 0.44000 +/- 0.09618 | 0.43000 +/- 0.13038 | 0.48000 +/- 0.16808 | 0.48000 +/- 0.14832 |
| eval/best_success | 0.74000 +/- 0.04183 | 0.73000 +/- 0.04472 | 0.74000 +/- 0.10840 | 0.74000 +/- 0.06519 | 0.69000 +/- 0.09618 | 0.78000 +/- 0.05701 |
| eval/AUC_success | 0.31256 +/- 0.01356 | 0.30103 +/- 0.02824 | 0.31256 +/- 0.03319 | 0.32769 +/- 0.02881 | 0.30244 +/- 0.04446 | 0.31218 +/- 0.03278 |
| eval/first_hit_0.25 | 315,000 +/- 28,504.4 | 285,000 +/- 37,914.4 | 240,000 +/- 65,192.0 | 260,000 +/- 89,442.7 | 310,000 +/- 45,414.8 | 240,000 +/- 65,192.0 |
| eval/first_hit_0.50 | 595,000 +/- 54,198.7 | 565,000 +/- 62,749.5 | 570,000 +/- 108,109 | 480,000 +/- 116,458 | 535,000 +/- 57,554.3 | 515,000 +/- 76,239.8 |
| eval/first_hit_0.75 | 908,333 +/- 14,433.8 | 787,500 +/- 17,677.7 | 758,333 +/- 166,458 | 850,000 +/- 90,138.8 | 737,500 +/- 88,388.3 | 868,750 +/- 167,550 |

### FetchSlide Task Metrics
| Metric | B0 | V5-NoQ-LS1024 | V5-NoQ-LS512 | V5-NoQ-LS256 | V5-Quality-LS512 | V6-Anneal-LS512 |
| --- | --- | --- | --- | --- | --- | --- |
| fetch_slide/object_goal_dist_mean | 0.18975 +/- 0.03308 | 0.20231 +/- 0.02216 | 0.19275 +/- 0.03128 | 0.19295 +/- 0.03483 | 0.18887 +/- 0.01369 | 0.18021 +/- 0.01187 |
| fetch_slide/object_goal_dist_p50 | 0.12129 +/- 0.02585 | 0.14010 +/- 0.01178 | 0.12850 +/- 0.00749 | 0.12275 +/- 0.02162 | 0.13678 +/- 0.02619 | 0.12271 +/- 0.01173 |
| fetch_slide/object_goal_dist_p90 | 0.44423 +/- 0.11887 | 0.44532 +/- 0.05758 | 0.42857 +/- 0.09203 | 0.42710 +/- 0.05973 | 0.39593 +/- 0.02379 | 0.39047 +/- 0.01540 |
| fetch_slide/object_speed_mean | 0.01451 +/- 0.00294 | 0.01209 +/- 0.00110 | 0.01644 +/- 0.00328 | 0.01427 +/- 0.00135 | 0.01115 +/- 0.00257 | 0.01220 +/- 0.00280 |
| fetch_slide/object_speed_p90 | 0.03192 +/- 0.00314 | 0.02992 +/- 0.00163 | 0.03375 +/- 0.00346 | 0.03161 +/- 7.281e-04 | 0.02931 +/- 0.00564 | 0.02987 +/- 0.00264 |
| fetch_slide/object_velocity_alignment_mean | 0.24769 +/- 0.11291 | 0.25509 +/- 0.08960 | 0.28302 +/- 0.06109 | 0.23865 +/- 0.03725 | 0.26150 +/- 0.08273 | 0.26329 +/- 0.09365 |
| fetch_slide/strike_contact_like_rate | 0.57500 +/- 0.08663 | 0.65180 +/- 0.07738 | 0.49020 +/- 0.10224 | 0.57720 +/- 0.05498 | 0.73740 +/- 0.08104 | 0.65120 +/- 0.12880 |
| fetch_slide/progress_rate | 0.84653 +/- 0.06889 | 0.87673 +/- 0.03477 | 0.85776 +/- 0.06308 | 0.87898 +/- 0.03643 | 0.90020 +/- 0.05314 | 0.89612 +/- 0.06255 |
| fetch_slide/overshoot_rate | 1.000e-03 +/- 0.00224 | 0.00220 +/- 0.00383 | 0.00420 +/- 0.00626 | 0.00280 +/- 0.00409 | 0.00160 +/- 0.00358 | 8.000e-04 +/- 0.00179 |
| fetch_slide/phase_distribution_approach | 0.24440 +/- 0.07738 | 0.22540 +/- 0.05318 | 0.37120 +/- 0.08535 | 0.27220 +/- 0.01348 | 0.17680 +/- 0.03412 | 0.22800 +/- 0.08554 |
| fetch_slide/phase_distribution_strike | 0.52860 +/- 0.09910 | 0.58460 +/- 0.08753 | 0.45760 +/- 0.09274 | 0.54780 +/- 0.06644 | 0.65840 +/- 0.09148 | 0.59500 +/- 0.11313 |
| fetch_slide/phase_distribution_slide | 0.00580 +/- 0.00517 | 0.00460 +/- 0.00439 | 0.01240 +/- 0.00789 | 0.00540 +/- 0.00611 | 0.00540 +/- 0.01152 | 0.00260 +/- 0.00207 |
| fetch_slide/phase_distribution_near_goal | 0.22120 +/- 0.08231 | 0.18540 +/- 0.05271 | 0.15880 +/- 0.03889 | 0.17460 +/- 0.07206 | 0.15940 +/- 0.07664 | 0.17440 +/- 0.05186 |
| fetch_slide/quality_speed_mean | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.02072 +/- 0.00438 | 0 +/- 0 |
| fetch_slide/quality_alignment_mean | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.03549 +/- 0.01425 | 0 +/- 0 |
| fetch_slide/quality_progress_mean | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.04353 +/- 0.01236 | 0 +/- 0 |
| fetch_slide/quality_near_stability_mean | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.00883 +/- 0.00437 | 0 +/- 0 |

### Phase: Approach Metrics
| Metric | B0 | V5-NoQ-LS1024 | V5-NoQ-LS512 | V5-NoQ-LS256 | V5-Quality-LS512 | V6-Anneal-LS512 |
| --- | --- | --- | --- | --- | --- | --- |
| phase/approach/count_rate | 0.30703 +/- 0.03773 | 0.29766 +/- 0.02669 | 0.29609 +/- 0.02582 | 0.30078 +/- 0.05153 | 0.30078 +/- 0.05116 | 0.30859 +/- 0.06347 |
| phase/approach/mean_w_P | 1.0000 +/- 0 | 1.4288 +/- 0.03179 | 1.2750 +/- 0.08394 | 1.1859 +/- 0.05573 | 1.1662 +/- 0.03200 | 1.0000 +/- 0 |
| phase/approach/support_rate | 0 +/- 0 | 0.42877 +/- 0.03179 | 0.27502 +/- 0.08394 | 0.18592 +/- 0.05573 | 0.29700 +/- 0.05609 | 0 +/- 0 |
| phase/approach/fallback_rate | 1.0000 +/- 0 | 0.57123 +/- 0.03179 | 0.72498 +/- 0.08394 | 0.81408 +/- 0.05573 | 0.70300 +/- 0.05609 | 1.0000 +/- 0 |
| phase/approach/local_scan_truncated_rate | 0 +/- 0 | 1.0000 +/- 0 | 1.0000 +/- 0 | 1.0000 +/- 0 | 1.0000 +/- 0 | 0 +/- 0 |
| phase/approach/td_error_abs_mean | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| phase/approach/support_quality_mean | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.12254 +/- 0.03134 | 0 +/- 0 |
| phase/approach/support_score_mean | n/a | n/a | 0.27502 +/- 0.08394 | 0.18592 +/- 0.05573 | 0.16616 +/- 0.03200 | 0 +/- 0 |
| phase/approach/same_bin_match_count_mean | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.84688 +/- 0.29892 | 0 +/- 0 |

### Phase: Strike Metrics
| Metric | B0 | V5-NoQ-LS1024 | V5-NoQ-LS512 | V5-NoQ-LS256 | V5-Quality-LS512 | V6-Anneal-LS512 |
| --- | --- | --- | --- | --- | --- | --- |
| phase/strike/count_rate | 0.27500 +/- 0.03157 | 0.28281 +/- 0.04065 | 0.27969 +/- 0.02480 | 0.28828 +/- 0.05051 | 0.28203 +/- 0.03520 | 0.27266 +/- 0.03226 |
| phase/strike/mean_w_P | 1.0000 +/- 0 | 1.3236 +/- 0.04568 | 1.2762 +/- 0.05437 | 1.1441 +/- 0.07064 | 1.0712 +/- 0.02066 | 1.0000 +/- 0 |
| phase/strike/support_rate | 0 +/- 0 | 0.32358 +/- 0.04568 | 0.27620 +/- 0.05437 | 0.14412 +/- 0.07064 | 0.23117 +/- 0.05812 | 0 +/- 0 |
| phase/strike/fallback_rate | 1.0000 +/- 0 | 0.67642 +/- 0.04568 | 0.72380 +/- 0.05437 | 0.85588 +/- 0.07064 | 0.76883 +/- 0.05812 | 1.0000 +/- 0 |
| phase/strike/local_scan_truncated_rate | 0 +/- 0 | 0.99474 +/- 0.00731 | 0.99110 +/- 0.00815 | 0.99344 +/- 0.00904 | 0.99750 +/- 0.00559 | 0 +/- 0 |
| phase/strike/td_error_abs_mean | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| phase/strike/support_quality_mean | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.01818 +/- 0.00897 | 0 +/- 0 |
| phase/strike/support_score_mean | n/a | n/a | 0.27620 +/- 0.05437 | 0.14412 +/- 0.07064 | 0.07122 +/- 0.02066 | 0 +/- 0 |
| phase/strike/same_bin_match_count_mean | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.46254 +/- 0.22331 | 0 +/- 0 |

### Phase: Slide Metrics
| Metric | B0 | V5-NoQ-LS1024 | V5-NoQ-LS512 | V5-NoQ-LS256 | V5-Quality-LS512 | V6-Anneal-LS512 |
| --- | --- | --- | --- | --- | --- | --- |
| phase/slide/count_rate | 0.01328 +/- 0.00810 | 0.01484 +/- 0.01217 | 0.01250 +/- 0.00327 | 0.01797 +/- 0.01254 | 0.01641 +/- 0.01336 | 0.01172 +/- 0.00916 |
| phase/slide/mean_w_P | 1.0000 +/- 0 | 1.3000 +/- 0.29814 | 1.3167 +/- 0.32489 | 1.3175 +/- 0.23132 | 1.6235 +/- 0.35664 | 1.0000 +/- 0 |
| phase/slide/support_rate | 0 +/- 0 | 0.30000 +/- 0.29814 | 0.31667 +/- 0.32489 | 0.31746 +/- 0.23132 | 0.66667 +/- 0.31180 | 0 +/- 0 |
| phase/slide/fallback_rate | 1.0000 +/- 0 | 0.70000 +/- 0.29814 | 0.68333 +/- 0.32489 | 0.68254 +/- 0.23132 | 0.33333 +/- 0.31180 | 1.0000 +/- 0 |
| phase/slide/local_scan_truncated_rate | 0 +/- 0 | 0.11111 +/- 0.24845 | 0.66667 +/- 0.25685 | 0.83810 +/- 0.16701 | 0.66111 +/- 0.39107 | 0 +/- 0 |
| phase/slide/td_error_abs_mean | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| phase/slide/support_quality_mean | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.60915 +/- 0.37195 | 0 +/- 0 |
| phase/slide/support_score_mean | n/a | n/a | 0.31667 +/- 0.32489 | 0.31746 +/- 0.23132 | 0.62353 +/- 0.35664 | 0 +/- 0 |
| phase/slide/same_bin_match_count_mean | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 2.9222 +/- 1.2258 | 0 +/- 0 |

### Phase: Near Goal Metrics
| Metric | B0 | V5-NoQ-LS1024 | V5-NoQ-LS512 | V5-NoQ-LS256 | V5-Quality-LS512 | V6-Anneal-LS512 |
| --- | --- | --- | --- | --- | --- | --- |
| phase/near_goal/count_rate | 0.40469 +/- 0.02096 | 0.40469 +/- 0.01055 | 0.41172 +/- 0.04585 | 0.39297 +/- 0.01477 | 0.40078 +/- 0.05015 | 0.40703 +/- 0.04998 |
| phase/near_goal/mean_w_P | 1.0000 +/- 0 | 1.0521 +/- 0.02290 | 1.0283 +/- 0.01352 | 1.0098 +/- 0.01178 | 1.0132 +/- 0.00428 | 1.0000 +/- 0 |
| phase/near_goal/support_rate | 0 +/- 0 | 0.05206 +/- 0.02290 | 0.02830 +/- 0.01352 | 0.00983 +/- 0.01178 | 0.02162 +/- 0.00904 | 0 +/- 0 |
| phase/near_goal/fallback_rate | 1.0000 +/- 0 | 0.94794 +/- 0.02290 | 0.97170 +/- 0.01352 | 0.99017 +/- 0.01178 | 0.97838 +/- 0.00904 | 1.0000 +/- 0 |
| phase/near_goal/local_scan_truncated_rate | 0 +/- 0 | 0.99234 +/- 0.00798 | 1.0000 +/- 0 | 0.99794 +/- 0.00461 | 0.99815 +/- 0.00414 | 0 +/- 0 |
| phase/near_goal/td_error_abs_mean | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 |
| phase/near_goal/support_quality_mean | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.02133 +/- 0.00789 | 0 +/- 0 |
| phase/near_goal/support_score_mean | n/a | n/a | 0.02830 +/- 0.01352 | 0.00983 +/- 0.01178 | 0.01324 +/- 0.00428 | 0 +/- 0 |
| phase/near_goal/same_bin_match_count_mean | n/a | n/a | 0 +/- 0 | 0 +/- 0 | 0.04759 +/- 0.01354 | 0 +/- 0 |

## Quick Reading
- Best mean final success among these runs: B0 at `0.60000`; V6-Anneal-LS512 reached `0.53000`.
- Best mean AUC among these runs: V5-NoQ-LS256 at `0.32769` versus B0 `0.31256`.
- Best mean best-success among SABER variants here: V6-Anneal-LS512 at `0.78000` versus B0 `0.74000`.
- V6-Anneal-LS512 is much faster than fixed V5 LS512: `62.490` vs `26.847` steps/sec, because support lookup is disabled after annealing.


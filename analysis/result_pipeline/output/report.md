# SABER Result Statistics Pipeline Report

Generated from configured `summary.json` and metric CSV artifacts.

## Aggregate Metrics

| method_id | n | final_success_mean | final_success_std | success_auc_mean | success_auc_std | median_hit90_step | hit90_reached | steps_per_sec_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drawer_ddpg_her_2m | 5 | 0.39 | 0.5343 | 0.2073 | 0.2868 | 960000 | 2 | 41.0518 |
| drawer_ddpg_random_fixed_count_2m | 5 | 0.8 | 0.4472 | 0.6211 | 0.352 | 400000 | 4 | 41.0005 |
| drawer_ddpg_saber_2m | 5 | 1 | 0 | 0.7537 | 0.0444 | 500000 | 5 | 23.333 |
| drawer_td3_actor_critic_2m | 5 | 0.6 | 0.5477 | 0.4316 | 0.3947 | 600000 | 3 | 21.7535 |
| drawer_td3_actor_only_2m | 5 | 0.4 | 0.5477 | 0.2491 | 0.341 | 780000 | 2 | 21.4837 |
| drawer_td3_her_2m | 5 | 0.19 | 0.4249 | 0.1265 | 0.2827 | 740000 | 1 | 41.5473 |
| drawer_td3_random_fixed_count_2m | 5 | 0.2 | 0.4472 | 0.1439 | 0.3217 | 520000 | 1 | 41.5974 |
| drawer_td3_random_support_2m | 5 | 0.6 | 0.5477 | 0.4264 | 0.392 | 550000 | 3 | 20.8377 |
| drawer_td3_saber_2m | 5 | 0.59 | 0.539 | 0.3618 | 0.3383 | 780000 | 3 | 21.5696 |
| drawer_td3_shuffled_weight_2m | 5 | 0 | 0 | 0.0001 | 0.0001 |  | 0 | 20.0339 |
| fetchslide_ddpg_her | 5 | 0.63 | 0.1037 | 0.3244 | 0.0229 |  | 0 | 65.1288 |
| fetchslide_ddpg_saber | 5 | 0.59 | 0.114 | 0.301 | 0.0282 |  | 0 | 22.9009 |
| fetchslide_sac_her | 5 | 0.82 | 0.1151 | 0.5003 | 0.03 | 900000 | 3 | 66.7848 |
| fetchslide_sac_saber | 5 | 0.84 | 0.0822 | 0.5354 | 0.0223 | 912500 | 4 | 30.5729 |
| fetchslide_td3_her | 5 | 0.63 | 0.1823 | 0.2814 | 0.066 |  | 0 | 111.4325 |
| fetchslide_td3_saber | 5 | 0.73 | 0.1718 | 0.3373 | 0.0115 | 900000 | 1 | 40.2427 |
| fpp_ddpg_her | 5 | 0.94 | 0.0418 | 0.5388 | 0.1254 | 600000 | 5 | 65.0232 |
| fpp_ddpg_per_full | 5 | 0.78 | 0.4382 | 0.4603 | 0.2384 | 640000 | 4 | 58.0162 |
| fpp_ddpg_per_style | 5 | 0.93 | 0.1037 | 0.5684 | 0.0688 | 630000 | 5 | 61.0697 |
| fpp_ddpg_random_fixed_count_k94 | 5 | 0.96 | 0.0418 | 0.5402 | 0.0173 | 630000 | 5 | 64.5169 |
| fpp_ddpg_random_support | 5 | 0.95 | 0.05 | 0.6041 | 0.0161 | 540000 | 5 | 41.6346 |
| fpp_ddpg_saber | 5 | 0.96 | 0.0418 | 0.6091 | 0.0378 | 540000 | 5 | 41.717 |
| fpp_ddpg_shuffled_weight | 5 | 0.96 | 0.0548 | 0.579 | 0.0342 | 620000 | 5 | 41.5842 |
| fpp_sac_her | 5 | 0.79 | 0.2881 | 0.4629 | 0.0772 | 800000 | 3 | 38.9266 |
| fpp_sac_saber | 5 | 0.99 | 0.0224 | 0.5654 | 0.0302 | 750000 | 5 | 29.0943 |
| fpp_td3_her | 5 | 0.84 | 0.1884 | 0.3716 | 0.0321 | 980000 | 3 | 65.3211 |
| fpp_td3_saber | 5 | 0.97 | 0.0447 | 0.4424 | 0.035 | 840000 | 5 | 41.2453 |

## Paired Bootstrap CI

| pair_id | metric | n | mean_delta | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| drawer_ddpg_random_fixed_count_minus_her | final_success | 5 | 0.41 | -0.39 | 1 |
| drawer_ddpg_random_fixed_count_minus_her | success_auc | 5 | 0.4137 | -0.0717 | 0.8004 |
| drawer_ddpg_saber_minus_her | final_success | 5 | 0.61 | 0.21 | 1 |
| drawer_ddpg_saber_minus_her | success_auc | 5 | 0.5464 | 0.3399 | 0.741 |
| drawer_ddpg_saber_minus_random_fixed_count | final_success | 5 | 0.2 | 0 | 0.6 |
| drawer_ddpg_saber_minus_random_fixed_count | success_auc | 5 | 0.1326 | -0.069 | 0.4514 |
| drawer_random_support_minus_saber | final_success | 5 | 0.01 | -0.57 | 0.6 |
| drawer_random_support_minus_saber | success_auc | 5 | 0.0646 | -0.2718 | 0.4571 |
| drawer_saber_minus_her | final_success | 5 | 0.4 | 0 | 0.8 |
| drawer_saber_minus_her | success_auc | 5 | 0.2353 | -0.0582 | 0.5403 |
| drawer_shuffled_minus_saber | final_success | 5 | -0.59 | -0.99 | -0.19 |
| drawer_shuffled_minus_saber | success_auc | 5 | -0.3618 | -0.6261 | -0.0975 |
| fetchslide_ddpg_saber_minus_her | final_success | 5 | -0.04 | -0.15 | 0.06 |
| fetchslide_ddpg_saber_minus_her | last_quarter_auc | 5 | -0.073 | -0.1305 | -0.0135 |
| fetchslide_ddpg_saber_minus_her | success_auc | 5 | -0.0233 | -0.0445 | 0.0055 |
| fetchslide_sac_saber_minus_her | final_success | 5 | 0.02 | -0.1 | 0.14 |
| fetchslide_sac_saber_minus_her | last_quarter_auc | 5 | 0.0305 | 0.0015 | 0.056 |
| fetchslide_sac_saber_minus_her | success_auc | 5 | 0.0351 | -0.0035 | 0.0682 |
| fetchslide_td3_saber_minus_her | final_success | 5 | 0.1 | -0.03 | 0.26 |
| fetchslide_td3_saber_minus_her | last_quarter_auc | 5 | 0.083 | 0.0195 | 0.1895 |
| fetchslide_td3_saber_minus_her | success_auc | 5 | 0.0559 | 0.0042 | 0.1092 |
| fpp_ddpg_random_fixed_count_minus_her | final_success | 5 | 0.02 | -0.02 | 0.07 |
| fpp_ddpg_random_fixed_count_minus_her | success_auc | 5 | 0.0014 | -0.0799 | 0.1134 |
| fpp_ddpg_random_support_minus_her | final_success | 5 | 0.01 | -0.06 | 0.07 |
| fpp_ddpg_random_support_minus_her | success_auc | 5 | 0.0653 | -0.0169 | 0.1707 |
| fpp_ddpg_random_support_minus_random_fixed_count | final_success | 5 | -0.01 | -0.07 | 0.06 |
| fpp_ddpg_random_support_minus_random_fixed_count | success_auc | 5 | 0.0639 | 0.0384 | 0.087 |
| fpp_ddpg_random_support_minus_shuffled | final_success | 5 | -0.01 | -0.06 | 0.03 |
| fpp_ddpg_random_support_minus_shuffled | success_auc | 5 | 0.0251 | -0.0094 | 0.0466 |
| fpp_ddpg_saber_minus_her | final_success | 5 | 0.02 | -0.04 | 0.07 |
| fpp_ddpg_saber_minus_her | success_auc | 5 | 0.0703 | -0.0163 | 0.1925 |
| fpp_ddpg_saber_minus_per_style | final_success | 5 | 0.03 | -0.04 | 0.12 |
| fpp_ddpg_saber_minus_per_style | success_auc | 5 | 0.0407 | -0.0158 | 0.1229 |
| fpp_ddpg_saber_minus_random_fixed_count | final_success | 5 | 0 | -0.04 | 0.05 |
| fpp_ddpg_saber_minus_random_fixed_count | success_auc | 5 | 0.0689 | 0.0376 | 0.1034 |
| fpp_ddpg_saber_minus_random_support | final_success | 5 | 0.01 | -0.03 | 0.06 |
| fpp_ddpg_saber_minus_random_support | success_auc | 5 | 0.005 | -0.0258 | 0.0365 |
| fpp_ddpg_saber_minus_shuffled | final_success | 5 | -0 | -0.03 | 0.03 |
| fpp_ddpg_saber_minus_shuffled | success_auc | 5 | 0.0301 | 0.0021 | 0.0639 |
| fpp_ddpg_shuffled_minus_her | final_success | 5 | 0.02 | -0.04 | 0.07 |
| fpp_ddpg_shuffled_minus_her | success_auc | 5 | 0.0402 | -0.0596 | 0.1827 |
| fpp_ddpg_shuffled_minus_per_style | final_success | 5 | 0.03 | -0.02 | 0.09 |
| fpp_ddpg_shuffled_minus_per_style | success_auc | 5 | 0.0106 | -0.0208 | 0.0571 |
| fpp_sac_saber_minus_her | final_success | 5 | 0.2 | -0.02 | 0.43 |
| fpp_sac_saber_minus_her | success_auc | 5 | 0.1024 | 0.0574 | 0.1495 |
| fpp_td3_saber_minus_her | final_success | 5 | 0.13 | -0.03 | 0.31 |
| fpp_td3_saber_minus_her | success_auc | 5 | 0.0708 | 0.0319 | 0.0946 |

## Figures

- `analysis/result_pipeline/output/figures/drawer_controls_success_curve.pdf`
- `analysis/result_pipeline/output/figures/drawer_controls_success_curve.png`
- `analysis/result_pipeline/output/figures/drawer_ddpg_success_curve.pdf`
- `analysis/result_pipeline/output/figures/drawer_ddpg_success_curve.png`
- `analysis/result_pipeline/output/figures/drawer_td3_success_curve.pdf`
- `analysis/result_pipeline/output/figures/drawer_td3_success_curve.png`
- `analysis/result_pipeline/output/figures/fetchslide_ddpg_success_curve.pdf`
- `analysis/result_pipeline/output/figures/fetchslide_ddpg_success_curve.png`
- `analysis/result_pipeline/output/figures/fetchslide_sac_success_curve.pdf`
- `analysis/result_pipeline/output/figures/fetchslide_sac_success_curve.png`
- `analysis/result_pipeline/output/figures/fetchslide_td3_success_curve.pdf`
- `analysis/result_pipeline/output/figures/fetchslide_td3_success_curve.png`
- `analysis/result_pipeline/output/figures/fpp_ddpg_success_curve.pdf`
- `analysis/result_pipeline/output/figures/fpp_ddpg_success_curve.png`
- `analysis/result_pipeline/output/figures/fpp_ddpg_per_success_curve.pdf`
- `analysis/result_pipeline/output/figures/fpp_ddpg_per_success_curve.png`
- `analysis/result_pipeline/output/figures/fpp_sac_success_curve.pdf`
- `analysis/result_pipeline/output/figures/fpp_sac_success_curve.png`
- `analysis/result_pipeline/output/figures/fpp_td3_success_curve.pdf`
- `analysis/result_pipeline/output/figures/fpp_td3_success_curve.png`

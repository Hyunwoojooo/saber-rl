# TD3 SABER Action-Diversity Ablation

All methods are recomputed on exact evaluations at 10k, 20k, ..., 1M.
No interpolation, smoothing, extrapolation, or forward filling is used.
Success uses `eval/env_success_mean` with `eval/success_mean` only as fallback.
Production Diverse and Random-Diversity-Count artifacts use `Dmin=8`.

## Discovered Runs

| Method | Seed | Run |
|---|---:|---|
| TD3+HER | 1 | `experiment_FPP/runs/td3_her_baseline_ls512/fetch_pick_and_place_td3_her_baseline_ls512_s1_1000000_ls512` |
| TD3+HER | 2 | `experiment_FPP/runs/td3_her_baseline_ls512/fetch_pick_and_place_td3_her_baseline_ls512_s2_1000000_ls512` |
| TD3+HER | 3 | `experiment_FPP/runs/td3_her_baseline_ls512/fetch_pick_and_place_td3_her_baseline_ls512_s3_1000000_ls512` |
| TD3+HER | 4 | `experiment_FPP/runs/td3_her_baseline_ls512/fetch_pick_and_place_td3_her_baseline_ls512_s4_1000000_ls512` |
| TD3+HER | 5 | `experiment_FPP/runs/td3_her_baseline_ls512/fetch_pick_and_place_td3_her_baseline_ls512_s5_1000000_ls512` |
| TD3+HER+SABER | 1 | `experiment_FPP/runs/td3_saber_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s1_1000000_ls512` |
| TD3+HER+SABER | 2 | `experiment_FPP/runs/td3_saber_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s2_1000000_ls512` |
| TD3+HER+SABER | 3 | `experiment_FPP/runs/td3_saber_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s3_1000000_ls512` |
| TD3+HER+SABER | 4 | `experiment_FPP/runs/td3_saber_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s4_1000000_ls512` |
| TD3+HER+SABER | 5 | `experiment_FPP/runs/td3_saber_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s5_1000000_ls512` |
| TD3+HER+SABER-Diverse | 1 | `experiment_FPP/runs/td3_saber_diverse_d8_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_diverse_d8_critic_only_v4_s1_1000000_ls512` |
| TD3+HER+SABER-Diverse | 2 | `experiment_FPP/runs/td3_saber_diverse_d8_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_diverse_d8_critic_only_v4_s2_1000000_ls512` |
| TD3+HER+SABER-Diverse | 3 | `experiment_FPP/runs/td3_saber_diverse_d8_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_diverse_d8_critic_only_v4_s3_1000000_ls512` |
| TD3+HER+SABER-Diverse | 4 | `experiment_FPP/runs/td3_saber_diverse_d8_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_diverse_d8_critic_only_v4_s4_1000000_ls512` |
| TD3+HER+SABER-Diverse | 5 | `experiment_FPP/runs/td3_saber_diverse_d8_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_diverse_d8_critic_only_v4_s5_1000000_ls512` |
| Random-Diversity-Count | 1 | `experiment_FPP/runs/td3_saber_random_diversity_count_d8_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_random_diversity_count_d8_critic_only_v4_s1_1000000_ls512` |
| Random-Diversity-Count | 2 | `experiment_FPP/runs/td3_saber_random_diversity_count_d8_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_random_diversity_count_d8_critic_only_v4_s2_1000000_ls512` |
| Random-Diversity-Count | 3 | `experiment_FPP/runs/td3_saber_random_diversity_count_d8_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_random_diversity_count_d8_critic_only_v4_s3_1000000_ls512` |
| Random-Diversity-Count | 4 | `experiment_FPP/runs/td3_saber_random_diversity_count_d8_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_random_diversity_count_d8_critic_only_v4_s4_1000000_ls512` |
| Random-Diversity-Count | 5 | `experiment_FPP/runs/td3_saber_random_diversity_count_d8_critic_only_v4_ls512/fetch_pick_and_place_td3_saber_random_diversity_count_d8_critic_only_v4_s5_1000000_ls512` |

## Aggregate Results

Mean is arithmetic mean and standard deviation is sample std (ddof=1).

| Method | AUC mean +/- std | Final mean +/- std | Hit90 reached | Median Hit90 |
|---|---:|---:|---:|---:|
| TD3+HER | 0.371616 +/- 0.032057 | 0.840000 +/- 0.188414 | 3/5 | 980000 |
| TD3+HER+SABER | 0.442374 +/- 0.035036 | 0.970000 +/- 0.044721 | 5/5 | 840000 |
| TD3+HER+SABER-Diverse | 0.357273 +/- 0.045122 | 0.800000 +/- 0.183712 | 3/5 | 990000 |
| Random-Diversity-Count | 0.361263 +/- 0.052388 | 0.790000 +/- 0.265518 | 3/5 | 920000 |

AUC ranking: TD3+HER+SABER > TD3+HER > Random-Diversity-Count > TD3+HER+SABER-Diverse.

## Paired AUC Differences

Intervals are 95% paired percentile bootstrap CIs with 10,000 resamples
and RNG seed 27027. Paired seed differences are resampled directly.

| Comparison | Mean delta | 95% CI |
|---|---:|---:|
| Diverse - main SABER | -0.085101 | [-0.112222, -0.059444] |
| Diverse - HER | -0.014343 | [-0.040354, 0.011667] |
| Diverse - Random-Diversity-Count | -0.003990 | [-0.051414, 0.065303] |

The intervals are uncertainty summaries for paired AUC differences; this report does not add a separate significance claim.

## Reproduction

`python experiment_FPP/td3_saber_action_diversity_ablation/analyze_results.py --repo-root . --diverse-root experiment_FPP/runs/td3_saber_diverse_d8_critic_only_v4_ls512 --random-count-root experiment_FPP/runs/td3_saber_random_diversity_count_d8_critic_only_v4_ls512 --output-dir experiment_FPP/reports/td3_saber_action_diversity_d8 --bootstrap-resamples 10000 --bootstrap-seed 27027`

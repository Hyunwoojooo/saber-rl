# FPP TD3 Context-Eligible Count-Matched Control

E is the pre-membership context-eligible mask, M is the Main current-policy membership mask, k=sum(M), and Z uniformly samples exactly k positions without replacement from E.
Main weights M; the control weights Z. Natural M/Z overlap is allowed.
The isolated variable is membership-aligned sample assignment. Query semantics remain current-policy and per-update counts and raw/normalized weight histograms are matched.

Performance uses exact evaluations at 10k, 20k, ..., 1M. No interpolation, smoothing, extrapolation, or forward filling is used.
Success uses eval/env_success_mean with eval/success_mean only as fallback.
AUC is trapezoidal area normalized over 10k to 1M. Mean is the arithmetic mean and std is sample std (ddof=1).
Context-count-only is a count-unmatched descriptive reference.
Random-support is excluded from the primary table because its generation-code provenance is not established.

## Discovered Runs

| Method | Seed | Run |
|---|---:|---|
| TD3+HER+SABER (Main) | 1 | data/main/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s1_1000000_ls512 |
| TD3+HER+SABER (Main) | 2 | data/main/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s2_1000000_ls512 |
| TD3+HER+SABER (Main) | 3 | data/main/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s3_1000000_ls512 |
| TD3+HER+SABER (Main) | 4 | data/main/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s4_1000000_ls512 |
| TD3+HER+SABER (Main) | 5 | data/main/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s5_1000000_ls512 |
| Context-eligible count-matched | 1 | data/control/fetch_pick_and_place_td3_saber_context_eligible_count_matched_s1_1000000_ls512 |
| Context-eligible count-matched | 2 | data/control/fetch_pick_and_place_td3_saber_context_eligible_count_matched_s2_1000000_ls512 |
| Context-eligible count-matched | 3 | data/control/fetch_pick_and_place_td3_saber_context_eligible_count_matched_s3_1000000_ls512 |
| Context-eligible count-matched | 4 | data/control/fetch_pick_and_place_td3_saber_context_eligible_count_matched_s4_1000000_ls512 |
| Context-eligible count-matched | 5 | data/control/fetch_pick_and_place_td3_saber_context_eligible_count_matched_s5_1000000_ls512 |
| Context-count-only (count-unmatched reference) | 1 | data/context_only/fetch_pick_and_place_td3_saber_context_count_only_ls512_s1_1000000_ls512 |
| Context-count-only (count-unmatched reference) | 2 | data/context_only/fetch_pick_and_place_td3_saber_context_count_only_ls512_s2_1000000_ls512 |
| Context-count-only (count-unmatched reference) | 3 | data/context_only/fetch_pick_and_place_td3_saber_context_count_only_ls512_s3_1000000_ls512 |
| Context-count-only (count-unmatched reference) | 4 | data/context_only/fetch_pick_and_place_td3_saber_context_count_only_ls512_s4_1000000_ls512 |
| Context-count-only (count-unmatched reference) | 5 | data/context_only/fetch_pick_and_place_td3_saber_context_count_only_ls512_s5_1000000_ls512 |

## Per-Seed Native vs Exact-Grid AUC

| Method | Seed | Native | Exact-grid | Native - exact | Final | Best | Hit90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TD3+HER+SABER (Main) | 1 | 0.430051 | 0.430051 | 0.000000 | 1.000000 | 1.000000 | 840000 |
| TD3+HER+SABER (Main) | 2 | 0.456818 | 0.456818 | 0.000000 | 1.000000 | 1.000000 | 860000 |
| TD3+HER+SABER (Main) | 3 | 0.495960 | 0.495960 | 0.000000 | 0.950000 | 1.000000 | 630000 |
| TD3+HER+SABER (Main) | 4 | 0.406566 | 0.406566 | 0.000000 | 0.900000 | 0.950000 | 630000 |
| TD3+HER+SABER (Main) | 5 | 0.422475 | 0.422475 | 0.000000 | 1.000000 | 1.000000 | 910000 |
| Context-eligible count-matched | 1 | 0.311616 | 0.311616 | 0.000000 | 0.750000 | 0.850000 |  |
| Context-eligible count-matched | 2 | 0.404545 | 0.404545 | 0.000000 | 0.950000 | 0.950000 | 990000 |
| Context-eligible count-matched | 3 | 0.416667 | 0.416667 | 0.000000 | 0.950000 | 0.950000 | 920000 |
| Context-eligible count-matched | 4 | 0.486869 | 0.486869 | 0.000000 | 1.000000 | 1.000000 | 630000 |
| Context-eligible count-matched | 5 | 0.334091 | 0.334091 | 0.000000 | 0.900000 | 0.900000 | 1000000 |
| Context-count-only (count-unmatched reference) | 1 | 0.292929 | 0.292929 | 0.000000 | 0.450000 | 0.750000 |  |
| Context-count-only (count-unmatched reference) | 2 | 0.440152 | 0.440152 | 0.000000 | 1.000000 | 1.000000 | 850000 |
| Context-count-only (count-unmatched reference) | 3 | 0.359596 | 0.359596 | 0.000000 | 0.650000 | 0.850000 |  |
| Context-count-only (count-unmatched reference) | 4 | 0.525758 | 0.525758 | 0.000000 | 1.000000 | 1.000000 | 630000 |
| Context-count-only (count-unmatched reference) | 5 | 0.359848 | 0.359848 | 0.000000 | 0.900000 | 0.900000 | 1000000 |

## Aggregate Results

| Method | AUC mean +/- std | Final mean +/- std | Best mean +/- std | Hit90 reached | Median Hit90 |
|---|---:|---:|---:|---:|---:|
| TD3+HER+SABER (Main) | 0.442374 +/- 0.035036 | 0.970000 +/- 0.044721 | 0.990000 +/- 0.022361 | 5/5 | 840000 |
| Context-eligible count-matched | 0.390758 +/- 0.069952 | 0.910000 +/- 0.096177 | 0.930000 +/- 0.057009 | 4/5 | 955000 |
| Context-count-only (count-unmatched reference) | 0.395657 +/- 0.089501 | 0.800000 +/- 0.242384 | 0.900000 +/- 0.106066 | 3/5 | 850000 |

## Primary Paired AUC

Main - control mean: 0.051616.
95% paired percentile bootstrap CI: [-0.016465, 0.099192] using 10,000 resamples and RNG seed 27027.
Exact two-sided sign-flip value: 0.250000.
Positive deltas: 4/5.
Leave-one-seed-out mean range: [0.034912, 0.084596].

The generated paired CI includes zero, so an independent membership-alignment contribution is not confirmed.
This is an FPP TD3 result only; it does not establish a universal or task-agnostic mechanism.

## Assignment Diagnostics

| Seed | Updates | E rate | Main rate | Assigned rate | Overlap/k | Jaccard |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 995001 | 0.959056 | 0.210955 | 0.210955 | 0.219929 | 0.126282 |
| 2 | 995001 | 0.957385 | 0.209422 | 0.209422 | 0.218603 | 0.124403 |
| 3 | 995001 | 0.956419 | 0.211973 | 0.211973 | 0.221471 | 0.126079 |
| 4 | 995001 | 0.955538 | 0.239731 | 0.239731 | 0.250916 | 0.145805 |
| 5 | 995001 | 0.951159 | 0.231479 | 0.231479 | 0.243106 | 0.140572 |

All exact-k, histogram, normalized-sum, and window-contiguity checks passed before this report was written.

## Reproduction

python experiment_FPP/td3_saber_context_eligible_count_matched_ls512/analyze_results.py --repo-root . --output-dir results

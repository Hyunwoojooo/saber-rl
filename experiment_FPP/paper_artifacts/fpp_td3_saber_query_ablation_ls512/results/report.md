# TD3 SABER Support-Query Ablation

All methods are recomputed on exact evaluations at 10k, 20k, ..., 1M.
No interpolation, smoothing, extrapolation, or forward filling is used.
Success uses `eval/env_success_mean` with `eval/success_mean` only as fallback.
Only the support-predicate query action changes across Main SABER and the three variants.
Replay-query retains the sampled transition in the support pool, so self-support is a stated limitation.
Context-count-only has no action membership gate and may therefore change the positive count.

## Discovered Runs

| Method | Seed | Run |
|---|---:|---|
| TD3+HER | 1 | `data/her/fetch_pick_and_place_td3_her_baseline_ls512_s1_1000000_ls512` |
| TD3+HER | 2 | `data/her/fetch_pick_and_place_td3_her_baseline_ls512_s2_1000000_ls512` |
| TD3+HER | 3 | `data/her/fetch_pick_and_place_td3_her_baseline_ls512_s3_1000000_ls512` |
| TD3+HER | 4 | `data/her/fetch_pick_and_place_td3_her_baseline_ls512_s4_1000000_ls512` |
| TD3+HER | 5 | `data/her/fetch_pick_and_place_td3_her_baseline_ls512_s5_1000000_ls512` |
| TD3+HER+SABER (Main) | 1 | `data/saber/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s1_1000000_ls512` |
| TD3+HER+SABER (Main) | 2 | `data/saber/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s2_1000000_ls512` |
| TD3+HER+SABER (Main) | 3 | `data/saber/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s3_1000000_ls512` |
| TD3+HER+SABER (Main) | 4 | `data/saber/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s4_1000000_ls512` |
| TD3+HER+SABER (Main) | 5 | `data/saber/fetch_pick_and_place_td3_saber_critic_only_v4_ls512_s5_1000000_ls512` |
| Replay-query | 1 | `data/replay/fetch_pick_and_place_td3_saber_query_replay_ls512_s1_1000000_ls512` |
| Replay-query | 2 | `data/replay/fetch_pick_and_place_td3_saber_query_replay_ls512_s2_1000000_ls512` |
| Replay-query | 3 | `data/replay/fetch_pick_and_place_td3_saber_query_replay_ls512_s3_1000000_ls512` |
| Replay-query | 4 | `data/replay/fetch_pick_and_place_td3_saber_query_replay_ls512_s4_1000000_ls512` |
| Replay-query | 5 | `data/replay/fetch_pick_and_place_td3_saber_query_replay_ls512_s5_1000000_ls512` |
| Shuffled-policy-query | 1 | `data/shuffled/fetch_pick_and_place_td3_saber_query_shuffled_ls512_s1_1000000_ls512` |
| Shuffled-policy-query | 2 | `data/shuffled/fetch_pick_and_place_td3_saber_query_shuffled_ls512_s2_1000000_ls512` |
| Shuffled-policy-query | 3 | `data/shuffled/fetch_pick_and_place_td3_saber_query_shuffled_ls512_s3_1000000_ls512` |
| Shuffled-policy-query | 4 | `data/shuffled/fetch_pick_and_place_td3_saber_query_shuffled_ls512_s4_1000000_ls512` |
| Shuffled-policy-query | 5 | `data/shuffled/fetch_pick_and_place_td3_saber_query_shuffled_ls512_s5_1000000_ls512` |
| Context-count-only | 1 | `data/context_only/fetch_pick_and_place_td3_saber_context_count_only_ls512_s1_1000000_ls512` |
| Context-count-only | 2 | `data/context_only/fetch_pick_and_place_td3_saber_context_count_only_ls512_s2_1000000_ls512` |
| Context-count-only | 3 | `data/context_only/fetch_pick_and_place_td3_saber_context_count_only_ls512_s3_1000000_ls512` |
| Context-count-only | 4 | `data/context_only/fetch_pick_and_place_td3_saber_context_count_only_ls512_s4_1000000_ls512` |
| Context-count-only | 5 | `data/context_only/fetch_pick_and_place_td3_saber_context_count_only_ls512_s5_1000000_ls512` |

## Aggregate Results

Mean is arithmetic mean and standard deviation is sample std (ddof=1).

| Method | AUC mean +/- std | Final mean +/- std | Hit90 reached | Median Hit90 |
|---|---:|---:|---:|---:|
| TD3+HER | 0.371616 +/- 0.032057 | 0.840000 +/- 0.188414 | 3/5 | 980000 |
| TD3+HER+SABER (Main) | 0.442374 +/- 0.035036 | 0.970000 +/- 0.044721 | 5/5 | 840000 |
| Replay-query | 0.367778 +/- 0.042310 | 0.660000 +/- 0.219089 | 1/5 | 920000 |
| Shuffled-policy-query | 0.346616 +/- 0.051300 | 0.550000 +/- 0.226385 | 1/5 | 890000 |
| Context-count-only | 0.395657 +/- 0.089501 | 0.800000 +/- 0.242384 | 3/5 | 850000 |

AUC ranking: TD3+HER+SABER (Main) > Context-count-only > TD3+HER > Replay-query > Shuffled-policy-query.

## Paired AUC Differences

Intervals are 95% paired percentile bootstrap CIs with 10,000 resamples
and RNG seed 27027. Paired seed differences are resampled directly.
Exact sign-flip p-values are two-sided over all 32 sign assignments for five paired seeds.

| Comparison | Mean delta | 95% CI | Exact sign-flip p |
|---|---:|---:|---:|
| Main SABER - Replay-query | 0.074596 | [0.059343, 0.092273] | 0.062500 |
| Main SABER - Shuffled-policy | 0.095758 | [0.063131, 0.127576] | 0.062500 |
| Main SABER - Context-count-only | 0.046717 | [-0.040909, 0.121919] | 0.312500 |

## Mechanism Snapshots

Values are means over exact 10k evaluation snapshots, then arithmetic means across available seeds.
Blank entries were not logged by the legacy run and are not imputed.

| Method | Context eligible | Support positive | Query bin present given context |
|---|---:|---:|---:|
| TD3+HER |  | 0.000000 |  |
| TD3+HER+SABER (Main) |  | 0.219063 |  |
| Replay-query | 0.958000 | 0.425859 | 0.444305 |
| Shuffled-policy-query | 0.955125 | 0.153805 | 0.160971 |
| Context-count-only | 0.956297 | 0.956297 |  |

The intervals, sign-flip values, and leave-one-seed-out summaries are reported as generated; this report does not add a separate significance claim.

## Reproduction

`python code/analyze_results.py --repo-root . --her-root data/her --saber-root data/saber --replay-root data/replay --shuffled-root data/shuffled --context-only-root data/context_only --output-dir results --bootstrap-resamples 10000 --bootstrap-seed 27027`

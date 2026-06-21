# DrawerOpen TD3 B3 Support-Rate-Cap 0.4

This is an independent code snapshot for a DrawerOpen TD3+HER+B3-Critic-Only ablation. It was copied from the existing DrawerOpen TD3 runtime-optimized B3 code and is kept in a separate folder so the previous experiment code is not overwritten.

## Variant

- Task: MetaWorld `drawer-open-v3`
- Backbone: TD3 + HER
- SABER/B3 scope: critic-only
- Method: `b3_support_capped_runtime_opt`
- Support cap: `support_rate_cap = 0.4`
- Default run length: 2M steps, seeds 1-5, 5 parallel CUDA workers

## Weighting Rule

The code first computes true SABER support-positive samples exactly as in the runtime-optimized B3 critic-only implementation:

1. local context lookup
2. policy-action support check
3. `N_local_min`, `M_Q_min`, `local_search_max` filtering
4. support cache, local-scan cache, Numba scanner, and short-circuit optimizations

Then it caps high-weight assignment:

- Let `S` be the support-positive samples in the batch.
- Let `k_cap = floor(support_rate_cap * batch_size)`.
- Assign high weight to at most `min(|S|, k_cap)` samples.
- If `|S| > k_cap`, select the retained samples only from `S` using the weight RNG.
- All non-retained samples receive raw weight `1.0`.
- Batch normalization is applied afterward when `weighting_mode=batch_norm`.

This is not Fixed-Random: non-support samples cannot receive high weight. The cap only reduces density when the SABER support-positive rate becomes too large.

## Diagnostics

The CSV logs include the existing random/shuffled diagnostics plus cap-specific fields:

- `reference_high_weight_rate`
- `assigned_high_weight_rate`
- `support_weight_alignment_rate`
- `support_weight_overlap_rate`
- `support_rate_cap`
- `support_rate_cap_count`
- `support_rate_cap_reference_count`
- `support_rate_cap_kept_count`
- `support_rate_cap_applied`

Expected behavior for this variant:

- `assigned_high_weight_rate <= 0.4`
- `support_weight_alignment_rate` should stay near `1.0` when high weights are assigned
- runtime optimization metrics should remain comparable to true B3 runtime-opt runs

## Runner

From the repository root:

```bash
bash metaworld_Drawer_Open/TD3-HER-B3-Critic-Only-SupportRateCap-04/run_td3_drawer_open_b3_support_rate_cap_04_2m_5parallel.sh
```

The runner writes results under this folder's `results_2m/` directory and logs under `logs_2m/`.

## 2M Results

Status as of 2026-06-21 10:32 KST:

- `TD3-HER-B3-Critic-Only-SupportRateCap-04-2M` completed 5 seeds at 2M steps.
- The run used the same sparse DrawerOpen goal wrapper, TD3+HER backbone,
  `drawer-open-v3`, `reward_function_version=v2`, and 20 evaluation episodes as
  the completed DrawerOpen TD3 experiments.
- The cap was active with `support_rate_cap=0.4` and critic-only weighting.

Aggregate metrics use 5 seeds:

| Variant | Final success | Best success | AUC success | Hit90 | Speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3-HER-B3-Critic-Only-SupportRateCap-04-2M | 0.190 +/- 0.425 | 0.210 +/- 0.442 | 0.126 +/- 0.282 | 800k, 1/5 seeds | 20.92 +/- 0.75 steps/s |

Seed-level final success:

| seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.95 | 0.00 | 0.00 | 0.00 |

Seed-level best success:

| seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 1.00 | 0.00 | 0.00 | 0.05 |

Implementation diagnostics, averaged over the final logged window of each seed:

| Metric | Value |
| --- | ---: |
| `support_rate_cap` | 0.400 +/- 0.000 |
| `assigned_high_weight_rate` | 0.326 +/- 0.099 |
| `reference_high_weight_rate` | 0.576 +/- 0.389 |
| `support_rate_cap_applied` | 0.584 +/- 0.534 |
| `support_weight_alignment_rate` | 1.000 +/- 0.000 |
| `support_weight_overlap_rate` | 0.326 +/- 0.099 |
| `support_cache_hit_rate` | 0.323 +/- 0.274 |

Interpretation:

- The 40% support-rate cap did not improve DrawerOpen performance. It solved
  only 1/5 seeds and underperformed the completed uncapped TD3-HER+B3-Critic-Only
  2M result.
- The cap preserved sample-level support alignment for retained high-weight
  samples, but it discarded part of the original SABER-positive set whenever the
  support-positive rate exceeded the cap.
- This suggests the main B3 effect is not explained by a fixed high-weight
  density such as 30-50%. The more important factor is computing the support
  signal accurately and assigning the resulting critic weights to the correct
  support-positive samples.

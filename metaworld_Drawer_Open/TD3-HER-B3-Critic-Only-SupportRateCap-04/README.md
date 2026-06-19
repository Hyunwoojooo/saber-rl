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

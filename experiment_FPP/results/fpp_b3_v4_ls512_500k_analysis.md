# FPP B3 v4 local_search_max=512 500k screening

Date: 2026-05-29

Source runs:
- `SABER/FetchPickAndPlace/runs/b3_critic_only_runtime_opt_v4_ls512/*_500000_ls512`
- comparison uses raw metrics from local B0 and B3 v4 1024 runs, cut to 500k and seed 1-3 only

## Conclusion

`local_search_max=512` is a promising B3 v4 variant. On the same seed subset, it improves runtime over B3 v4 1024 and does not hurt 500k learning performance. It is still slower than B0 in wall-clock terms, so it should be reported as a bounded local-scan ablation rather than a pure runtime-only optimization.

## Same Seed 1-3, 500k

| method | final success | best success | tail50k success | tail100k success | AUC | final steps/s |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.550 +/- 0.218 | 0.833 +/- 0.029 | 0.639 +/- 0.069 | 0.553 +/- 0.104 | 0.276 +/- 0.044 | 65.3 |
| B3 v4 1024 | 0.550 +/- 0.218 | 0.783 +/- 0.076 | 0.586 +/- 0.123 | 0.532 +/- 0.102 | 0.267 +/- 0.033 | 41.1 |
| B3 v4 512 | 0.683 +/- 0.236 | 0.850 +/- 0.087 | 0.694 +/- 0.128 | 0.630 +/- 0.117 | 0.304 +/- 0.048 | 49.6 |

## Runtime

| metric | B3 v4 1024 | B3 v4 512 | readout |
|---|---:|---:|---|
| final steps/s | 41.1 | 49.6 | +20.7% vs 1024 |
| time/local_context_lookup tail50 | 11.03 ms | 8.86 ms | lookup reduced |
| time/numba_support_scan tail50 | 4.36 ms | 2.87 ms | scanner reduced |
| mean scanned candidates tail50 | 594.4 | 349.7 | scan budget effective |

## Support Diagnostics

| metric | B3 v4 1024 | B3 v4 512 |
|---|---:|---:|
| psi_q_rank tail50 | 0.525 | 0.421 |
| mean_w_P tail50 | 1.525 | 1.421 |
| ESS normalized tail50 | 0.904 | 0.893 |
| local support count mean tail50 | 54.6 | 35.0 |
| place support tail50 | 0.562 | 0.554 |

The 512 variant lowers overall support rate but keeps local support counts far above `N_local_min=4` and `M_Q_min=4`. Place-phase support is roughly preserved, which is likely why performance does not collapse despite scanning fewer candidates.

## Files

- `experiment_FPP/results/fpp_b3_v4_ls512_500k_method_summary.csv`
- `experiment_FPP/results/fpp_b3_v4_ls512_500k_per_seed.csv`
- `experiment_FPP/results/fpp_b3_v4_ls512_500k_summary.json`
- compact per-run artifacts under `experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512/`

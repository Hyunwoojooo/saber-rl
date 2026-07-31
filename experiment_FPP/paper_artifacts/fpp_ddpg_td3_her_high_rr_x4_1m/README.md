# FPP DDPG/TD3 HER High-RR x4 (1M)

Compact, self-verifying seeds 1–5 artifact for the FPP DDPG and TD3 HER update-ratio-four runs. It contains sanitized configurations, environment snapshots, summaries, exact evaluation curves, runtime segments, recomputed FQ-AUC and throughput, retained launch snapshots, and cryptographic source bindings. Raw metrics, logs, checkpoints, replay buffers, and model weights are excluded.

Both DDPG and TD3 began from scratch and completed without checkpoint continuation. TD3 configs and retained logs contain stale resume fields; operator audit confirmed that these fields do not describe execution. TD3 seeds 1, 3, and 4 lack an 800k evaluation, so the Table 16 common grid is the exact 50k–1M grid with 800k removed for every row; no interpolation, smoothing, extrapolation, or forward-fill is used.

The TD3 summary throughput retained only the final runtime-counter segment. `analysis/per_seed_metrics.csv` instead computes steps/s from the sum of retained monotonic runtime segments. Segment boundaries are logger resets, not process restart or checkpoint-continuation boundaries.

Verify from this directory with `python3 code/verify_fpp_high_rr_compact_artifact.py --artifact-root .`.

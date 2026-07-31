# FPP DDPG/TD3 HER High-RR x4 (1M)

Compact, self-verifying seeds 1–5 artifact for the FPP DDPG and TD3 HER update-ratio-four runs. It contains sanitized configurations, environment snapshots, summaries, exact evaluation curves, runtime segments, recomputed FQ-AUC and throughput, retained launch snapshots, and cryptographic source bindings. Raw metrics, logs, checkpoints, replay buffers, and model weights are excluded.

DDPG completed without resume. TD3 resumed after reboot at step 800001 for seeds 1–4 and at step 630001 from `best_checkpoint.pt` for seed 5. Exact equivalence to an uninterrupted TD3 trajectory is not claimed. TD3 seeds 1, 3, and 4 lack an 800k evaluation, so the Table 16 common grid is the exact 50k–1M grid with 800k removed for every row; no interpolation, smoothing, extrapolation, or forward-fill is used.

The prior TD3 summary throughput used only the post-resume runtime counter. `analysis/per_seed_metrics.csv` instead computes steps/s from the sum of retained monotonic runtime segments. Reboot downtime and unretained discarded work are not estimated.

Verify from this directory with `python3 code/verify_fpp_high_rr_compact_artifact.py --artifact-root .`.

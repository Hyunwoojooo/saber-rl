# FPP TD3 HER vs SABER paired n=10 artifact

This directory contains the completed seeds 1-10 for the frozen confirmatory
FetchPickAndPlace TD3 experiment. Seeds 1-5 are the pre-existing runs; seeds
6-10 are the confirmatory extension. No checkpoints or replay buffers are
included.

Primary endpoint: trapezoidal success AUC over the exact 10k-1M grid, divided by
990k. No interpolation, smoothing, forward filling, seed replacement, or
post-hoc hyperparameter selection is permitted.

Run `python3 analysis/analyze_td3_n10.py --artifact-root .` from this directory,
or pass this directory explicitly from the repository root.

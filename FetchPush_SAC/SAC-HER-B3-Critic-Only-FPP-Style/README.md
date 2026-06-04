# SAC-HER-B3-Critic-Only-FPP-Style

FetchPush SAC + HER with SABER B3 support-only critic loss weighting. The support thresholds follow the FetchPickAndPlace B3-Critic-Only setting: M_B=16, M_Q_min=4, N_local_min=4, N_gap_min=128, local_search_max=1024.

Run through `FetchPush_SAC/run_all_sac_fetchpush_5parallel.sh` or the queue script. Results are written under each experiment folder's `results/` directory.

# DDPG-HER-B3-Critic-Only-FPP-Style

FetchPush DDPG + HER with SABER B3 support-only critic loss weighting. The support thresholds follow the FetchPickAndPlace B3-Critic-Only setting: M_B=16, M_Q_min=4, N_local_min=4, N_gap_min=128, local_search_max=1024.

Run through `FetchPush_DDPG/run_all_ddpg_fetchpush_5parallel.sh` or the queue script. Results are written under each experiment folder's `results/` directory.

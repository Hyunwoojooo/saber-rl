# DrawerOpen DDPG Shuffled-Policy Query, 2M

Compact paper artifact for paired seeds 1–5. Shuffled-Policy Query applies a within-minibatch derangement to current-policy actions, preserving the action multiset while disrupting sample–query alignment. All other SABER settings remain critic-only and unchanged.

AUC is the normalized trapezoidal area over the exact observed 10k–2M success grid. No smoothing, interpolation, forward filling, or extrapolation is used.

| Method | AUC mean ± sample SD | Final mean ± sample SD | Hit90 median (coverage) |
|---|---:|---:|---:|
| DrawerOpen DDPG HER | 0.207312 ± 0.286818 | 0.390000 ± 0.534322 | 960k (2/5) |
| DrawerOpen DDPG HER + SABER Main | 0.753668 ± 0.044374 | 1.000000 ± 0.000000 | 500k (5/5) |
| DrawerOpen DDPG Shuffled-Policy Query | 0.648291 ± 0.246038 | 1.000000 ± 0.000000 | 630k (5/5) |

Main − Shuffled AUC difference: 0.105377; exhaustive paired-bootstrap 95% CI [-0.080151, 0.335126]; Main is higher in 3/5 seeds.

The interval includes zero, so this cohort does not independently establish a sample-alignment effect. It does show a higher Main mean and earlier median Hit90, while the shuffled control remains substantially above HER.

Training source commit: `53e5dca5b84c90243aebb5a251c44aceb8f6f34b`. HER and Main comparator curves are copied from the verified strict context-eligible count-matched artifact and were not rerun for this comparison.

Verify from this directory:

    python artifact_tools.py verify --artifact-dir .

Checkpoints, complete training-row metrics, JSONL files, and logs are excluded.

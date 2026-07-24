# DrawerClose DDPG Shuffled-Policy Query, 2M

Compact paper artifact for paired seeds 1–5. Shuffled-Policy Query applies a within-minibatch derangement to current-policy actions, preserving the action multiset while disrupting sample–query alignment. All other SABER settings remain critic-only and unchanged.

AUC is the normalized trapezoidal area over the exact observed 10k–2M success grid. No smoothing, interpolation, forward filling, or extrapolation is used.

| Method | AUC mean ± sample SD | Final mean ± sample SD | Hit90 median (coverage) |
|---|---:|---:|---:|
| DrawerClose DDPG HER | 0.251005 ± 0.029646 | 0.160000 ± 0.181659 | 70k (5/5) |
| DrawerClose DDPG HER + SABER Main | 0.324422 ± 0.082453 | 0.280000 ± 0.251496 | 20k (5/5) |
| DrawerClose DDPG Shuffled-Policy Query | 0.291508 ± 0.055968 | 0.320000 ± 0.422197 | 60k (5/5) |

Main − Shuffled AUC difference: 0.032915; exhaustive paired-bootstrap 95% CI [-0.062739, 0.119070]; Main is higher in 4/5 seeds.

The interval includes zero, so this cohort does not independently establish a sample-alignment effect. It does show a higher Main mean and earlier median Hit90, while the shuffled control remains substantially above HER.

Training source commit: `e84e00cabf217a235ddaa84b572f5502ebc6053d`. HER and Main comparator curves are copied from the verified strict context-eligible count-matched artifact and were not rerun for this comparison.

Verify from this directory:

    python artifact_tools.py verify --artifact-dir .

Checkpoints, complete training-row metrics, JSONL files, and logs are excluded.

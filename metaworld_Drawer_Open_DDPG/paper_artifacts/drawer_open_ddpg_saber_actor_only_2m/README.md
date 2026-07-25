# DrawerOpen DDPG SABER Actor-Only, 2M

Compact paper artifact for paired seeds 1–5. Actor-Only keeps SABER’s current-policy support query and normalized per-sample weights, applies those weights only to the actor objective, and trains the critic with the standard unweighted HER Bellman loss.

AUC is the normalized trapezoidal area over the exact observed 10k–2M success grid. No smoothing, interpolation, forward filling, or extrapolation is used.

| Method | AUC mean ± sample SD | Final mean ± sample SD | Hit90 median (coverage) |
|---|---:|---:|---:|
| DrawerOpen DDPG HER | 0.207312 ± 0.286818 | 0.390000 ± 0.534322 | 960k (2/5) |
| DrawerOpen DDPG HER + SABER Main | 0.753668 ± 0.044374 | 1.000000 ± 0.000000 | 500k (5/5) |
| DrawerOpen DDPG HER + SABER Actor-Only | 0.612412 ± 0.343192 | 0.800000 ± 0.447214 | 470k (4/5) |

Main − Actor-Only AUC difference: 0.141256; exhaustive paired-bootstrap 95% CI [-0.051407, 0.478090]; Main is higher in 2/5 seeds.

This comparison tests optimization scope: Main applies support-derived weights only to critic Bellman losses, whereas Actor-Only applies the same weighting rule only to the actor. Scope-specific interpretation should use the paired interval together with failure and Hit90 coverage.

Training source commit: `823cebc`. HER and Main comparator curves are copied from the verified strict context-eligible count-matched artifact and were not rerun for this comparison.

Verify from this directory:

    python artifact_tools.py verify --artifact-dir .

Checkpoints, complete training-row metrics, JSONL files, and logs are excluded.

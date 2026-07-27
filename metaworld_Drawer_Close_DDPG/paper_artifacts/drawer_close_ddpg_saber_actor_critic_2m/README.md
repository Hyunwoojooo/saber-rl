# DrawerClose DDPG SABER Actor+Critic, 2M

Compact paper artifact for paired seeds 1–5. Actor+Critic keeps SABER’s current-policy support query and normalized per-sample weights and applies those weights to both the actor objective and critic Bellman loss.

AUC is the normalized trapezoidal area over the exact observed 10k–2M success grid. No smoothing, interpolation, forward filling, or extrapolation is used.

| Method | AUC mean ± sample SD | Final mean ± sample SD | Hit90 median (coverage) |
|---|---:|---:|---:|
| DrawerClose DDPG HER | 0.251005 ± 0.029646 | 0.160000 ± 0.181659 | 70k (5/5) |
| DrawerClose DDPG HER + SABER Main | 0.324422 ± 0.082453 | 0.280000 ± 0.251496 | 20k (5/5) |
| DrawerClose DDPG HER + SABER Actor+Critic | 0.298116 ± 0.071262 | 0.440000 ± 0.369797 | 50k (5/5) |

Main − Actor+Critic AUC difference: 0.026307; exhaustive paired-bootstrap 95% CI [-0.049925, 0.090503]; Main is higher in 3/5 seeds.

This comparison tests optimization scope: Main applies support-derived weights only to critic Bellman losses, whereas Actor+Critic extends the same weighting rule to the actor objective. Scope-specific interpretation should use the paired interval together with failure and Hit90 coverage.

Training source commit: `77b5e56eb0712497b55efac6dd0cfd6d841b2670`. HER and Main comparator curves are copied from the verified strict context-eligible count-matched artifact and were not rerun for this comparison.

Verify from this directory:

    python artifact_tools.py verify --artifact-dir .

Checkpoints, complete training-row metrics, JSONL files, and logs are excluded.

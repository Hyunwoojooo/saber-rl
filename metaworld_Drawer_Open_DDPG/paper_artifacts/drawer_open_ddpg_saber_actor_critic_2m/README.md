# DrawerOpen DDPG SABER Actor+Critic, 2M

Compact paper artifact for paired seeds 1–5. Actor+Critic keeps SABER’s current-policy support query and normalized per-sample weights and applies the same detached weight once to the critic objective and once to the actor objective. Bellman regression remains evaluated at replay actions.

AUC is the normalized trapezoidal area over the exact observed 10k–2M success grid. No smoothing, interpolation, forward filling, or extrapolation is used.

| Method | AUC mean ± sample SD | Final mean ± sample SD | Hit90 median (coverage) |
|---|---:|---:|---:|
| DrawerOpen DDPG HER | 0.207312 ± 0.286818 | 0.390000 ± 0.534322 | 960k (2/5) |
| DrawerOpen DDPG HER + SABER Main | 0.753668 ± 0.044374 | 1.000000 ± 0.000000 | 500k (5/5) |
| DrawerOpen DDPG HER + SABER Actor+Critic | 0.728015 ± 0.084312 | 0.990000 ± 0.022361 | 540k (5/5) |

Main − Actor+Critic AUC difference: 0.025653; exhaustive paired-bootstrap 95% CI [-0.052337, 0.091156]; Main is higher in 3/5 seeds.

This comparison tests optimization scope: Main applies support-derived weights only to critic Bellman losses, whereas Actor+Critic applies the same weighting rule to both objectives. Scope-specific interpretation should use the paired interval together with failure and Hit90 coverage.

Training source commit: `cb5b58889ec0f798fd5478903fc46718eb403a1f`. HER and Main comparator curves are copied from the verified strict context-eligible count-matched artifact and were not rerun for this comparison.

Verify from this directory:

    python artifact_tools.py verify --artifact-dir .

Checkpoints, complete training-row metrics, JSONL files, and logs are excluded.

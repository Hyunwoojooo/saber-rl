# DrawerClose DDPG SABER Actor-Only, 2M

Compact paper artifact for paired seeds 1–5. Actor-Only keeps SABER’s current-policy support query and normalized per-sample weights, applies those weights only to the actor objective, and trains the critic with the standard unweighted HER Bellman loss.

AUC is the normalized trapezoidal area over the exact observed 10k–2M success grid. No smoothing, interpolation, forward filling, or extrapolation is used.

| Method | AUC mean ± sample SD | Final mean ± sample SD | Hit90 median (coverage) |
|---|---:|---:|---:|
| DrawerClose DDPG HER | 0.251005 ± 0.029646 | 0.160000 ± 0.181659 | 70k (5/5) |
| DrawerClose DDPG HER + SABER Main | 0.324422 ± 0.082453 | 0.280000 ± 0.251496 | 20k (5/5) |
| DrawerClose DDPG HER + SABER Actor-Only | 0.307085 ± 0.038837 | 0.280000 ± 0.299166 | 40k (5/5) |

Main − Actor-Only AUC difference: 0.017337; exhaustive paired-bootstrap 95% CI [-0.025829, 0.078970]; Main is higher in 3/5 seeds.

This comparison tests optimization scope: Main applies support-derived weights only to critic Bellman losses, whereas Actor-Only applies the same weighting rule only to the actor. Scope-specific interpretation should use the paired interval together with failure and Hit90 coverage.

Training source commit: `77b5e56eb0712497b55efac6dd0cfd6d841b2670`. HER and Main comparator curves are copied from the verified strict context-eligible count-matched artifact and were not rerun for this comparison.

Verify from this directory:

    python artifact_tools.py verify --artifact-dir .

Checkpoints, complete training-row metrics, JSONL files, and logs are excluded.

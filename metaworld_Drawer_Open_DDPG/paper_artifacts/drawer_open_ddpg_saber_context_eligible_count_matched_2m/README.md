# DDPG-HER-B3-Strict-Context-Eligible-Count-Matched-2M

Compact, independently verifiable artifact for seeds 1–5. Primary AUC uses only the
observed 10k–2M grid, trapezoidal integration, and no smoothing, interpolation,
forward-fill, or extrapolation.

| Method | AUC mean ± sample SD | Final mean ± sample SD |
|---|---:|---:|
| DrawerOpen DDPG HER | 0.207312 ± 0.286818 | 0.390000 ± 0.534322 |
| DrawerOpen DDPG HER + SABER Main | 0.753668 ± 0.044374 | 1.000000 ± 0.000000 |
| DrawerOpen DDPG Strict Context-Eligible Count-Matched | 0.605075 ± 0.344941 | 0.800000 ± 0.447214 |

Primary paired contrast (SABER Main − Strict Control): 0.148593, exhaustive paired-bootstrap 95% CI [-0.041709, 0.442010], exact two-sided sign-flip p=0.437500.

SABER Main exceeded the strict control on mean paired AUC. This is evidence that exact action-bin membership may contribute beyond context eligibility and positive count; five seeds do not establish a universal causal effect.

Frozen source commit: `881b5c1567bc8a979f42eac22695dcb092722420`.

Verify from this directory with:

```bash
python verify_artifact.py --artifact-dir .
```

Checkpoints, full raw metrics, JSONL, logs, correctness, smoke, and pilot outputs
are intentionally excluded.

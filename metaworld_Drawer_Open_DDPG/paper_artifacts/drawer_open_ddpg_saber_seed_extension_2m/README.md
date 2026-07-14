# DrawerOpen DDPG SABER seed-extension artifact

Blind artifact ID: `DRAWEROPEN-DDPG-SABER-SEED-EXTENSION-2M`

This package separates the original seeds 1--5 result from an independent diagnostic extension on seeds 6--10. It also reports a descriptive pooled seeds 1--10 summary. The extension does not add matched HER or control seeds, so it must not be used to construct a paired 10-seed method comparison.

## Recomputed performance

| Cohort | Final success | Success AUC | Hit90 |
| --- | ---: | ---: | ---: |
| Original seeds 1--5 | 1.000 +/- 0.000 | 0.754 +/- 0.044 | 5/5 |
| Extension seeds 6--10 | 0.790 +/- 0.442 | 0.550 +/- 0.329 | 4/5 |
| Pooled seeds 1--10 | 0.895 +/- 0.315 | 0.652 +/- 0.246 | 9/10 |

All performance rows use the exact observed 10k--2M grid with 200 evaluations per seed. The original paired five-seed comparisons in the paper remain the confirmatory method comparisons.

## Diagnostic scope

Seeds 6--10 add a no-gradient post-update TD-error pass. It does not change replay sampling, loss weights, or optimizer steps. Diagnostic summaries use the training row at each exact 10k checkpoint. The added pass and a Python rather than Numba support scanner make runtime incomparable with seeds 1--5; runtime is therefore not pooled or used for efficiency claims.

The diagnostic is observational. It can describe TD-error enrichment and immediate residual correction but cannot establish lower-noise targets or a causal mechanism.

## Verification

```bash
python3 code/verify_artifact.py
```

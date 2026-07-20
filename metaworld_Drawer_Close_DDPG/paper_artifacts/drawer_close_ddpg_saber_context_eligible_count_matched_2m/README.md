# DDPG-HER-B3-Strict-Context-Eligible-Count-Matched-2M

Compact, independently verifiable artifact for seeds 1–5. Primary AUC uses only the
observed 10k–2M grid, trapezoidal integration, and no smoothing, interpolation,
forward-fill, or extrapolation.

| Method | AUC mean ± sample SD | Final mean ± sample SD |
|---|---:|---:|
| DrawerClose DDPG HER | 0.251005 ± 0.029646 | 0.160000 ± 0.181659 |
| DrawerClose DDPG HER + SABER Main | 0.324422 ± 0.082453 | 0.280000 ± 0.251496 |
| DrawerClose DDPG Strict Context-Eligible Count-Matched | 0.294799 ± 0.063108 | 0.330000 ± 0.460435 |

Primary paired contrast (SABER Main − Strict Control): 0.029623, exhaustive paired-bootstrap 95% CI [-0.033191, 0.092437], exact two-sided sign-flip p=0.437500.

SABER Main exceeded the strict control on mean paired AUC. This is evidence that exact action-bin membership may contribute beyond context eligibility and positive count; five seeds do not establish a universal causal effect.

Frozen source commit: `de13e3a0a14d3dc6984d3c592452517f3e32c109`.

Verify from this directory with:

```bash
python verify_artifact.py --artifact-dir .
```

Checkpoints, full raw metrics, JSONL, logs, correctness, smoke, and pilot outputs
are intentionally excluded.

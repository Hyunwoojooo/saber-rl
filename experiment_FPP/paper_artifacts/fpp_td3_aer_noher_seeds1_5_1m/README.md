# FPP TD3 no-HER AER three-arm results, seeds 1-5, 1M steps

This self-contained paper artifact packages the completed
`FetchPickAndPlace-v4` sparse-reward comparison with HER disabled:

- `td3_er`: uniform replay and fixed learning rates;
- `td3_annealing_er`: uniform replay with the AER learning-rate schedule;
- `td3_aer`: cosine attention replay with the same learning-rate schedule.

All three arms contain seeds 1-5 at 1,000,000 environment steps. Every run
passed its completion gate and has exactly 100 deterministic online evaluation
points at 10k, 20k, ..., 1M plus a fixed 100-episode final evaluation.

## Primary result

The endpoint is normalized trapezoidal `eval/success_mean` AUC on the exact
10k-1M grid. No interpolation, smoothing, forward filling, or best-checkpoint
selection is used.

| Method | n | AUC mean | AUC SD | Final mean | Best mean |
|---|---:|---:|---:|---:|---:|
| `td3_er` | 5 | 0.039343 | 0.001918 | 0.000000 | 0.170000 |
| `td3_annealing_er` | 5 | 0.039394 | 0.001847 | 0.000000 | 0.170000 |
| `td3_aer` | 5 | 0.039293 | 0.001969 | 0.000000 | 0.170000 |

| Paired comparison | Mean AUC difference | 95% paired bootstrap CI | Exact sign-flip p |
|---|---:|---:|---:|
| AER - AnnealingER | -0.000101 | [-0.000455, 0.000253] | 0.75 |
| AER - ER | -0.000051 | [-0.000152, 0.000000] | 1.00 |

These runs are at the sparse-exploration floor; the data do not show an AER
advantage over either control.

## Contents

- `arms/td3_{er,annealing_er,aer}/`: independently verifiable five-seed arm
  artifacts with configuration, environment snapshot, summary, best-checkpoint
  metadata, fixed final evaluation, and the exact selected 10k metrics grid.
- `analysis/`: paired per-seed, aggregate, mean-curve, bootstrap, and exact
  sign-flip outputs.
- `code/analyze_td3_aer_noher.py`: frozen standard-library analysis.
- `code/verify_bundle.py`: verifies every file hash, all three nested arm
  artifacts, and reproduces the numerical analysis from the packaged data.

The AER and AnnealingER selected-grid files retain schedule and AER diagnostics
including lambda, candidate/selected counts, similarities, duplicate rates,
transition ages, success fractions, phase fractions, effective learning rates,
and sampling time. Full-resolution metrics, event streams, logs, and model
checkpoints are intentionally excluded because the raw result root is 5.6 GB
and individual files exceed GitHub's 100 MB limit.

Result-producing source commit:
`8c53c789f6f5cb6a45beceb4462a209cf6100d7e`.

## Verification

From this directory:

```bash
python code/verify_bundle.py .
```

`SHA256SUMS` covers every artifact file except itself.

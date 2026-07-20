# FPP SAC no-HER AER three-arm results

This compact artifact contains the completed paired seeds 1-5 runs for
uniform ER, uniform ER with the AER learning-rate schedule, and attentive
experience replay with the same schedule.

| Method | Success AUC mean +/- SD | 20-episode final mean +/- SD | Fixed final-100 mean +/- SD |
|---|---:|---:|---:|
| SAC ER | 0.063990 +/- 0.042061 | 0.130000 +/- 0.198746 | 0.164000 +/- 0.232551 |
| SAC Annealing ER | 0.040808 +/- 0.002497 | 0.000000 +/- 0.000000 | 0.032000 +/- 0.008367 |
| SAC AER | 0.040000 +/- 0.002910 | 0.000000 +/- 0.000000 | 0.030000 +/- 0.007071 |

The primary paired comparison is `sac_aer - sac_annealing_er`;
`sac_aer - sac_er` is the secondary full-recipe comparison.

## Contents

- `data/`: sanitized per-run config, environment, summary, best-checkpoint
  metadata, fixed 100-episode evaluation, and exact 100-point metrics.
- `results/`: canonical paired analyzer outputs, fixed-final table, and
  machine-readable validation evidence.
- `provenance/`: the queue-frozen source manifest and the verified hashes of
  the exact source files used by the completed run.
- `input_manifest.json` and `artifact_manifest.json`: SHA256 inventories.

Online curve evaluations use 20 deterministic episodes at 10k, 20k, ...,
1M steps. The success AUC is trapezoidal integration divided by 990k. The
fixed final evaluation uses the final policy for 100 deterministic episodes
and is reported separately from the final point on the learning curve.

Raw model checkpoints, replay state, full metrics, JSONL streams, and logs
are intentionally excluded. They were checked by the strict completion gate
before this artifact was built.

Regenerate from the repository root with:

```bash
.venv/bin/python experiment_FPP/package_sac_aer_results.py
```

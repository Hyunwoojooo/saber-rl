# FPP DDPG no-HER AER three-arm results

This compact artifact contains the completed paired seeds 1-5 runs for
uniform ER, uniform ER with the AER learning-rate schedule, and attentive
experience replay with the same schedule.

| Method | Success AUC mean +/- SD | 20-episode final mean +/- SD | Fixed final-100 mean +/- SD |
|---|---:|---:|---:|
| DDPG ER | 0.147677 +/- 0.039279 | 0.460000 +/- 0.188414 | 0.440000 +/- 0.167631 |
| DDPG Annealing ER | 0.128586 +/- 0.111223 | 0.340000 +/- 0.362974 | 0.328000 +/- 0.302523 |
| DDPG AER | 0.044444 +/- 0.002020 | 0.000000 +/- 0.000000 | 0.040000 +/- 0.015811 |

The primary paired comparison is `ddpg_aer - ddpg_annealing_er`;
`ddpg_aer - ddpg_er` is the secondary full-recipe comparison.

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
.venv/bin/python experiment_FPP/package_ddpg_aer_results.py
```

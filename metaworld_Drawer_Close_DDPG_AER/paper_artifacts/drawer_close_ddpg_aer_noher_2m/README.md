# DrawerClose DDPG AER No-HER 2M Artifact

This artifact preserves the completed three-arm DrawerClose DDPG no-HER study:

- 2,000,000 environment steps per run
- seeds 1-5
- exact 10k evaluation grid with 20 deterministic episodes per checkpoint
- a separate fixed 100-episode evaluation after training
- DDPG-ER, DDPG-AnnealingER, and DDPG-AER

All three arms use the same single-critic DDPG backbone and set the HER ratio
to zero. DDPG-ER uses uniform replay and fixed learning rates.
DDPG-AnnealingER uses uniform replay with the AER learning-rate schedule.
DDPG-AER adds cosine top-k attentive replay sampling under the same schedule.

## Aggregate Results

Values are the mean and sample standard deviation over five seeds. Success AUC
is normalized over the exact 10k-2M grid.

| Arm | Success AUC | Final success | Best success | Hit90 | Fixed 100-episode success |
| --- | ---: | ---: | ---: | ---: | ---: |
| DDPG-ER | 0.150 +/- 0.054 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 5/5, median 280k | 0.000 +/- 0.000 |
| DDPG-AnnealingER | 0.199 +/- 0.122 | 0.330 +/- 0.391 | 1.000 +/- 0.000 | 5/5, median 380k | 0.334 +/- 0.391 |
| DDPG-AER | 0.194 +/- 0.067 | 0.200 +/- 0.447 | 1.000 +/- 0.000 | 5/5, median 160k | 0.200 +/- 0.447 |

All 15 runs reach Hit90 at least once, but most policies collapse before the
2M endpoint. AER reaches Hit90 earliest in median environment steps. Its mean
AUC is not higher than AnnealingER, however, so attentive sampling does not
show an incremental average-AUC benefit over its learning-rate control in this
cohort.

The paired AUC differences are:

- AER minus ER: +0.0439, exact paired-bootstrap 95% interval
  [-0.0267, 0.1086], with 4/5 positive pairs.
- AER minus AnnealingER: -0.0057, interval [-0.1427, 0.1150], with 3/5
  positive pairs.
- AnnealingER minus ER: +0.0496, interval [-0.0514, 0.1367], with 4/5
  positive pairs.

All three intervals include zero. These results describe a matched transfer of
the standalone no-HER AER protocol, not a universal claim about AER. A
comparison against HER+SABER is an end-to-end algorithm comparison because
the relabeling pipelines differ.

The source config retains the `method=b0_ddpg_er` tag used by the shared
trainer to select unit critic-loss weights. The `replay_method` field
identifies each arm. The executed agent is DDPG with one critic, one actor
update per critic update, and no TD3 target-policy smoothing.

## Files

- `evaluation_curves_10k.csv`: all 3,000 exact-grid evaluation points.
- `per_seed_results.csv`: recomputed final, best, AUC, last-quarter AUC,
  Hit90, fixed evaluation, speed, and runtime for each run.
- `aggregate_results.csv`: five-seed means and sample standard deviations.
- `paired_comparisons.csv`: exact paired bootstrap and sign-flip diagnostics.
- `run_records/`: sanitized config, environment, summary, fixed evaluation,
  and selected 10k metrics for all 15 runs.
- `source_provenance.json`: source paths, sizes, and SHA256 hashes.
- `artifact_manifest.json`: SHA256 manifest for the committed artifact.

Checkpoints, full training `metrics.csv`, `metrics.jsonl`, and `run.log`
files are intentionally excluded from GitHub.

## Rebuild and Verify

From the repository root, with the original ignored run outputs available:

```bash
.venv/bin/python metaworld_Drawer_Close_DDPG_AER/paper_artifacts/drawer_close_ddpg_aer_noher_2m/code/build_artifact.py
```

The committed artifact can be verified without the raw run outputs:

```bash
.venv/bin/python metaworld_Drawer_Close_DDPG_AER/paper_artifacts/drawer_close_ddpg_aer_noher_2m/code/verify_artifact.py
```

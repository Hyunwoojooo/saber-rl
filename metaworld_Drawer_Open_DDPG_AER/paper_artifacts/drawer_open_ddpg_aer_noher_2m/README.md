# DrawerOpen DDPG AER No-HER 2M Artifact

This artifact preserves the completed three-arm DrawerOpen DDPG no-HER study:

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
| DDPG-ER | 0.579 +/- 0.342 | 0.800 +/- 0.447 | 0.810 +/- 0.425 | 4/5, median 525k | 0.800 +/- 0.447 |
| DDPG-AnnealingER | 0.111 +/- 0.248 | 0.200 +/- 0.447 | 0.200 +/- 0.447 | 1/5, 910k | 0.200 +/- 0.447 |
| DDPG-AER | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0/5 | 0.000 +/- 0.000 |

DDPG-ER reaches Hit90 on seeds 2-5 at 280k, 800k, 500k, and 550k,
respectively. DDPG-AnnealingER reaches Hit90 only on seed 3 at 910k, while
DDPG-AER remains at zero evaluation success on all five seeds. The mean paired
AER-minus-ER AUC difference is -0.5794 with an exact paired-bootstrap interval
of [-0.7899, -0.2842]. The exact two-sided sign-flip diagnostic is 0.0625.

The fixed-LR ER result shows that this no-HER DrawerOpen DDPG setting is not an
absolute exploration floor. The AER learning-rate schedule is strongly
associated with the degradation because AnnealingER also underperforms ER.
However, these three arms do not independently identify attentive sampling as
the cause: the AER-minus-AnnealingER interval [-0.3332, 0.0000] includes zero,
and a fixed-LR AER arm would be required for a clean sampling-only attribution.
This matched transfer is not a claim that AER fails universally.

The source config retains the `method=b0_ddpg_er` tag used by the shared
trainer to select unit loss weights. The executed agent is DDPG with one
critic, one actor update per critic update, and no TD3 target-policy smoothing.
The `replay_method` field identifies each experimental arm.

## Files

- `evaluation_curves_10k.csv`: all 3,000 exact-grid evaluation points.
- `per_seed_results.csv`: recomputed final, best, AUC, last-quarter AUC, Hit90,
  fixed evaluation, speed, and runtime for each run.
- `aggregate_results.csv`: five-seed means and sample standard deviations.
- `paired_comparisons.csv`: exact paired bootstrap and sign-flip diagnostics.
- `run_records/`: sanitized config, environment, summary, fixed evaluation, and
  selected 10k metrics for all 15 runs.
- `source_provenance.json`: source paths, sizes, and SHA256 hashes.
- `artifact_manifest.json`: SHA256 manifest for the committed artifact.

Checkpoints, full training `metrics.csv`, `metrics.jsonl`, and `run.log` files
are intentionally excluded from GitHub.

## Rebuild and Verify

From the repository root, with the original ignored run outputs available:

```bash
.venv/bin/python metaworld_Drawer_Open_DDPG_AER/paper_artifacts/drawer_open_ddpg_aer_noher_2m/code/build_artifact.py
```

The committed artifact can be verified without the raw run outputs:

```bash
.venv/bin/python metaworld_Drawer_Open_DDPG_AER/paper_artifacts/drawer_open_ddpg_aer_noher_2m/code/verify_artifact.py
```

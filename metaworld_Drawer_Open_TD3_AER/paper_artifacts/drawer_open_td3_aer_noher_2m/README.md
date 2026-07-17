# DrawerOpen TD3 AER No-HER 2M Artifact

This artifact preserves the completed three-arm DrawerOpen TD3 no-HER study:

- 2,000,000 environment steps per run
- seeds 1-5
- exact 10k evaluation grid with 20 deterministic episodes per checkpoint
- a separate fixed 100-episode evaluation after training
- TD3-ER, TD3-AnnealingER, and TD3-AER

All three arms use the same TD3 backbone and set the HER ratio to zero. TD3-ER
uses uniform replay and fixed learning rates. TD3-AnnealingER uses uniform replay
with the AER learning-rate schedule. TD3-AER adds cosine top-k attentive replay
sampling under the same schedule.

## Aggregate Results

Values are the mean and sample standard deviation over five seeds. Success AUC
is normalized over the exact 10k-2M grid.

| Arm | Success AUC | Final success | Best success | Hit90 | Fixed 100-episode success |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3-ER | 0.089 +/- 0.122 | 0.400 +/- 0.548 | 0.420 +/- 0.531 | 2/5, median 1.53M | 0.400 +/- 0.548 |
| TD3-AnnealingER | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0/5 | 0.000 +/- 0.000 |
| TD3-AER | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0/5 | 0.000 +/- 0.000 |

TD3-ER succeeds late on seeds 4 and 5, reaching Hit90 at 1.60M and 1.46M,
respectively. TD3-AER and TD3-AnnealingER remain at zero evaluation success on
all five seeds. The mean paired AER-minus-ER AUC difference is -0.0889 with an
exact paired-bootstrap interval of [-0.1808, -0.0002]. The exact two-sided
sign-flip diagnostic is 0.25 because only three pairs are nonzero.

The result supports a narrow claim: standalone AER does not overcome the
sparse-exploration floor in this matched DrawerOpen TD3 transfer. It is not a
claim that AER fails universally. Because AER and AnnealingER both remain at
the floor, these three arms do not independently identify attentive sampling
as the cause of failure; an AER-FixedLR arm would be required for that claim.

The source config retains the legacy `method=b0_ddpg_her` tag used by the
shared trainer to select unit loss weights. The executed agent is TD3: it uses
twin critics, target-policy smoothing 0.2/0.5, and policy delay 2. The
`replay_method` field identifies each experimental arm.

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
.venv/bin/python metaworld_Drawer_Open_TD3_AER/paper_artifacts/drawer_open_td3_aer_noher_2m/code/build_artifact.py
```

The committed artifact can be verified without the raw run outputs:

```bash
.venv/bin/python metaworld_Drawer_Open_TD3_AER/paper_artifacts/drawer_open_td3_aer_noher_2m/code/verify_artifact.py
```

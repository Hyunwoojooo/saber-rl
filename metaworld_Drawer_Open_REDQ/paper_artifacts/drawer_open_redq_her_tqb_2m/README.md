# DrawerOpen REDQ-HER TQB 2M artifact

Blind artifact ID: `DRAWEROPEN-REDQ-HER-TQB-2M`

This package contains five completed 2M-step DrawerOpen REDQ+HER runs with a 10-critic ensemble, target subset size 2, UTD 4, automatic entropy tuning, and target-Q bounding `[-50, 0]`. It is an architecture/compute-scaled reference, not a matched one-update-per-step SABER baseline.

## Recomputed result

- Final success: `1.000 +/- 0.000`
- Success AUC: `0.672 +/- 0.314`
- Hit90: `5/5`, median `370k` among reached seeds
- Mean recorded runtime: `111.05` hours per seed
- Evaluation grid: exact 10k through 2M, 200 observed points per seed

The mean Hit90 over seeds is 644k, but the paper reports the reached-seed median 370k to remain consistent with its other threshold tables.

## Scope

The raw config retains the legacy method label `b0_ddpg_her`. REDQ identity is verified from the packaged trainer, 10-critic ensemble, target-subset backup, UTD 4, auto-alpha, and target-Q-bound configuration. Large checkpoints, raw high-frequency metrics, and logs are excluded. Exact-grid evaluation rows, summaries, configs, environment snapshots, and the source snapshot are included.

## Verification

```bash
python3 code/verify_artifact.py
```

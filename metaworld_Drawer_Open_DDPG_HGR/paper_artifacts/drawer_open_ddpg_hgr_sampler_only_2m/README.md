# DrawerOpen DDPG HGR-style sampler-only 2M artifact

Blind artifact ID: `DRAWEROPEN-DDPG-HGR-SAMPLER-ONLY-2M`

This package contains five completed 2M-step DrawerOpen DDPG+HER runs. The method changes episode and future-goal sampling using TD-error ranking while leaving critic and actor loss weights equal to one. It omits canonical HGR importance correction and must be cited as an HGR-style sampler-only ablation.

## Recomputed result

- Final success: `0.400 +/- 0.548`
- Success AUC: `0.311 +/- 0.425`
- Hit90: `2/5`, median `450k` among reached seeds
- Evaluation grid: exact 10k through 2M, 200 observed points per seed

## Scope

This result is a sampling stress control, not a faithful or canonical HGR comparison. Large checkpoints, raw high-frequency metrics, and logs are excluded. Seed summaries, configs, environment snapshots, and exact-grid evaluation curves are included.

## Verification

```bash
python3 code/verify_artifact.py
```

# MetaWorld Drawer Close DDPG+HER HGR-Style Sampler-Only 2M Results

This compact artifact records the completed five-seed MetaWorld Drawer Close
DDPG+HER sampling ablation. Every seed reached exactly 2,000,000 environment
steps.

## Method boundary

The row changes episode and future-goal sampling using TD-error priorities but
keeps critic and actor loss weights equal to one. It omits canonical HGR's
episode/goal importance-sampling correction and uses
`hgr_alpha_episode=hgr_alpha_goal=1.0`. It must therefore be reported as
**HGR-style sampler-only**, not as canonical HGR or a faithful direct HGR
baseline.

## Included results

Each seed directory contains:

- "config.json"
- "env_snapshot.json"
- "metrics_selected_10k.csv"
- "summary.json"

Absolute repository prefixes in JSON path fields were converted to
repository-relative paths. CSV line endings were normalized to LF. Numerical
values were not changed. Original paths and SHA256 values are recorded in
"artifact_manifest.json".

| Seed | Final step | Native eval AUC | Final eval success |
|---:|---:|---:|---:|
| 1 | 2,000,000 | 0.275000000 | 0.300 |
| 2 | 2,000,000 | 0.342281879 | 1.000 |
| 3 | 2,000,000 | 0.256543624 | 0.000 |
| 4 | 2,000,000 | 0.194630872 | 0.600 |
| 5 | 2,000,000 | 0.242281879 | 0.750 |

## Source

- Launcher: "metaworld_Drawer_Close_DDPG_HGR/DDPG-HER-HGR-2M.sh"
- HGR implementation: "metaworld_Drawer_Close_DDPG_HGR/code/src/metacog_lite_v0/hgr.py"
- Trainer: "metaworld_Drawer_Close_DDPG_HGR/code/src/train_drawer_close_lite_v0.py"

The launcher and HGR implementation used by these runs are included in the
same repository commit as this artifact; their SHA256 values are recorded in
the manifest.

## Excluded files

Full "metrics.csv", "metrics.jsonl", run logs, "checkpoint.pt",
"best_checkpoint.pt", and checkpoint metadata are omitted. They are not needed
to audit the selected 10k evaluation curves and summaries, and the model files
are larger than GitHub's per-file limit.

# DDPG-HER-B3-Replay-Action-Query-2M

Compact, independently verifiable artifact for DrawerOpen seeds 1–5. Exact success AUC uses only observed 10k–2M points, with no smoothing, interpolation, forward-fill, or extrapolation.

Replay-action query does not remove the sampled transition from the local replay pool. Consequently the replay action can self-support when that transition is present in the bounded local scan. This is the same intended replay-query definition used by the FPP query ablation.

Only the SABER membership query changes. Actor objective/update and the Bellman target are unchanged. Bellman regression remains at the replay action, and weights remain critic-only.

| Method | AUC mean ± sample SD | Final mean ± sample SD |
|---|---:|---:|
| DrawerOpen DDPG HER | 0.207312 ± 0.286818 | 0.390000 ± 0.534322 |
| DrawerOpen DDPG HER + SABER Main | 0.753668 ± 0.044374 | 1.000000 ± 0.000000 |
| DrawerOpen DDPG Replay-Action Query | 0.317789 ± 0.436570 | 0.400000 ± 0.547723 |

Primary Main − Replay AUC difference: 0.435879; exhaustive paired-bootstrap 95% CI [0.120503, 0.729146]; exact two-sided sign-flip p=0.125000.

All five seeds and paired statistics are reported regardless of direction. If the CI includes zero, this artifact does not claim that an independent current-policy-query effect is established.

HER/Main were reused from verified artifact commit 93ae5c03633726f8e7783a157cac8516348f088f (packaging artifact source commit 881b5c1567bc8a979f42eac22695dcb092722420; original training source commit unknown/not recorded) and were not retrained. Original comparator config/env files were absent, so HER/Main config records are explicitly reconstructed metadata and env records are explicitly same-seed Strict proxies, never original snapshots.

Frozen replay source commit: babe7eddc93c4bfb6f2ba870f3db7a35625aed2f.

Verify from this directory with:

    python verify_artifact.py --artifact-dir .

Checkpoints, full raw metrics, JSONL, logs, smoke/correctness/pilot outputs, caches, and temporary files are excluded.

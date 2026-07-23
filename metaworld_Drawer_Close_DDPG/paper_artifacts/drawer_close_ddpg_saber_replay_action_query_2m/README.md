# DDPG-HER-B3-Replay-Action-Query-2M

Compact, independently verifiable artifact for DrawerClose seeds 1–5. Exact success AUC uses only observed 10k–2M points, with no smoothing, interpolation, forward-fill, or extrapolation.

Primary success field: `eval/env_success_mean` only; no fallback field is used.

Replay-action query does not remove the sampled transition from the local replay pool. Consequently the replay action can self-support when that transition is present in the bounded local scan. This matches the intended FPP replay-query definition.

Only the SABER membership query changes. Actor objective/update and the Bellman target are unchanged. Bellman regression remains at the replay action, and weights remain critic-only.

| Method | AUC mean ± sample SD | Final mean ± sample SD |
|---|---:|---:|
| DrawerClose DDPG HER | 0.251005 ± 0.029646 | 0.160000 ± 0.181659 |
| DrawerClose DDPG HER + SABER Main | 0.324422 ± 0.082453 | 0.280000 ± 0.251496 |
| DrawerClose DDPG Replay-Action Query | 0.275754 ± 0.016370 | 0.290000 ± 0.191703 |

Primary Main − Replay AUC difference: 0.048668; exhaustive paired-bootstrap 95% CI [-0.005553, 0.128392]; exact two-sided sign-flip p=0.312500.

All five seeds and paired statistics are reported regardless of direction. If the CI includes zero, this artifact does not claim that an independent current-policy-query effect is established.

HER/Main were reused, not retrained, from verified artifact commit `841eb706b1dcdc12b59db23ec545099d39d3c619` (packaging source `de13e3a0a14d3dc6984d3c592452517f3e32c109`). Their original training source commit and original config/env snapshots were not recorded; reconstructed metadata and explicitly labelled same-seed Strict environment proxies are used.

The legacy Replay run did not serialize `support_assignment_mode`. The artifact therefore records the serialized value as `null` and the source-derived effective mode as `support`: the canonical membership mask is directly applied without randomized assignment.

Checkpoint available-field audits passed for all seeds. Full checkpoint/resume reproducibility is explicitly `not_empirically_audited`; this artifact makes no stronger claim.

Frozen Replay execution source commit: `cb871323fed51c6c7d5f692d189ebd766f282503`.

Verify from this directory with:

    python verify_artifact.py --artifact-dir .

Checkpoints, best checkpoints, full raw metrics, JSONL, logs, smoke/correctness/pilot outputs, caches, and temporary files are excluded.

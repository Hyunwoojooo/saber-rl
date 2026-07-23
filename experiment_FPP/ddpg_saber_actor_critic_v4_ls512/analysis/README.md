# Actor+Critic SABER scope-ablation analysis

This directory contains standard-library-only analysis and paper-artifact tools.

- `analyze_results.py`: requires paired seeds 1–5 and computes exact observed-grid
  success AUC, final/best/Hit90, mean ± sample SD, exact-grid mean learning
  curves, and paired inference.
- `build_artifact.py`: builds
  `experiment_FPP/paper_artifacts/fpp_ddpg_saber_actor_critic_v4_ls512_1m/`.
- `verify_artifact.py`: recomputes the analysis and verifies `validation.json`,
  frozen executable `source/`, launch-time execution metadata, fixed config and
  environment contracts, config parity, file sets, and both SHA256 manifests.
- `test_scope_ablation.py`: synthetic paired fixtures, missing-grid/pair failures,
  config-parity checks, end-to-end packaging, and tamper detection.

Canonical raw roots:

```text
experiment_FPP/runs/ddpg_saber_actor_critic_v4_ls512
experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512
```

The analysis is fail-closed. Every requested seed must have both arms, a complete
1M/995001-update summary, and all 100 observed 10k-grid evaluations. It never
interpolates, smooths, or forward-fills.

Artifact construction additionally requires
`experiment_FPP/runs/ddpg_saber_actor_critic_v4_ls512/_execution/launch_metadata.json`.
Its `repo_head` and `origin_main` must both equal the supplied source commit, and
its clean-worktree and head-match flags must both be true. Executable smoke
launchers may be frozen under `source/`; smoke result payloads under `data/` are
prohibited.

Run tests:

```bash
python3 -m unittest discover -s experiment_FPP/ddpg_saber_actor_critic_v4_ls512/analysis -p 'test_*.py' -v
```

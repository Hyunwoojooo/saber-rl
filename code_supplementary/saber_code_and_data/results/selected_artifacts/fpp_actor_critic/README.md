# FPP DDPG+HER+SABER Actor+Critic V4 LS512 (1M)

Compact paired seeds 1–5 artifact. Source commit: `2d52b29666233616c7b1e2029aa8090e7d74bd66`.
Canonical critic source commit: `c43d40502b84ae713924ebf995be623e7acc7ca8`.

The recorded `thread_hint` differs between arms: the actor+critic launcher fixed
MKL, OpenMP, and OpenBLAS threads to one, while the earlier canonical critic run
left those variables unset. This is runtime metadata, not a trainer setting.

Non-finite optional diagnostics in raw summaries are represented as JSON `null`
in the compact copies. `source_inputs_manifest.json` retains each raw source
file's SHA256 and size, so this standards-compliance normalization is auditable.

The primary endpoint is normalized trapezoidal success AUC on the exact observed
10k–1M grid (100 points, denominator 990k). There is no interpolation, smoothing,
or forward-fill. `analysis/mean_learning_curve.csv` contains only exact-grid
per-arm means and sample SDs. `execution_metadata.json` binds the actor results
to a clean `origin/main` source commit. `source/` is the frozen executable
experiment package; `code/` is the reproducible analysis/build/verification
tooling. Raw run metrics, logs, checkpoints, and smoke result payloads are excluded.

Verify with `python3 code/verify_artifact.py --artifact-root .`.

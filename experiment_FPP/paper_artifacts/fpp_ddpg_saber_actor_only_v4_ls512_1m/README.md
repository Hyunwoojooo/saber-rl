# FPP DDPG+HER+SABER Actor-Only V4 LS512 (1M)

Compact paired seeds 1–5 artifact. Source commit: `e062064a5b08783c7d4147eaf072198a7874f09b`.
Canonical critic source commit: `c43d40502b84ae713924ebf995be623e7acc7ca8`.

The primary endpoint is normalized trapezoidal success AUC on the exact observed
10k–1M grid (100 points, denominator 990k). There is no interpolation, smoothing,
or forward-fill. `analysis/mean_learning_curve.csv` contains only exact-grid
per-arm means and sample SDs. `execution_metadata.json` binds the actor results
to a clean `origin/main` source commit. `source/` is the frozen executable
experiment package; `code/` is the reproducible analysis/build/verification
tooling. Raw run metrics, logs, checkpoints, and smoke result payloads are excluded.
Undefined non-core summary diagnostics are normalized from non-standard
NaN/Infinity values to JSON `null`; primary performance metrics remain subject to
strict finite-value validation.

Verify with `python3 -B code/verify_artifact.py --artifact-root .` so Python does
not add bytecode-cache files to the manifest-locked artifact directory.

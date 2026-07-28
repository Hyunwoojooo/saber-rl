# DrawerOpen DDPG SABER Fixed-Target Diagnostic (2M)

This experiment reruns only the DrawerOpen DDPG Main SABER diagnostic cohort.
It does not rerun HER, HGR, PER, TD-error weighting, FPP, or DrawerClose.
The source snapshot is based on repository commit `ec05735`; only the agent
and trainer are behaviorally changed for fixed-target instrumentation.

## Diagnostic correction

For every persisted diagnostic minibatch, the implementation computes the
pre-update Bellman target `y_pre`, performs the normal critic update, and then
reports the primary residual as `abs(Q_after(s, a_replay) - y_pre)`. The same
minibatch is also evaluated with the post-update target network and recorded
under `diagnostic/live_target_*` for audit purposes. The generic
`diagnostic/residual_*` fields always refer to the fixed pre-update target.

The diagnostic forward passes run only when a row will be persisted (episode
end or evaluation step). Earlier code recomputed post-update residuals at every
training update even though intermediate values were overwritten. This change
preserves the recorded minibatch cadence while avoiding discarded diagnostic
forwards. Training remains 2M environment steps, UTD 1, batch size 256, future
HER 0.8, and Main SABER critic-only weighting.

Resume is intentionally disabled because MetaWorld simulator state is not part
of the checkpoint. Each seed must run uninterrupted from step 0.

Install the exact scanner runtime before launching:

```bash
PYTHON="${PYTHON:-../../../.venv/bin/python}"
"$PYTHON" -m pip install -r requirements-runtime.txt
```

The launcher also checks that CUDA is available and that NumPy 2.3.5, Numba
0.66.0, and llvmlite 0.48.0 are active. It aborts instead of silently falling
back to the slower Python support scanner.

Set `PYTHON=/absolute/path/to/python` when the environment is outside the
repository-level `.venv`.

## Two-machine launch

Machine A (seeds 1-5):

```bash
SEEDS="1 2 3 4 5" CUDA_DEVICE=0 ./run_5seeds_parallel.sh
```

Machine B (seeds 6-10):

```bash
SEEDS="6 7 8 9 10" CUDA_DEVICE=0 ./run_5seeds_parallel.sh
```

The launcher requires exactly five unique seed IDs, validates the environment,
HER, and support paths, enables the pinned Numba scanner, runs all five
processes concurrently, and limits each
process to one CPU math thread to prevent oversubscription. It fails fast if
Numba or CUDA is unavailable.

For a concurrency benchmark before the full run, use a separate output root:

```bash
STEPS=20000 OUTPUT_ROOT=/tmp/draweropen_fixed_target_benchmark \
  SEEDS="1 2 3 4 5" ./run_5seeds_parallel.sh
```

Do not reuse benchmark outputs for the paper cohort.

## Expected provenance fields

Completed summaries must contain:

- `fixed_target_residual_diagnostic: true`
- `live_target_residual_diagnostic: true`
- `diagnostic_snapshot_only: true`
- `resume_supported: false`

Persisted diagnostic rows must contain
`diagnostic/fixed_target_from_pre_update = 1` and
`diagnostic/same_minibatch_one_update = 1`.
After all five processes exit, `validate_completed_runs.py` checks these
provenance fields, the exact top-10% sample count, the complete 200-step
diagnostic grid, and the absence of scanner fallback or audit mismatches.

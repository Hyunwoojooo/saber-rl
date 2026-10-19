# SAC power-interruption recovery

This is a new execution attempt of the original five-seed protocol, not a new
algorithm or a checkpoint continuation. The user authorized recovery on
2026-10-12 after the machine lost power. All original files are read-only inputs.

## Frozen scientific implementation

The top-level Python files, protocol, tests, code and upstream snapshots are
byte-identical copies of `2026-10-08_drawer_sac_fpp_v2_s1_5`. Their source-manifest
digest remains `bc9377ff0628c1048124909a7c32b6a25a0c29fb0e89a5b2f5a856a73f40cede`.
Recovery-specific orchestration is separately frozen in `recovery_manifest.json`.
Use `recovery/manage.py`, not the copied original full-cohort launchers.

## Fixed recovery selection

- Reuse DrawerOpen HER seeds 1-3, including the poorly performing seeds 2 and 3.
- Rerun DrawerOpen SABER seeds 1-4 and HER seed 4 from step zero to 2M.
- Start DrawerOpen HER/SABER seed 5 for the first time.
- Then start all ten DrawerClose HER/SABER runs, seeds 1-5, from zero to 2M.
- Preserve all five interrupted attempts and their partially written logs.
- Never splice old curves into a new run, resume a checkpoint, replace a seed
  according to its performance, or automatically expand to seeds 6-10.

This requires 17 new runs / 34M environment steps. Five workers run in parallel
in the original seed-major, method-major order after removing the three reused
runs. DrawerClose starts only after all ten DrawerOpen results validate.

## Validation and provenance

`prepare` verifies and copies the frozen source, records the original output
hashes and interruption inventory, and checks the current software/hardware
fields against every started parent run. No dependency installation is needed.
CPU validation is rerun, including the original regression tests and smokes.
`gpu-gate` reruns three CUDA reference updates and four fresh 600-step smokes.
The four earlier 100k stability runs are copied as explicitly inherited evidence,
rehashed and revalidated, not represented as new executions. Training remains
fail-closed behind the original CPU/GPU production gates.

The completed HER runs remain at their original locations. The combined analysis
uses a hashed catalog to validate their original configurations and environments
alongside the new attempts. It retains all interrupted-attempt records and the
original exploratory five-seed inference scope and bootstrap specification.

## Commands

```bash
python -B recovery/manage.py prepare
python -B -m unittest discover -s recovery -p 'test_*.py' -v
python -B validate.py
python -B recovery/manage.py gpu-gate
python -B recovery/manage.py verify
python -B recovery/manage.py run --execute
```

`run` without `--execute` only prints the fixed plan. Execution is one-shot;
existing status/claim/output files block a second invocation. The service is
configured with `Restart=no`: an additional interruption requires explicit
inspection and approval rather than silently continuing or replacing a run.

Monitor `recovery_state/status.json`, `recovery_state/*.log`, `.production_claims/`
and `runs/`. Final combined results are written under `analysis/`. All resumed
training is disabled; original tail-corrupted files are left untouched.

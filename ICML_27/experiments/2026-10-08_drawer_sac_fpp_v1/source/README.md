# Drawer SAC using the FPP automatic-temperature implementation

Independent implementation, frozen upstream copies, and validation for the four
requested cohorts. No historical fixed-alpha DrawerOpen results are reused.
No production training is started by setup, dry runs, or validation.

| Task | Method | Seeds | Steps per seed |
| --- | --- | --- | --- |
| DrawerOpen-v3 | SAC + HER | 1-10 | 2,000,000 |
| DrawerOpen-v3 | SAC + HER + SABER, critic-only | 1-10 | 2,000,000 |
| DrawerClose-v3 | SAC + HER | 1-10 | 2,000,000 |
| DrawerClose-v3 | SAC + HER + SABER, critic-only | 1-10 | 2,000,000 |

Total: 40 fresh runs, 80 million environment steps. All paths below are relative
to this directory. Use the existing research Python environment; no installation
or modification of that environment is performed by these scripts.

## Entry points

```bash
python -B protocol.py                  # check all 20 matched pairs
python -B launch.py                    # print 40-run plan; does not train
python -B train.py --task drawer_open --method saber --seed 1  # config only
python -B manifest.py                  # verify frozen source
python -B validate.py                  # sequential CPU tests + four 600-step smokes
```

Production requires both `validation/validation_report.json` and
`validation/gpu_validation_report.json` to pass on the frozen source. The latter
requires FPP-reference CUDA update parity, four 600-step GPU smokes, and four
100,000-step stability runs, with complete logs and reloaded checkpoints. No
success-rate or SABER-over-HER threshold is used. Direct production remains gated:

```bash
python -B launch.py --execute --parallel 5
```

## Approved deferred execution

On 2026-10-08 the user approved validation followed by automatic production,
DrawerOpen first, only after the existing retrieval experiment finishes. The
one-shot queue checks all ten Random512 runs, the paired analysis, the final
retrieval diagnostic, predecessor source hashes, and absence of GPU compute jobs.
It does not import Torch/CUDA while waiting and polls once per minute.

1. Re-run CPU validation on the frozen source.
2. Run CUDA update parity, then four 600-step GPU smokes (up to four parallel).
3. Run four 100,000-step stability checks (up to four parallel, validation seed 1).
4. Run DrawerOpen HER/SABER, seeds 1-10 each, 2M steps each, five parallel.
5. Validate and analyze the complete DrawerOpen cohort.
6. Run DrawerClose with the same budget, then analyze it and both tasks jointly.

```bash
python -B scheduled_run.py --check     # read-only prerequisite and plan report
python -B gpu_validate.py             # GPU plan only; does not initialize CUDA
python -B scheduled_run.py --execute  # arm once, explicitly authorized
```

The deployed user service is `saber-drawer-sac-after-retrieval.service`.
`automation/status.json` records queue state and child PID; `automation/*.log`
records each stage. `systemctl --user status saber-drawer-sac-after-retrieval`
shows service state. Stopping the service terminates its children and blocks the
queue, without automatic retries. A user service with lingering survives logout;
the transient service is not automatically restored after a machine reboot.
After interruption, inspect retained status before explicitly authorizing a new
attempt. Never delete claims to make an existing run appear fresh.

Any failed gate blocks later stages. Source refreezing is prohibited once the
queue is armed. Validation runs are kept separately and never reused as production
seeds. A 100k check does not guarantee numerical/convergence stability over 2M
steps or exercise a filled 300k replay buffer; failures during production remain
reported rather than replaced. No packages or running predecessor files change.

The launcher schedules the complete declared cohort, optionally restricted by
`--tasks`, `--methods`, or `--seeds` for distribution across machines. Keep each
HER-SABER task cohort on one matched software/hardware environment. Do not choose
which seeds to run based on interim outcomes. A failed child stops new scheduling
while existing children finish. Existing run directories/claims are never
overwritten, silently resumed, auto-renamed, or silently skipped.

## Sources and results

- `upstream/`, `upstream_manifest.json`: immutable byte copies and origin SHA256s.
- `code/src/metacog_lite_v0/`: independent integrated code.
- `protocol.json`, `protocol.py`: all scientific settings and paired presets.
- `source_manifest.json`: frozen source/config/test contract.
- `tests/`, `validate.py`: update parity, support, HER, environment, and smoke checks.
- `runs/`: production outputs, created only by production execution.
- `.production_claims/`: run status/failures, environment versions, source digest.
- `validation/`: disposable scientific-validation runs, not paper results.
- `analyze.py`: strict completion validation, seed metrics, paired bootstrap CIs.
- `schedule_spec.json`, `scheduled_run.py`: frozen dependency and stage order.
- `gates.py`, `gpu_validate.py`: fail-closed production authorization and GPU checks.

`python -B analyze.py` requires all 40 completed runs for inferential summaries;
otherwise it emits an explicit missing/failed list and exits nonzero. The queue
also uses `--tasks drawer_open` and `--tasks drawer_close` for complete per-task
summaries (20 runs each), never partial-seed summaries. Throughput
includes evaluation/checkpoint overhead, but not environment/Numba initialization.
Do not equate matched UTD with matched wall-clock cost.

See `PROTOCOL.md` for interpretation, failure handling, and implementation changes.

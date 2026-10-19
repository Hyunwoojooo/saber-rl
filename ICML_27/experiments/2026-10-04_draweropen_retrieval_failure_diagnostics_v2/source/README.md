# DrawerOpen retrieval and failure diagnostics

Independent v2 recovery copy created on 2026-10-04. The original 2026-10-02
analysis, blocked queue record, and all training runs remain unchanged.
See RECOVERY.md for the defect and validation plan. No new training is launched.

Independent offline follow-up, authorized to run after the PAL experiment.
See PROTOCOL.md for the fixed scope, limitations, and measurement definitions.

The `upstream/code/` directory is a byte-identical copy of the frozen assignment
implementation. Do not edit upstream files. Retained run data stay in the original
experiment directories and are only read; this folder owns all new outputs.

## Operations

Use the workspace `.venv/bin/python` from this directory:

```bash
python -B -m unittest discover -s tests -p test_core.py -v
python -B freeze.py --create
python -B after_pal.py
python -B after_pal.py --execute --poll-seconds 300
```

The first command uses synthetic, standard-library-only tests. The queue runs
the additional NumPy/Torch/scanner tests and real checkpoint pilot only after PAL.
The non-execute queue command is a read-only readiness check, not a launch.

The queue is one-shot. Do not start a second copy or reuse output directories.
Create `outputs/automation/STOP` to request cancellation of this follow-up only;
the scheduler never terminates the upstream PAL jobs. A blocked queue needs manual
review and is not silently resumed. This local queue does not survive a reboot
unless explicitly relaunched under a new reviewed execution record.

## Scheduled work

1. Verify successful PAL completion and its final analysis.
2. Run synthetic regression tests, including copied-tail mask parity.
3. Analyze all 30 assignment runs for failure/saturation patterns.
4. Run a 256-query seed-1 retrieval correctness/timing pilot.
5. Audit 1024 queries from each of ten final SABER checkpoints, comparing tail,
   five random candidate samples, and exhaustive matching.
6. Verify artifact hashes and write `outputs/final/REPORT.md`.

There is no further online training, no critic-branch experiment, no automatic
hyperparameter tuning, and no publication/push in this queue.

# DrawerOpen: support vs random vs TD-error assignment

Independent experiment for the post-review mechanism question. No existing
experiment, paper, or result is edited. Read [PROTOCOL.md](PROTOCOL.md) before running.

## Layout

- `upstream/code/src/`: unmodified frozen Strict source.
- `code/src/`: local trainer with assignment-only controls and fail-fast checks.
- `protocol.py`: fixed three-arm, ten-seed production configuration.
- `launch.py`, `train.py`: bounded parallel execution and fresh-attempt records.
- `analyze.py`: exact-grid validation, failure inventory, seed results and paired CIs.
- `tests/`, `validate.py`: regression tests and the production gate.
- `freeze.py`, `source_manifest.json`: source integrity, separate from Git identity.
- `outputs/`: ignored run data, logs, status records, environment snapshots.

## Commands

Use the existing tested experiment interpreter (Python 3.12 with PyTorch,
NumPy, Gymnasium, Gymnasium-Robotics, MetaWorld, MuJoCo and Numba). Do not upgrade
the environment during a cohort. Exact installed versions are saved per run.
The example interpreter below is local to this machine; the experiment itself
does not import any source from the parent repository.

```bash
cd /home/mimiclab-gigabyte-gen4/SABER/experiments_post_review/2026-09-25_draweropen_ddpg_assignment
PY=/home/mimiclab-gigabyte-gen4/SABER/.venv/bin/python
$PY freeze.py
$PY validate.py
$PY launch.py --profile smoke --seeds 1 --parallel 3 --output-root outputs/final_smoke
$PY validate.py --smoke-root outputs/final_smoke
```

After the gate passes, inspect commands without launching:

```bash
$PY launch.py --profile production --parallel 5 --dry-run
```

An explicit full launch (30 fresh runs, maximum five concurrent processes):

```bash
$PY launch.py --profile production --parallel 5
```

Subsets are allowed for scheduling, not for redefining the analysis population:

```bash
$PY launch.py --profile production --arms td_topk --seeds 1 2 3 4 5 --parallel 5
$PY analyze.py --output-root outputs
```

The analysis always requires seeds 1--10 in all three arms for primary inference.
Until then it writes the status/failure report and available descriptive seed
metrics, and exits nonzero. Do not interpret smoke results as performance evidence.

Run names cannot be overwritten. Historical results require a separate reuse
audit; changing RNG alone is not grounds to discard them, but no reuse is assumed
by this launcher. See the protocol for failure/restart rules and claim limits.

## Verification

Tests check frozen-update parity including nonempty optimizer state, exact-k
uniform selection, tied TD ranking, mean-one critic-only weights, unchanged
HER/replay/environment files, replay wrapping, eligibility when coarse action
support is absent, RNG isolation, NA handling, informative-update denominators,
joint bootstrap quantiles, config parity, and missing-checkpoint rejection.

The source manifest must be created once (`freeze.py --create`) after changes and
tests, before smoke or production. Once production starts, copy this experiment
to a new directory for any scientific-code change; do not refresh its manifest.

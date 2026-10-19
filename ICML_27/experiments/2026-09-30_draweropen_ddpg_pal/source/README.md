# DrawerOpen DDPG HER PAL comparison

Implementation prepared on 2026-09-30. Execution validation is pending.
No regression tests, smoke training, or production training have been run for
this experiment. No production gate is supplied. The existing assignment
experiment and its environment must remain unchanged.

This independent experiment adds ten fresh HER+PAL runs to compare against the
ten SABER runs in the frozen 2026-09-25 assignment experiment. It does not launch
SABER again and does not extend or modify that experiment's queue. PAL is a
published loss baseline, not another count-matched assignment control.

## Scope

- DrawerOpen-v3, reward v2, DDPG+HER, seeds 1--10, 2M steps, UTD=1.
- Author PAL loss, alpha=0.4 and min_priority=1, detached minibatch normalizer.
- Replay-action regression, original actor loss, targets, gradient clipping,
  optimizer settings, HER convention, and evaluation reset seeds are retained.
- No support lookup, policy support query, populated support index, E/k, or
  counterfactual assignment masks. HER's episode index is unchanged.
- No package installation, environment upgrade, checkpoint continuation, or
  automatic replacement of failed/poor-performing seeds.
- All new output paths must be below this experiment's `outputs/` directory.

The complete scientific contract and interpretation limits are in PROTOCOL.md.
The original author's source and MIT license are retained in third_party/lap_pal.

## Files

| File | Role |
|---|---|
| code/src/metacog_lite_v0/pal.py | Original piecewise loss and detached normalizer |
| code/src/metacog_lite_v0/agent.py | Single-critic DDPG PAL integration |
| code/src/metacog_lite_v0/trainer.py | Frozen training loop with support bypass |
| upstream/ | Immutable source and protocol copied from the running experiment |
| protocol.py | Fixed configuration, seeds, and expected SABER reference config |
| launch.py / train.py | Preview-first, PAL-only fresh-run launchers |
| analyze.py | Completion checks, reference audit, AUC and paired bootstrap |
| tests/ | Loss, gradient, update, HER, wiring, analysis, and safety tests |
| validate.py | Deferred CPU tests and same-device smoke certification |
| static_check.py | AST and snapshot checks only; no model/library imports |
| source_manifest.json | Source snapshot SHA256 inventory |

## Current nonexecution boundary

Only source copying, file editing, standard-library AST parsing, and file-hash
checks are allowed during preparation. Do not run `validate.py --run-tests`,
`train.py --execute`, or `launch.py --execute` while execution is on hold.
Do not install dependencies into the active training environment.

## Deferred validation commands

The commands below are instructions for a later authorized validation window;
they have NOT been run. Run them from this experiment directory using the
existing environment, without installing or upgrading packages.

```bash
../../.venv/bin/python static_check.py
../../.venv/bin/python validate.py --run-tests
../../.venv/bin/python launch.py --profile smoke --device cpu --seeds 1 --parallel 1 --output-root outputs/cpu_validation --execute
../../.venv/bin/python validate.py --run-tests --smoke-root outputs/cpu_validation
```

A CPU gate cannot authorize GPU production. After the shared GPU is available
and GPU execution is approved, run and certify a GPU smoke separately:

```bash
../../.venv/bin/python launch.py --profile smoke --device cuda --seeds 1 --parallel 1 --output-root outputs/cuda_validation --execute
../../.venv/bin/python validate.py --run-tests --smoke-root outputs/cuda_validation
```

Smoke is 6k steps with 1,001 critic updates, a 1k random-action prefix,
evaluations every 2k steps, and two evaluation episodes. It checks execution,
not performance. No smoke outcome is used for hyperparameter selection.
Tests run CPU-only. Revalidation invalidates old gates before executing tests;
certification binds a gate to source, device, and the observed environment.

## Deferred production commands

Preview only, with no training or output creation:

```bash
../../.venv/bin/python launch.py --profile production --device cuda --parallel 5 --dry-run
```

Only after validation and separate approval for full training:

```bash
../../.venv/bin/python launch.py --profile production --device cuda --parallel 5 --execute
```

This launches PAL seeds 1--10 only. There is no background session creation,
automatic start-after-current-experiment hook, or mutation of existing tmux.
Output: outputs/production/pal/seed_XX/attempt_YY.
Completed runs are never overwritten. Infrastructure retries require an explicit
reason and start from zero; original attempts and logs are retained.

## Deferred analysis

```bash
../../.venv/bin/python analyze.py --output-root outputs
../../.venv/bin/python analyze.py --output-root outputs --saber-root ../2026-09-25_draweropen_ddpg_assignment
```

The reference directory is read-only. No earlier paper cohort is substituted.
The reference audit checks the pinned source, all ten support-arm configs,
fresh-run history, full scheduled evaluation grid, and matching physical host
and software stack. A mismatch withholds inference; it is not silently waived.

Analysis writes all attempt states, seed-level metrics, and (only with all ten
valid pairs) the primary SABER-minus-PAL AUC difference and percentile 95% CI.
Missing checkpoints never shorten the 10k--2M AUC interval. Poor finite outcomes
remain in the analysis. No equivalence claim follows from a CI containing zero.

## Runtime interpretation

Actual per-evaluation elapsed times, total loop time, and steps/s are retained.
They include evaluation/checkpoint/logging costs in the inherited loop but not
all process setup. The SABER reference computes extra assignment diagnostics.
The analysis therefore marks cross-method runtime comparison invalid and does
not report a speedup or time-efficiency ranking. A separate matched-instrumentation
measurement is needed for that claim; it is not authorized or launched here.

# DrawerOpen DDPG HER PAL comparison

Independent v2 copy prepared on 2026-10-01. PAL runtime validation is pending.
The v1 source and the running assignment experiment remain unchanged. V2 fixes
the analysis-test environment fixture and policy-Q logging, adds regression tests,
and automates the six-step final validation. It does not change the PAL objective,
hyperparameters, replay, actor/target update rules, or evaluation protocol.

This independent experiment adds ten fresh HER+PAL runs to compare against the
ten SABER runs in the frozen 2026-09-25 assignment experiment. It does not launch
SABER again or modify that experiment's queue. PAL is a
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
| after_current.py | Standard-library-only waiter, then fail-closed validation and launch |
| final_checks.py | Reference audit, CUDA loss parity, checkpoint reload and analysis checks |

## Authorized conditional execution

The user authorized automatic validation and, only if all checks pass, PAL
production after the existing 30-run assignment cohort finishes. Preparation
uses source copying, edits, AST/hash checks, and standard-library queue tests.
No Torch regression tests, simulator smoke, or training run during the wait.
No dependency installation or upgrade is permitted.

The local one-shot waiter polls every five minutes, requires all 30 complete
records and no assignment training/launcher process, and then runs:

1. Immutable source and syntax checks.
2. The full CPU regression and synthetic-analysis suite.
3. Read-only validation of the ten actual SABER references and matched environment.
4. CUDA loss/gradient comparison to the preserved author's implementation.
5. Real 6k CUDA smoke, CPU tests again, and same-device smoke certification.
6. Checkpoint reload/re-evaluation and final readiness certification, then preview.

Only then does it launch ten fresh PAL runs, five in parallel, and finally
recompute the paired SABER-minus-PAL results. Any failed check, unknown GPU
occupant, incomplete reference, or source/environment mismatch withholds
production. There are no automatic fixes, outcome-based retries, or restarts.
Zero-success completed runs are valid. Smoke success is not a pass threshold.

```bash
../../.venv/bin/python -B after_current.py --check
../../.venv/bin/python -B after_current.py --execute --poll-seconds 300
```

The worker is intended for a dedicated tmux session. Keep the machine on;
tmux survives terminal disconnection, not reboot. It waits at most 72 hours.
Duplicate workers are blocked by a file lock and a one-shot status record.
State and stage logs are in `outputs/automation/`. Creating
`outputs/automation/STOP` requests cancellation, including stopping the worker's
own active stage. Existing assignment processes are never stopped by this worker.
No email, chat notification, or restart after reboot is configured.

## Deferred validation commands

The following are manual alternatives for an authorized validation window;
do not run them concurrently with the automatic worker. Run them from this
experiment directory in the existing environment, without package changes.

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
../../.venv/bin/python final_checks.py --stage reference
../../.venv/bin/python final_checks.py --stage cuda_loss
../../.venv/bin/python final_checks.py --stage readiness --smoke-root outputs/cuda_validation
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

Only after the same-source final readiness checks (conditional approval was
given on 2026-10-01):

```bash
../../.venv/bin/python launch.py --profile production --device cuda --parallel 5 --execute
```

This launches PAL seeds 1--10 only. The launcher itself does not create a
background session. The separate waiter never mutates the existing tmux session.
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

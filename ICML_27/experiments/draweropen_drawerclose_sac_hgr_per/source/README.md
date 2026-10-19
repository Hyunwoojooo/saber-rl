# Drawer SAC HGR / full-PER baseline extension

Independent experiment created on 2026-10-16. The user authorized implementation
and a gated queue after the current Drawer SAC HER/SABER cohort. This directory
does not modify any earlier trainer, artifact, or Python environment.

## Scope

| Task | Methods | Seeds | Steps/run |
| --- | --- | --- | --- |
| DrawerOpen-v3 | SAC + HGR; SAC + HER + full PER | 1-5 | 2,000,000 |
| DrawerClose-v3 | SAC + HGR; SAC + HER + full PER | 1-5 | 2,000,000 |

20 new runs, 40M environment steps, UTD=1, at most five workers. DrawerOpen runs
first, then DrawerClose. Seeds 6-10 are not scheduled. Reference HER/SABER runs
are not rerun: all 20 must finish and pass the reference audit before this queue
may start validation or training. No failed or zero-success seed is excluded.

This is an exploratory baseline extension decided after inspecting the reference
cohort. It is not a prospectively selected confirmatory test of superiority.

## Scientific contract

The common SAC, sparse reward, environment reset, termination, evaluation and
network settings are copied from `2026-10-12_drawer_sac_fpp_v2_recovery_s1_5`.
The original FPP HGR and full-PER implementations are frozen under `upstream/`.
See [PROTOCOL.md](PROTOCOL.md) for their exact sampling/weight conventions and
the intentional HGR versus HER future-goal indexing distinction.

- Automatic entropy tuning, initial alpha=0.2, target entropy=-4.
- Two 256-unit hidden layers, gamma=0.98, tau=0.005, learning rates=3e-4.
- Replay=300,000, batch=256, warmup=5,000, random actions=10,000.
- 1,995,001 critic/actor/alpha updates per production run.
- Both baselines disable SABER support queries. Actor and alpha losses remain
  unweighted; their input distribution changes with prioritized replay.
- No changes to reward, success thresholds, evaluation seeds, or timeout bootstrap.
- Frozen source SHA256, configs, full package versions, RNG/model/replay state,
  curves, summaries, and execution/failure claims are retained locally.

## Validation and scheduling

Use the existing repository `.venv/bin/python`; do not install into the live
experiment environment. Commands below are relative to this directory.

```bash
python -B manifest.py --write       # only before queue/production reservation
python -B validate.py              # CPU-only tests and 4 x 600-step real-env smokes
python -B scheduled_run.py --check  # read-only queue/dependency check
python -B scheduled_run.py --execute
```

The scheduler waits without importing Torch/CUDA. After the predecessor succeeds
and the GPU is idle it executes:

1. CPU revalidation on the frozen source.
2. FPP SAC CUDA update parity with baseline priority extraction enabled.
3. Four 600-step GPU smokes, then four 100k-step stability runs (parallel=4).
4. DrawerOpen production, five workers, followed by its complete paired analysis.
5. DrawerClose production, followed by its analysis and a combined final report.

Every GPU validation arm must pass finite-loss/priority checks, evaluation-grid
and update-budget checks, checkpoint reload, baseline-specific IS checks and
reference-environment matching. Success rate is NOT an acceptance gate. The
100k runs retain production network, replay capacity, warmup and beta schedules.
They are validation-only and are never continued or counted as production.

Any failed stage blocks all later stages. There is no automatic retry, restart,
resume, or replacement seed. The detached systemd service survives logout but is
not configured to restart after a reboot. Inspect a power failure manually.

## Files and monitoring

- `protocol.json`, `schedule_spec.json`: frozen settings and stage order.
- `upstream_manifest.json`: immutable copied sources with original relative paths.
- `source_manifest.json`: local source/config/test/document SHA256 manifest.
- `validation/validation_report.json`: CPU verification evidence.
- `validation/gpu_validation_report.json`: deferred GPU/stability gate.
- `automation/status.json`, `automation/*.log`: queue heartbeat and stages.
- `.production_claims/`, `launch_logs/`, `runs/`: all execution records and outputs.
- `analysis/final.json`: all four methods per task, seed-level metrics and paired CIs.

No production launch is possible through the canonical entry points without
matching CPU and GPU gates. Existing results are read-only and checked against
their original final report, CSVs, execution claims, source manifests and configs.

Failure analysis of the reference SAC cohort is separate from this queue. This
queue does not start additional failure-analysis or seed-extension jobs.

# Offline retrieval and failure/saturation diagnostics

Status: **completed_offline**. Snapshot: **2026-10-19T05:52:33.729730+00:00 UTC / 2026-10-19T14:52:33.729730+09:00 KST**.

Completed offline analyses; no new training runs.

- Ten SABER final checkpoints: 1,024 fixed HER queries per seed, five random-retrieval repeats.
- Tail / random / full support rates: 0.537109 / 0.625352 / 0.902734.
- Miss rates conditional on full-positive support: 0.410458 / 0.311727 / 0.
- Failure/saturation analysis covers all 30 assignment runs. Seven had zero evaluation success;
  saturation occurred in 7/7 of these and 0/23 of the nonzero-success runs.

These are final-checkpoint and observational diagnostics, not causal failure explanations.
Full search is a reference for the same discretized context, not true continuous support.
Timing is CPU reference implementation timing, not production GPU/Numba speed.
See [the detailed report](results/outputs/final/REPORT.md) and the seed-level CSVs.

## Code and provenance

- [source/](source/) contains the frozen executed source, protocol, tests, upstream copies and manifests.
- [source/README.md](source/README.md) is the ORIGINAL historical instruction file; its old pending/status text is not this snapshot's status.
- [provenance.json](provenance.json) records byte-identical source hashes and exported report hashes.
- Checkpoints, replay buffers, raw training logs, virtual environments and live process control files are omitted.
- Source dependencies and original local-path instructions are retained. Use the root materialization instructions before running tests or launchers; this is not a standalone reproduction bundle.
- No training is launched by browsing, exporting or verifying this archive. See [the archive guide](../../README.md).

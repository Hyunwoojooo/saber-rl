# Drawer SAC preparation v1

Status: **superseded_before_production**. Snapshot: **2026-10-19T05:52:33.729730+00:00 UTC / 2026-10-19T14:52:33.729730+09:00 KST**.

Pre-production preparation; its waiting queue was stopped before production and replaced by v2. No performance results belong to this version.
## Known observation-representation caveat

The inherited MetaWorld observation includes the original goal in its final three components, while HER changes the separately concatenated goal. The executed source is preserved, not silently fixed. This is shared by the compared SAC methods; its causal effect on performance has not been established. A protocol-changing correction would require a separate matched experiment.


## Code and provenance

- [source/](source/) contains the frozen executed/prepared source, protocol, tests, upstream copies and manifests.
- [source/README.md](source/README.md) is the ORIGINAL historical instruction file; its old pending/status text is not this snapshot's status.
- [provenance.json](provenance.json) records byte-identical source hashes and exported report hashes.
- Checkpoints, replay buffers, raw training logs, virtual environments and live process control files are omitted.
- Source dependencies and original local-path instructions are retained. Use the root materialization instructions before running tests or launchers; this is not a standalone reproduction bundle.
- No training is launched by browsing, exporting or verifying this archive. See [the archive guide](../../README.md).

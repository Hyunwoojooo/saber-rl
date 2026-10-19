# Drawer SAC original execution

Status: **interrupted_recovered_separately**. Snapshot: **2026-10-19T05:52:33.729730+00:00 UTC / 2026-10-19T14:52:33.729730+09:00 KST**.

Original SAC execution interrupted by a power outage. DrawerOpen HER seeds 1-3 completed; five other started runs were interrupted. Completed and interrupted attempts are recorded in the recovery cohort's final analysis. The final 20-run result belongs to ../2026-10-12_drawer_sac_fpp_v2_recovery_s1_5/ and must not be counted twice.
## Known observation-representation caveat

The inherited MetaWorld observation includes the original goal in its final three components, while HER changes the separately concatenated goal. The executed source is preserved, not silently fixed. This is shared by the compared SAC methods; its causal effect on performance has not been established. A protocol-changing correction would require a separate matched experiment.


## Code and provenance

- [source/](source/) contains the frozen executed/prepared source, protocol, tests, upstream copies and manifests.
- [source/README.md](source/README.md) is the ORIGINAL historical instruction file; its old pending/status text is not this snapshot's status.
- [provenance.json](provenance.json) records byte-identical source hashes and exported report hashes.
- Checkpoints, replay buffers, raw training logs, virtual environments and live process control files are omitted.
- Source dependencies and original local-path instructions are retained. Use the root materialization instructions before running tests or launchers; this is not a standalone reproduction bundle.
- No training is launched by browsing, exporting or verifying this archive. See [the archive guide](../../README.md).

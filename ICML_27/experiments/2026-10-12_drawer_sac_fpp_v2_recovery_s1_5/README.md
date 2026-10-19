# Drawer SAC HER / SABER completed cohort

Status: **completed**. Snapshot: **2026-10-19T05:52:33.729730+00:00 UTC / 2026-10-19T14:52:33.729730+09:00 KST**.

Completed: 20 runs, seeds 1-5, 2M steps each; auto-temperature SAC follows the FPP reference.

| Task | Method | Mean AUC | AUC SD | Mean final success |
| --- | --- | ---: | ---: | ---: |
| drawer_open | her | 0.447035 | 0.410848 | 0.600 |
| drawer_open | saber | 0.448492 | 0.410829 | 0.600 |
| drawer_close | her | 0.260628 | 0.284573 | 0.200 |
| drawer_close | saber | 0.103317 | 0.056453 | 0.000 |

drawer_open, SABER minus HER: +0.001457, 95% CI [-0.046633, 0.041283].

drawer_close, SABER minus HER: -0.157312, 95% CI [-0.419070, 0.025025].

Neither interval excludes zero. DrawerClose has a negative mean difference; unfavorable outcomes are retained.
Three completed DrawerOpen HER seeds were reused after the power outage; five interrupted runs
were rerun from step zero. Seventeen runs in this recovery plus three retained runs form ONE
20-run cohort, not an additional independent replication. No curves were spliced and no checkpoint continuation was used.
This is the single published SAC HER/SABER source folder. Its frozen scientific source also covers the three retained pre-outage runs.
See [final.json](results/analysis/final.json) for seed-level outcomes and the interruption inventory.
## Known observation-representation caveat

The inherited MetaWorld observation includes the original goal in its final three components, while HER changes the separately concatenated goal. The executed source is preserved, not silently fixed. This is shared by the compared SAC methods; its causal effect on performance has not been established. A protocol-changing correction would require a separate matched experiment.


## Code and provenance

- [source/](source/) contains the frozen executed source, protocol, tests, upstream copies and manifests.
- [source/README.md](source/README.md) is the ORIGINAL historical instruction file; its old pending/status text is not this snapshot's status.
- [provenance.json](provenance.json) records byte-identical source hashes and exported report hashes.
- Checkpoints, replay buffers, raw training logs, virtual environments and live process control files are omitted.
- Source dependencies and original local-path instructions are retained. Use the root materialization instructions before running tests or launchers; this is not a standalone reproduction bundle.
- No training is launched by browsing, exporting or verifying this archive. See [the archive guide](../../README.md).

# DrawerOpen and DrawerClose / SAC: HGR and PER

Status: **completed**. Snapshot: **2026-10-24T11:58:31.855827+00:00 UTC / 2026-10-24T20:58:31.855827+09:00 KST**.

Completed: 20 new HGR/PER runs plus 20 reused HER/SABER references; 40 unique SAC runs total.
Both tasks use seeds 1-5 and 2M steps, with FPP-reference auto-temperature SAC.
This is an exploratory extension after observing HER/SABER outcomes; no automatic seeds 6-10 extension.
PER is full prioritized replay with IS critic weights, not loss-only TD weighting.

| Task | Method | Mean AUC | AUC SD | Mean final success |
| --- | --- | ---: | ---: | ---: |
| drawer_open | her | 0.447035 | 0.410848 | 0.600 |
| drawer_open | saber | 0.448492 | 0.410829 | 0.600 |
| drawer_open | hgr | 0.706583 | 0.065872 | 1.000 |
| drawer_open | per | 0.797588 | 0.072233 | 1.000 |
| drawer_close | her | 0.260628 | 0.284573 | 0.200 |
| drawer_close | saber | 0.103317 | 0.056453 | 0.000 |
| drawer_close | hgr | 0.211784 | 0.131396 | 0.030 |
| drawer_close | per | 0.487312 | 0.304340 | 0.550 |

SABER-minus-PER paired AUC intervals lie below zero for both tasks; unfavorable results are retained.
Both SABER-minus-HER intervals include zero. This neither establishes superiority nor equivalence.
DrawerClose's early Hit90 events must not be confused with sustained or final success.
See [final.json](results/analysis/final.json) for all seed metrics and adjusted paired intervals.
Compact evaluation data: [DrawerOpen](../../../metaworld_Drawer_Open_SAC/) and [DrawerClose](../../../metaworld_Drawer_Close_SAC/).
Recorded runtimes include instrumentation and varying concurrent workloads, not controlled wall-clock comparisons.
## Known observation-representation caveat

The inherited MetaWorld observation includes the original goal in its final three components, while HER changes the separately concatenated goal. The executed source is preserved, not silently fixed. This is shared by the compared SAC methods; its causal effect on performance has not been established. A protocol-changing correction would require a separate matched experiment.


## Code and provenance

- [source/](source/) contains the frozen executed source, protocol, tests, upstream copies and manifests.
- [source/README.md](source/README.md) is the ORIGINAL historical instruction file; its old pending/status text is not this snapshot's status.
- [provenance.json](provenance.json) records byte-identical source hashes and exported report hashes.
- Checkpoints, replay buffers, raw training logs, virtual environments and live process control files are omitted.
- Source dependencies and original local-path instructions are retained. Use the root materialization instructions before running tests or launchers; this is not a standalone reproduction bundle.
- No training is launched by browsing, exporting or verifying this archive. See [the archive guide](../../README.md).

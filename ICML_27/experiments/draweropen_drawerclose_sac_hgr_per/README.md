# DrawerOpen and DrawerClose / SAC: HGR and PER

Status: **in_progress**. Snapshot: **2026-10-19T05:52:33.729730+00:00 UTC / 2026-10-19T14:52:33.729730+09:00 KST**.

Question: how do HER/SABER compare with HGR and full replay-prioritized PER under the matched Drawer SAC setup?
Seeds 1-5, 2M steps per run, two tasks x two new methods = 20 planned runs.
PER changes replay sampling; it is not the older PER-style loss-only control.

| Task | Method | Seed | Status | Latest step | Eval step | Success at eval | AUC to eval step |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| drawer_open | hgr | 1 | completed | 2000000 | 2000000 | 1.000 | 0.723618 |
| drawer_open | hgr | 2 | completed | 2000000 | 2000000 | 1.000 | 0.660050 |
| drawer_open | hgr | 3 | completed | 2000000 | 2000000 | 1.000 | 0.800251 |
| drawer_open | hgr | 4 | completed | 2000000 | 2000000 | 1.000 | 0.719598 |
| drawer_open | hgr | 5 | running | 964800 | 960000 | 1.000 | 0.236842 |
| drawer_open | per | 1 | completed | 2000000 | 2000000 | 1.000 | 0.864824 |
| drawer_open | per | 2 | completed | 2000000 | 2000000 | 1.000 | 0.844221 |
| drawer_open | per | 3 | completed | 2000000 | 2000000 | 1.000 | 0.776131 |
| drawer_open | per | 4 | completed | 2000000 | 2000000 | 1.000 | 0.820101 |
| drawer_open | per | 5 | completed | 2000000 | 2000000 | 1.000 | 0.682663 |
| drawer_close | hgr | 1 | pending | 0 | NA | NA | NA |
| drawer_close | hgr | 2 | pending | 0 | NA | NA | NA |
| drawer_close | hgr | 3 | pending | 0 | NA | NA | NA |
| drawer_close | hgr | 4 | pending | 0 | NA | NA | NA |
| drawer_close | hgr | 5 | pending | 0 | NA | NA | NA |
| drawer_close | per | 1 | pending | 0 | NA | NA | NA |
| drawer_close | per | 2 | pending | 0 | NA | NA | NA |
| drawer_close | per | 3 | pending | 0 | NA | NA | NA |
| drawer_close | per | 4 | pending | 0 | NA | NA | NA |
| drawer_close | per | 5 | pending | 0 | NA | NA | NA |

Running AUC is partial-horizon, NOT a final 2M result. Do not average it with completed-run AUCs.
The five-seed HGR cohort and the DrawerClose extension are not yet complete at this snapshot.
DrawerClose is queued after DrawerOpen completion and analysis; maximum concurrency is five.
There is no automatic expansion to seeds 6-10. This GitHub status is static, not live monitoring.
The HER/SABER reference is the completed recovery cohort; see [progress.json](results/progress.json).
## Known observation-representation caveat

The inherited MetaWorld observation includes the original goal in its final three components, while HER changes the separately concatenated goal. The executed source is preserved, not silently fixed. This is shared by the compared SAC methods; its causal effect on performance has not been established. A protocol-changing correction would require a separate matched experiment.


## Code and provenance

- [source/](source/) contains the frozen executed source, protocol, tests, upstream copies and manifests.
- [source/README.md](source/README.md) is the ORIGINAL historical instruction file; its old pending/status text is not this snapshot's status.
- [provenance.json](provenance.json) records byte-identical source hashes and exported report hashes.
- Checkpoints, replay buffers, raw training logs, virtual environments and live process control files are omitted.
- Source dependencies and original local-path instructions are retained. Use the root materialization instructions before running tests or launchers; this is not a standalone reproduction bundle.
- No training is launched by browsing, exporting or verifying this archive. See [the archive guide](../../README.md).

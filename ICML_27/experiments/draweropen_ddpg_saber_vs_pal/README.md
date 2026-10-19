# DrawerOpen / DDPG: SABER vs PAL

Status: **completed**. Snapshot: **2026-10-19T05:52:33.729730+00:00 UTC / 2026-10-19T14:52:33.729730+09:00 KST**.

| Method | Mean AUC | AUC SD | Mean final success |
| --- | ---: | ---: | ---: |
| HER + PAL | 0.545352 | 0.302841 | 0.800 |
| SABER | 0.652613 | 0.248402 | 0.890 |

All results use seeds 1-10, DrawerOpen-v3 / DDPG / HER, 2M steps per run, UTD=1.

Question: does SABER offer value over a published uniform-replay loss-side baseline?
PAL uses alpha=0.4, min_priority=1, and the detached minibatch normalizer.
Only PAL's ten runs are new; the audited SABER reference is the assignment cohort.
SABER minus PAL: +0.107261; 95% CI [-0.153141, 0.357651].
Instrumentation differs; recorded runtime does not establish algorithmic time efficiency.

The intervals include zero: these descriptive mean differences do not establish
statistical superiority, equivalence, universal necessity of support, or generality beyond this cohort.
AUC uses the exact 10k-2M grid and denominator 2M-10k. See the full paired analysis in
[report.json](results/outputs/analysis/report.json).

## Code and provenance

- [source/](source/) contains the frozen executed source, protocol, tests, upstream copies and manifests.
- [source/README.md](source/README.md) is the ORIGINAL historical instruction file; its old pending/status text is not this snapshot's status.
- [provenance.json](provenance.json) records byte-identical source hashes and exported report hashes.
- Checkpoints, replay buffers, raw training logs, virtual environments and live process control files are omitted.
- Source dependencies and original local-path instructions are retained. Use the root materialization instructions before running tests or launchers; this is not a standalone reproduction bundle.
- No training is launched by browsing, exporting or verifying this archive. See [the archive guide](../../README.md).

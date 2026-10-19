# DrawerOpen / DDPG: SABER Tail vs Random-512 Retrieval

Status: **completed**. Snapshot: **2026-10-19T05:52:33.729730+00:00 UTC / 2026-10-19T14:52:33.729730+09:00 KST**.

| Method | Mean AUC | AUC SD | Mean final success |
| --- | ---: | ---: | ---: |
| SABER Tail-512 (reused) | 0.652613 | 0.248402 | 0.890 |
| SABER Random-512 retrieval | 0.430226 | 0.377206 | 0.590 |

All results use seeds 1-10, DrawerOpen-v3 / DDPG / HER, 2M steps per run, UTD=1.

Question: does changing coarse-bucket candidate retrieval from tail to uniform random
selection, with the same 512-candidate cap, change online learning?
This random operation selects replay candidates, not high-weight minibatch positions.
Ten Random-512 runs are new; all ten Tail references are reused, including the failed-to-learn seed.
Random-512 minus Tail: -0.222387; 95% CI [-0.518372, 0.080932].
Final checkpoint retrieval audits for both arms are included in results/outputs/final_retrieval/.

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

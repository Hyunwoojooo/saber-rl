# Support / Random / TD-error Top-k assignment

Status: **completed**. Snapshot: **2026-10-19T05:52:33.729730+00:00 UTC / 2026-10-19T14:52:33.729730+09:00 KST**.

| Method | Mean AUC | AUC SD | Mean final success |
| --- | ---: | ---: | ---: |
| SABER | 0.652613 | 0.248402 | 0.890 |
| Context-eligible random assignment | 0.515678 | 0.359443 | 0.700 |
| Context-eligible TD-error Top-k | 0.534623 | 0.380191 | 0.700 |

All results use seeds 1-10, DrawerOpen-v3 / DDPG / HER, 2M steps per run, UTD=1.

Question: does support-based high-weight placement add value when context eligibility,
the per-update high-weight count, 2:1 raw weights, and mean-one normalization are controlled?
Each online run computes its own eligibility and count; buffers and counts are not shared between runs.

- support_minus_random: delta AUC +0.136935; 95% CI [-0.127451, 0.384334]; Bonferroni 97.5% CI [-0.162888, 0.414626].
- support_minus_td_topk: delta AUC +0.117990; 95% CI [-0.054866, 0.324917]; Bonferroni 97.5% CI [-0.071018, 0.360704].

The intervals include zero: these descriptive mean differences do not establish
statistical superiority, equivalence, universal necessity of support, or generality beyond this cohort.
AUC uses the exact 10k-2M grid and denominator 2M-10k. See the full paired analysis in
[report.json](results/outputs/analysis/report.json).

## Code and provenance

- [source/](source/) contains the frozen executed/prepared source, protocol, tests, upstream copies and manifests.
- [source/README.md](source/README.md) is the ORIGINAL historical instruction file; its old pending/status text is not this snapshot's status.
- [provenance.json](provenance.json) records byte-identical source hashes and exported report hashes.
- Checkpoints, replay buffers, raw training logs, virtual environments and live process control files are omitted.
- Source dependencies and original local-path instructions are retained. Use the root materialization instructions before running tests or launchers; this is not a standalone reproduction bundle.
- No training is launched by browsing, exporting or verifying this archive. See [the archive guide](../../README.md).

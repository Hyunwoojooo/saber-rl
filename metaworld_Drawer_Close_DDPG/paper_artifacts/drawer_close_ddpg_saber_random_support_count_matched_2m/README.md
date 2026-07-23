# DrawerClose DDPG Random-Support Count-Matched

Compact artifact for seeds 1–5. AUC uses only `eval/env_success_mean` at the
observed 10k–2M grid (200 points), without fallback, smoothing, interpolation,
forward-fill, or extrapolation.

| Method | AUC mean ± sample SD | Final mean ± sample SD |
|---|---:|---:|
| DrawerClose DDPG HER | 0.251005 ± 0.029646 | 0.160000 ± 0.181659 |
| DrawerClose DDPG HER + SABER Main | 0.324422 ± 0.082453 | 0.280000 ± 0.251496 |
| DrawerClose DDPG Random-Support Count-Matched | 0.276683 ± 0.049642 | 0.240000 ± 0.429244 |

Main − Random AUC difference:
`0.047739`; exhaustive paired-bootstrap
95% CI
`[-0.004070,
0.114724]`; exact two-sided sign-flip
`p=0.375000`.

Random-Support first computes Main SABER's support-positive count `k` on the
same minibatch, then assigns the high raw weight to exactly `k` uniformly
sampled batch positions without replacement. The packaged count traces audit
all 10,176 serialized update-era metrics rows per seed, including the first
update at step 5,000: reference and assigned
counts, raw weight histograms, and canonical support rates match exactly.
Optimizer updates between serialized rows are supported by the frozen source
definition but are not directly observed in the raw artifact.
The top-level `positive_count_invariant_audit.json` provides a machine-readable
five-seed rollup of this check.

## Provenance limitation

The legacy Random-Support result committed complete metrics and summaries but
did not preserve original config files, environment snapshots, checkpoints,
logs, or a serialized git SHA. Config/environment records in this package are
explicitly labelled reconstructed proxies. Repository history verifies that
the `code/src` tree and Random-Support launcher are identical between launcher
commit `bf5fe867b08ab3c374a80802dd706c2d6f88df2c` and raw-result commit
`ae6d93de0d1dad5bf8328fd3334addb00caec205`. This is source-tree consistency, **not** cryptographic
proof of the exact checkout used by the training processes.
Host-specific absolute `run_dir` strings in Random-Support summaries are
replaced with package-relative paths. Their original source-file hashes remain
recorded in `input_manifest.json`, and completion records retain only a
basename and SHA-256 digest of each original path.

HER and Main are reused, not retrained, from verified comparator artifact
commit `841eb706b1dcdc12b59db23ec545099d39d3c619`; their original training source commit was also
not recorded.

Verify independently from this directory:

    python3 verify_artifact.py --artifact-dir .

Full raw metrics, checkpoints, JSONL, and logs are excluded.

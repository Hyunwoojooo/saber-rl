# Recovery of the post-PAL retrieval diagnostic

## Observed defect

The 2026-10-02 queue completed its 41 synthetic tests and the 30-run failure
analysis, then stopped at the seed-1 retrieval pilot. The saved config contains
`thread_hint`, which the training code adds as execution-environment metadata.
The diagnostic loader incorrectly treated that known field as an unknown
learning parameter and raised before loading the checkpoint or measuring queries.

The original trainer already classifies `thread_hint` as serialized metadata.
The previous tests covered synthetic retrieval and masks, not the production
config deserialization boundary. PAL training and its completed paired analysis
were not affected by this post-processing error.

## Independent recovery

- Preserve the original scripts, blocked status, outputs, and source manifest.
- Copy source and tests here without carrying over any generated outputs.
- Preserve the original diagnostic manifest at
  `upstream/diagnostic_v1_source_manifest.json`; the copied training implementation
  remains byte-identical to the original frozen assignment source.
- Explicitly recognize saved metadata while continuing to reject unknown fields,
  missing learning parameters, malformed thread metadata, and altered seed schemas.
- Preserve complete saved configs for the existing checkpoint identity comparison.
- Keep all ten seeds, 1024 queries per seed, five random repeats, thresholds,
  masks, analysis definitions, and original AUC intervals unchanged.

## Verification and rerun

Run the expanded regression suite and audit all ten retained production configs.
Freeze this new diagnostic source, then execute the original fail-closed pipeline:
tests, a fresh 30-run failure analysis, the real-checkpoint pilot, all ten final
checkpoints, and artifact verification/report generation. Compare regenerated
failure-analysis tables against v1 to confirm that the repair does not change them.

All work is CPU-only and reads the original checkpoints/logs without resuming
training. Input and output SHA256 manifests are verified before reporting success.

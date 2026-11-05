# Assignment extension deep audit

Start with [한국어 보고서](outputs/REPORT_ko.md). This is a post-hoc analysis of completed runs, not a replacement experiment or a new primary result.

The audit reads the frozen study at `~/ICML2027/draweropen_ddpg_assignment_extension_20261020`. It never writes there or takes an optimizer step. It uses the repository's existing Python environment and the already isolated Matplotlib dependencies from the earlier Q-error audit. No shared package versions are changed.

Reproduction requires a new copy of this source directory without `outputs/` because stages refuse to replace generated CSVs/checkpoint records. Run, in order:

```bash
python -B test_audit.py
python -B run.py --stage logs
python -B run.py --stage checkpoints
python -B run.py --stage gradients
python -B run.py --stage saturation
python -B run.py --stage geometry
python -B run.py --stage finish
python -B verify_artifact.py
```

Checkpoint stages use CPU with one thread. The 120 diagnostic episodes replay the original final evaluation resets for four episodes per policy; they are 24,000 evaluation environment steps and zero new training updates. All source runs remain in the primary aggregate.

`outputs/manifest.json` records original/intermediate input SHA256 values, analysis source hashes and derived output hashes. Original learning source and all original result files are unchanged. These integrity checks do not prove causal identification or exclude every possible implementation problem.

The gradient and geometry follow-ups were added after initial terminal action saturation was observed; see [PROTOCOL.md](PROTOCOL.md) for the adaptive diagnostic history and interpretation limits.

# SABER Code Supplementary Builder

This directory contains the reviewer-facing Code & Data Supplement and the
deterministic builder used to refresh source snapshots, compact result
artifacts, provenance records, and the submission ZIP.

Run from the repository root:

```bash
python code_supplementary/build_package.py
python code_supplementary/saber_code_and_data/reviewer_tools/verify_package.py
```

The builder intentionally excludes checkpoints, replay buffers, raw logs,
untracked files, and repository metadata.

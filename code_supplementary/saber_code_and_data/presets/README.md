# Primary Experiment Presets

These seven JSON files are the only supported production entry points in this
review archive. Each file explicitly fixes the paper budget, HER convention,
SABER support parameters, critic-only scope, evaluation cadence, and seed
mapping. `reviewer_tools/launch_experiment.py` validates every preset before
constructing the underlying source-snapshot command.

The `paper_steps` value may be overridden only through the explicit reviewer
flag `--steps`; a dry run prints any override.

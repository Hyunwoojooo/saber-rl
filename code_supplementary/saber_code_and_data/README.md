# SABER Code and Data Supplement

This anonymous package accompanies the paper **SABER: Support-Conditioned
Adaptive Bellman Error Reweighting**. It contains the exact source snapshots
used by the seven primary task-backbone cohorts, fixed experiment presets,
compact result tables, selected diagnostic artifacts, tests, and scripts that
reproduce the reported aggregate statistics.

## What can be reproduced

1. Verify package integrity and anonymity.
2. Recompute primary means, sample standard deviations, paired differences,
   and 10,000-resample percentile-bootstrap confidence intervals from the
   included seed-level CSV.
3. Validate every compact derived artifact manifest.
4. Run unit and source-contract tests for SABER weighting, local support,
   HER future-goal indexing, replay-action critic regression, fixed-target
   diagnostics, primary-preset parity, and the assignment and loss-scope
   contracts of the four mechanism controls.
5. Inspect or launch any of the seven primary SABER configurations through one
   package-relative entry point.

Full training is computationally expensive: FPP cohorts use 1M environment
steps per seed and Drawer cohorts use 2M. The quick path below does not retrain
the models.

## Five-minute reviewer path

```bash
python reviewer_tools/verify_package.py
python reviewer_tools/reproduce_main_results.py --check
bash reviewer_tools/run_smoke_tests.sh
python reviewer_tools/launch_experiment.py --list
python reviewer_tools/launch_experiment.py --preset fpp_td3 --seed 1 --dry-run
```

See `QUICKSTART.md` for installation and training commands.

## Directory map

- `source/`: exact primary source snapshots, one per task-backbone cohort.
- `presets/`: explicit paper configurations; these override unsafe generic
  defaults such as actor+critic weighting or a larger replay scan.
- `results/derived/`: compact seed-level and aggregate CSVs used by the paper
  and Technical Supplement.
- `results/selected_artifacts/`: selected compact diagnostic and sensitivity
  artifacts with source hashes and provenance.
- `reviewer_tools/`: integrity, anonymity, result-reproduction, and launch
  interfaces.
- `tests/`: package-level scientific contract tests.
- `provenance/`: experiment catalog, source hashes, package inventory, and
  omitted-large-source disclosure.

## Scope and caveats

The package contains no checkpoints, replay buffers, or full raw training logs.
Those files are unnecessary for the reported aggregate tables and would make
the reviewer archive impractically large. Their omission is recorded in
`provenance/omitted_large_sources.csv`.

The verified reproduction stack is documented in `environment/`. Each paired
HER-SABER cohort was executed on one physical host under a matched environment.
Numerical trajectories can still vary across hardware and low-level simulator
libraries; the paper's paired-seed comparisons should therefore be treated as
the scientific unit.

Original SABER code and package tooling are released under the MIT License.
Third-party dependencies are not redistributed; see
`THIRD_PARTY_NOTICES.md`.

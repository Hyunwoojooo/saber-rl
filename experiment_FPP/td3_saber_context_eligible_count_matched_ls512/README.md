# FPP TD3 SABER Context-Eligible Count-Matched Control

This experiment isolates membership-aligned sample assignment from
context-gated, count-preserving critic reweighting in FetchPickAndPlace-v4.
The completed Main SABER runs under
`experiment_FPP/runs/td3_saber_critic_only_v4_ls512` are reused only after
source, config, seed, and evaluation parity checks pass.

## Control Definition

- `E`: the Main predicate's local-count/context-eligible mask immediately
  before current-policy action membership.
- `M`: `E` intersected with current-policy action-bin membership.
- `k`: `sum(M)`.
- `Z`: exactly `k` indices sampled uniformly without replacement from the
  true indices of `E`.

The control recomputes `E`, counterfactual `M`, and `k` from the current
trajectory at every update. Selection of `Z` uses only `E`, `k`, and the
assignment RNG. It does not use individual `M` positions, policy actions, or
action bins. Natural overlap between `M` and `Z` is allowed.

`k=0` produces an all-false `Z`; `k=|E|` produces `Z=E`; `k>|E|`
raises immediately. Both Main and control keep
`support_query_mode=current_policy`. The only intended difference is
`support_assignment_mode=context_eligible_count_matched`.

## Per-Update Invariants

The implementation fails immediately unless all conditions hold:

- `M` is a subset of `E`.
- `Z` is a subset of `E`.
- `sum(Z) == sum(M) == k <= |E|`.
- Main and control raw-weight histograms are identical.
- Main and control normalized-weight histograms are identical.
- Each normalized weight sum equals batch size within float32 tolerance.

Diagnostics aggregate non-overlapping 1,000-update windows and record `|E|`,
`k`, rates, overlap, overlap/k, Jaccard, weight statistics, mismatch counts,
and invariant failures.

## RNG Isolation

Assignment uses a dedicated `np.random.Generator` with seed `30000+s`.
Replay sampling, network initialization, environment, exploration, weight, and
global NumPy RNGs are not consumed by assignment. The assignment RNG and
diagnostic window state are stored in checkpoints and restored on resume.

Seed mapping remains:

- environment, Torch, NumPy, action noise: `s`
- weight RNG: `20000+s`
- assignment RNG: `30000+s`
- evaluation: `10000+s`

## Fixed Production Settings

- FetchPickAndPlace-v4, sparse reward
- TD3 + HER, 1,000,000 steps, seeds 1-5
- batch size 256, HER ratio 0.8, one update per environment step
- exact evaluation every 10k using 20 episodes
- `K=5`, `M_B=16`, `M_Q_min=4`, `N_local_min=4`
- `local_search_max=512`, `lambda_support=1`
- `weight_scope=critic_only`, `weighting_mode=batch_norm`
- `support_scan_mode=deterministic_tail_v1`
- TD3 policy delay/noise/clip: `2 / 0.2 / 0.5`

## Validation

Pinned scanner dependencies are in `requirements-numba.txt`.

```bash
.venv/bin/python experiment_FPP/td3_saber_context_eligible_count_matched_ls512/code/src/test_context_eligible_count_matched.py
.venv/bin/python experiment_FPP/td3_saber_context_eligible_count_matched_ls512/code/src/test_support_query_modes.py
.venv/bin/python experiment_FPP/td3_saber_context_eligible_count_matched_ls512/code/src/test_current_policy_legacy_parity.py
env PARITY_NUMBA=1 .venv/bin/python experiment_FPP/td3_saber_context_eligible_count_matched_ls512/code/src/test_current_policy_legacy_parity.py
.venv/bin/python experiment_FPP/td3_saber_context_eligible_count_matched_ls512/code/src/test_numba_runtime_opt_fpp.py
.venv/bin/python experiment_FPP/td3_saber_context_eligible_count_matched_ls512/code/src/test_numba_support_scan_fpp.py
.venv/bin/python experiment_FPP/td3_saber_context_eligible_count_matched_ls512/test_run_config_parity.py
.venv/bin/python experiment_FPP/td3_saber_context_eligible_count_matched_ls512/test_analyze_results.py
```

The required sequence is seed-1 5k smoke, an exercised checkpoint resume, then
seed-1 50k technical gate. Gate decisions use only crash/NaN, invariant,
exact-k, logging, RNG isolation, and checkpoint/resume criteria. Performance is
not used. Passing validations are embedded in `freeze_manifest.json`; the
production launcher rejects any later change to frozen source hashes.

## Execution

```bash
.venv/bin/python experiment_FPP/td3_saber_context_eligible_count_matched_ls512/run.py --stage smoke --seed 1 --device cuda
.venv/bin/python experiment_FPP/td3_saber_context_eligible_count_matched_ls512/run.py --stage smoke --seed 1 --device cuda --resume
.venv/bin/python experiment_FPP/td3_saber_context_eligible_count_matched_ls512/run.py --stage gate --seed 1 --device cuda
experiment_FPP/td3_saber_context_eligible_count_matched_ls512/run_all_seeds.sh
```

`run_all_seeds.sh` starts production seeds 1-5 in parallel and records
per-seed logs and exit codes under the orchestration run directory.

## Analysis

The analyzer requires the exact 10k, 20k, ..., 1M grid and never interpolates,
smooths, extrapolates, or forward-fills. AUC is trapezoidal area normalized by
990,000. Hit90 is the first exact checkpoint with success at least 0.90.
Aggregates use arithmetic means and sample standard deviation (`ddof=1`).

The primary comparison is Main SABER minus the count-matched control. Outputs
include 10,000 paired percentile bootstrap resamples with seed 27027, exact
two-sided sign-flip, positive-delta count, and leave-one-seed-out means.
Context-count-only is a count-unmatched descriptive reference. Random-support
is excluded from the primary table.

```bash
.venv/bin/python experiment_FPP/td3_saber_context_eligible_count_matched_ls512/analyze_results.py \
  --repo-root . \
  --output-dir experiment_FPP/reports/td3_saber_context_eligible_count_matched_ls512
```

## Artifact

```bash
.venv/bin/python experiment_FPP/td3_saber_context_eligible_count_matched_ls512/package_artifact.py \
  --repo-root . \
  --output-dir experiment_FPP/paper_artifacts/fpp_td3_saber_context_eligible_count_matched
```

The compact artifact contains sanitized configs, environment snapshots,
summaries, exact-grid selected metrics, control diagnostics, analyzer/tests,
technical-gate evidence, and SHA256 manifests. Checkpoints, full metrics,
JSONL, and logs are excluded. Selected performance CSVs are checked for
byte-level identity before and after packaging.

Interpretation is restricted to FPP TD3. No universal mechanism or
task-agnostic causal claim is made.

# FPP–DDPG SABER Strict Context-Eligible Count-Matched Control

This package implements the official Strict Context-Eligible Count-Matched
Control for the canonical FetchPickAndPlace DDPG Main SABER experiment.

## Scientific contract

The canonical reference is
experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/.

For each minibatch, the control first computes Main SABER's current-policy
support result:

- E is the canonical context-population eligible mask.
- M is E intersected with current-policy action-bin membership.
- k is the number of positive positions in M.

It then samples exactly k positions uniformly without replacement from E and
calls that mask Z. Main-positive positions remain ordinary candidates; they are
not excluded or forced disjoint from Z. Raw weights are 1 + lambda_support on Z
and one elsewhere, followed by canonical minibatch mean-one normalization.
The resulting weights apply only to the DDPG critic loss. Every actor
coefficient remains one.

The implementation fails immediately unless every update satisfies:

- M is a subset of E, Z is a subset of E, and sum(Z) equals sum(M).
- Main and control raw-weight histograms match exactly.
- Main and control normalized-weight histograms match exactly.
- Normalized weights are positive, have mean one, and sum to batch size.
- Actor weights are all one.

k=0 and k=|E| are deterministic and consume no assignment RNG. Assignment uses
a dedicated NumPy Generator with seed 30000 + experiment seed. It is
independent of replay, environment, exploration, weight, global NumPy, and
PyTorch RNGs. Its state and the diagnostic-window accumulator are mandatory
checkpoint fields; strict resume fails if either is absent.

## Matched configuration

| Field | Main SABER | Strict control |
|---|---|---|
| support_query_mode | implicit current_policy | current_policy |
| support_assignment_mode | implicit support | context_eligible_count_matched |

Control-only bookkeeping is fixed to assignment_seed=30000+seed and
assignment_diagnostic_window_updates=1000. Identifiers (folder, run_slug,
output_root, purpose/reference metadata) also differ.
validate_config_parity.py rejects every other config or production-source
drift.

The matched backbone remains DDPG, FetchPickAndPlace-v4, sparse future-goal HER
at 0.8, horizon 50, uniform replay, batch 256, capacity 300k, warmup 5k,
random actions 10k, UTD 1, 256 by 256 networks, actor/critic LR 3e-4,
gamma 0.98, tau 0.005, exploration noise 0.2, critic-only batch-normalized
SABER, LS512, deterministic-tail support scan, and deterministic 20-episode
evaluation every 10k steps. Success relabels are nonterminal and time limits
bootstrap.

This is an assignment control, not replay prioritization, PER, HGR, or a query
ablation. It recomputes its own counterfactual Main mask on its own evolving
trajectory and never imports a mask or count sequence from a Main run.

## Paths

- Source: experiment_FPP/ddpg_saber_context_eligible_count_matched_ls512/
- 5k smoke: experiment_FPP/runs/ddpg_saber_context_eligible_count_matched_smoke_5k/
- 50k gate: experiment_FPP/runs/ddpg_saber_context_eligible_count_matched_gate_50k/
- 1M results: experiment_FPP/runs/ddpg_saber_context_eligible_count_matched_ls512/
- Artifact: experiment_FPP/paper_artifacts/fpp_ddpg_saber_context_eligible_count_matched_ls512_1m/

## Validation and execution

Run the complete static, unit, and synthetic suite:

    PYTHON_BIN=.venv/bin/python bash experiment_FPP/ddpg_saber_context_eligible_count_matched_ls512/run_tests.sh

Then run the preregistered technical stages in order:

    DEVICE=cuda bash experiment_FPP/ddpg_saber_context_eligible_count_matched_ls512/run_smoke_5k_seed1.sh
    DEVICE=cuda bash experiment_FPP/ddpg_saber_context_eligible_count_matched_ls512/run_technical_gate_50k_seed1.sh

The 5k smoke has one final evaluation at step 5,000. Both the 5k smoke and 50k
gate explicitly resume their completed checkpoints and validate restored
assignment RNG/diagnostic state. The 50k gate also requires zero assignment
invariant or histogram mismatches and writes freeze_manifest.json. Performance
is not a gate criterion. Each stage records a deterministic full-package
source fingerprint before training and verifies it again after training. Gate
startup requires the smoke fingerprint to match its current source, and the
freeze is created only when smoke, gate, and current file maps/tree hashes are
identical. Smoke/gate outputs are isolated from production.

Commit and push the frozen source, including freeze_manifest.json, before
production. Because experiment_FPP/.gitignore ignores analysis directories,
force-add this package's analysis files and confirm every path named by the
freeze manifest is tracked. From a clean main checkout with HEAD equal to
origin/main:

    MAX_PARALLEL=5 DEVICE=cuda bash experiment_FPP/ddpg_saber_context_eligible_count_matched_ls512/run_main_1m_seeds1_5_parallel.sh

The launcher refuses existing exact or timestamp-suffixed targets, records the
source commit/tree/config/freeze provenance, and verifies before launching that
every frozen path exists in HEAD with byte-identical SHA256 content. It records
and rechecks a production source fingerprint, runs seeds 1–5 in distinct
directories, and requires all five completion gates. MAX_PARALLEL may be 1–5;
choose it only from machine capacity, not observed performance.

If an already-registered production seed is interrupted, resume that seed from
its canonical checkpoint without changing any setting:

    python3 experiment_FPP/ddpg_saber_context_eligible_count_matched_ls512/run.py --stage production --seed N --device cuda --resume

The completion validator accepts either uninterrupted or resumed provenance
and still requires the exact step/update/evaluation/diagnostic state.

The old 20k, screening, and generic all-seed launchers are deliberate
fail-fast compatibility stubs and cannot start an alternate experiment.

## Results, analysis, and artifact

Each 1M seed must contain 995,001 updates, the exact 100-point 10k–1M
evaluation grid, checkpoints through 1M, complete 1,000-update assignment
diagnostics (including the final one-update window), finite critical
loss/evaluation/assignment values, and zero invariant/histogram mismatch
counters. Canonical optional diagnostic NaN sentinels are allowed only in
their known raw telemetry fields; critical NaN or Inf fails. During packaging,
optional raw NaNs become JSON null, and the packaged verifier remains strict.

Primary analysis is normalized trapezoidal success AUC on exact observed
10k–1M points divided by 990,000, with no smoothing, filling, interpolation,
extrapolation, or seed exclusion. It reports per-seed AUC/final/best/Hit90,
mean plus/minus sample SD, Main minus Strict paired differences, a
10,000-resample paired bootstrap CI with seed 27027, the exact two-sided
32-assignment sign-flip test, positive count, leave-one-seed-out results,
exact-grid mean curves, and assignment diagnostics.

    python3 experiment_FPP/ddpg_saber_context_eligible_count_matched_ls512/analysis/analyze_results.py
    python3 experiment_FPP/ddpg_saber_context_eligible_count_matched_ls512/analysis/build_artifact.py
    python3 experiment_FPP/paper_artifacts/fpp_ddpg_saber_context_eligible_count_matched_ls512_1m/code/verify_artifact.py --artifact-root experiment_FPP/paper_artifacts/fpp_ddpg_saber_context_eligible_count_matched_ls512_1m

The compact artifact contains frozen code/config, execution provenance and
hashes, Main/control compact seed data, exact selected metrics, every strict
assignment window, technical-gate evidence, deterministic analysis outputs,
test evidence, and a standalone hash/recomputation verifier. It excludes
checkpoints, full per-update metrics, JSONL, raw logs, and raw smoke/gate
directories. Packaging never starts training.

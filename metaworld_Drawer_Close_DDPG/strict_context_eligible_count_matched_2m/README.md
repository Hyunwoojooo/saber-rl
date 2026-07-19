# DrawerClose DDPG-HER Strict Context-Eligible Count-Matched Control

This directory contains an isolated DrawerClose DDPG implementation of the
strict context-eligible count-matched control. It is a control for SABER's
sample assignment, not a new replay sampler and not the earlier
context-count-only control.

For each uniformly sampled and HER-relabeled minibatch, the unchanged SABER
Main computation produces:

- `M`: positions whose current-policy action bin is supported by a sufficiently
  populated same-context replay set.
- `E`: positions satisfying the same context-population condition without
  consulting current-policy action-bin membership.

The control draws exactly `k = |M|` positions uniformly without replacement
from `E` using a dedicated, checkpointed RNG with
`assignment_seed = 30000 + seed`. Main positions are not excluded, so natural
Main/control overlap is allowed. Runtime invariants enforce `M subset E`,
`C subset E`, and `|C| = |M|`. The resulting raw-weight histogram is therefore
identical to Main. Weights are normalized to minibatch mean one and applied
only to the DDPG critic loss; the actor loss remains unweighted.

## Matched production contract

The production control matches
`metaworld_Drawer_Close_DDPG/DDPG-HER-B3-Critic-Only-2M.sh` in every learning
and evaluation setting except run identity/output and the strict assignment,
RNG, diagnostic, and checkpoint-audit fields required by this control.

- task/environment: `drawer_close` / `drawer-close-v3`
- budget: 2,000,000 environment steps, seeds 1--5
- horizon: 200
- replay/HER: capacity 300,000, batch 256, future HER ratio 0.8
- optimization: UTD 1, gamma 0.98, tau 0.005, actor/critic LR 3e-4
- evaluation: every 10,000 steps, 20 deterministic episodes
- SABER support: `M_B=16`, `M_Q_min=4`, `N_local_min=4`, scan 512,
  five action bins per dimension, `lambda_support=1`
- scope: critic-only, minibatch-mean-normalized weights
- scanner runtime: NumPy 2.3.5, Numba 0.66.0, llvmlite 0.48.0; the
  production gate rejects Python fallback

`validate_config_parity.py` constructs Main and strict configs through the same
DrawerClose parser for all five seeds and rejects every non-allowlisted
difference. `validate_source_parity.py` verifies that the DDPG agent, networks,
replay/HER, environment wrapper, DrawerClose task definition, and I/O code are
byte-identical to the existing DrawerClose source.

## Non-training validation

Run from the repository root:

```bash
.venv/bin/python -m pip install --target /tmp/saber-numba-0.66.0 \
  -r metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/requirements-numba.txt
export PYTHONPATH=/tmp/saber-numba-0.66.0${PYTHONPATH:+:$PYTHONPATH}
```

The isolated target avoids replacing the repository virtual environment's
NumPy installation. Executable modes verify all three pinned versions before
starting; integration and completion gates also require the Numba scanner to
be warmed with zero fallback.

```bash
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/validate_config_parity.py --output metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/config_parity.json
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/validate_source_parity.py --output metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/source_parity.json
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/audit_checkpoint_resume.py --output metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/checkpoint_resume_audit.json
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/test_config_parity.py
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/code/src/test_strict_context_eligible_count_matched.py
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/code/src/test_strict_drawer_close_scanner.py
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/code/src/test_strict_ddpg_loss_scope.py
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/code/src/test_strict_config_resume_contract.py
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/run.py --mode main2m --seeds 1 --validate-only
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/run.py --mode main2m --dry-run
```

These tests are executable scripts rather than `unittest.TestCase` classes;
run them directly as shown above.

## Correctness, smoke, and production

Run the short gates before the 2M cohort:

```bash
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/run.py --mode correctness5k
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/validate_integration.py --mode correctness5k
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/run.py --mode smoke20k_seed1
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/validate_integration.py --mode smoke20k_seed1
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/run.py --mode smoke20k_seeds1_5
.venv/bin/python metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/validate_integration.py --mode smoke20k_seeds1_5
```

After committing the final source, the validated pipeline runs all gates,
freezes the source, and launches production:

```bash
bash metaworld_Drawer_Close_DDPG/strict_context_eligible_count_matched_2m/run_validated_pipeline.sh
```

Set `WAIT_FOR_TMUX_SESSION` to defer the pipeline until another tmux session
ends. Production results are written only to:

```text
metaworld_Drawer_Close_DDPG/results_2m/
  DDPG-HER-B3-Strict-Context-Eligible-Count-Matched-2M/
```

The launcher refuses to overwrite existing runs. Resume requires a matching
config fingerprint, complete assignment and global RNG state, strict diagnostic
state, and an episode-boundary checkpoint.

## Completion and artifact

After all five seeds reach 2M, `validate_completion.py` checks exact update and
evaluation coverage, all strict invariants, config/source parity, environment
identity, checkpoint state, and the frozen source. `analyze_results.py` compares
DrawerClose HER, SABER Main, and the strict control on the exact observed
10k--2M env-success grid.

The compact artifact is written to:

```text
metaworld_Drawer_Close_DDPG/paper_artifacts/
  drawer_close_ddpg_saber_context_eligible_count_matched_2m/
```

It excludes checkpoints, full raw metrics, JSONL, logs, correctness runs, and
smoke runs. Correctness and smoke performance must not be used in the paper.

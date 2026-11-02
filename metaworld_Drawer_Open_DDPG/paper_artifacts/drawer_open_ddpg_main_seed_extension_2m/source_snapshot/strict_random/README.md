# DDPG-HER-B3-Strict-Context-Eligible-Count-Matched-2M

This directory contains the isolated DrawerOpen DDPG implementation of the
strict context-eligible count-matched control. It is a minimal transplant of
the validated FPP implementation in
`experiment_FPP/td3_saber_context_eligible_count_matched_ls512/`; it is not the
context-count-only control.

For every relabeled minibatch, the code first computes the unchanged SABER
Main mask `M` and the membership-independent context-eligible mask `E`. It then
draws exactly `k = |M|` positions uniformly without replacement from `E` using
the checkpointed `assignment_seed = 30000 + seed` RNG. Main positions are not
excluded, so natural Main/control overlap is allowed. The implementation fails
immediately unless `M subset E`, `C subset E`, and `|C| = |M|`. The raw control
weight is `1 + C`, is normalized to minibatch mean one, and is passed only to
the DDPG critic loss. The actor loss remains unweighted and Bellman regression
uses replay actions.

The production recipe is identical to
`metaworld_Drawer_Open_DDPG/DDPG-HER-B3-Critic-Only-2M.sh` except for the run
identity/output, strict assignment mode, dedicated assignment RNG, and strict
diagnostics. `validate_config_parity.py` produces a recursive, allowlisted diff
for all five seeds and refuses any other change.

## Required execution order

From the repository root:

```bash
.venv/bin/python metaworld_Drawer_Open_DDPG/strict_context_eligible_count_matched_2m/validate_config_parity.py
.venv/bin/python -m unittest discover metaworld_Drawer_Open_DDPG/strict_context_eligible_count_matched_2m/code/src -p 'test_*.py'
.venv/bin/python metaworld_Drawer_Open_DDPG/strict_context_eligible_count_matched_2m/run.py --mode correctness5k
.venv/bin/python metaworld_Drawer_Open_DDPG/strict_context_eligible_count_matched_2m/run.py --mode smoke20k_seed1
.venv/bin/python metaworld_Drawer_Open_DDPG/strict_context_eligible_count_matched_2m/run.py --mode smoke20k_seeds1_5
.venv/bin/python metaworld_Drawer_Open_DDPG/strict_context_eligible_count_matched_2m/run.py --mode main2m
```

The correctness and smoke outputs are kept under this experiment directory and
are never included in paper analysis. Production results use the collision-free
root:

```text
metaworld_Drawer_Open_DDPG/results_2m/
  DDPG-HER-B3-Strict-Context-Eligible-Count-Matched-2M/
```

Do not use partial production runs for analysis. Resume is accepted only when
the checkpoint contains the strict assignment RNG/diagnostic state, the full
RNG audit state, a matching config fingerprint, and an episode-boundary marker.

## Completion and artifact

After all seeds reach 2M, run the completion validator and analyzer supplied in
this directory, then build and independently verify:

```text
metaworld_Drawer_Open_DDPG/paper_artifacts/
  drawer_open_ddpg_saber_context_eligible_count_matched_2m/
```

The artifact contains only exact-grid compact data, configs, environment and
strict diagnostics, analysis tables, manifests, and frozen source. It excludes
checkpoints, raw metrics/JSONL, logs, correctness runs, and smoke runs.

No performance conclusion is made from correctness or smoke runs. No commit or
push is performed by these scripts.

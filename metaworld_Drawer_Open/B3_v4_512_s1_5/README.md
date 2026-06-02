# MetaWorld Drawer-Open-v3: B3_v4_512_s1_5

This snapshot ports the FPP main method `B3_v4_512_s1_5` to MetaWorld
`drawer-open-v3`.

## Preserved Algorithm Settings

- Method: `b3_support_runtime_opt`
- Weight scope: `critic_only`
- `local_search_max = 512`
- `lambda_support = 1.0`
- `N_local_min = 4`
- `M_Q_min = 4`
- `M_B = 16`
- Weighting mode: `batch_norm`
- Support cache enabled
- Local scan cache enabled
- Exact Numba support scan enabled
- Predicate early exit enabled
- Exact negative short-circuit enabled

## Task-Specific Changes

Only the task abstraction is changed from FetchPickAndPlace to MetaWorld
drawer-open-v3.

The drawer-open support context is:

- coarse key: hand-handle distance bin, handle-y bin, gripper bin, horizon bin
- full context key: phase, hand-handle distance bin, handle-goal distance bin,
  handle-y bin, gripper bin, horizon bin

Phases:

- `approach`: hand is far from the drawer handle
- `contact`: hand is near the drawer handle
- `opening`: hand is near and handle-goal distance is below `drawer_d_opening_mid`
- `open`: handle-goal distance is below `drawer_d_goal_near`

## Main Files

- `code/src/train_drawer_open_lite_v0.py`
- `code/src/metacog_lite_v0/envs.py`
- `code/src/metacog_lite_v0/task_drawer_open.py`
- `code/src/metacog_lite_v0/numba_support_scan_drawer_open.py`
- `code/src/metacog_lite_v0/weights.py`
- `run.py`
- `config.json`

## Validation

After installing MetaWorld, run:

```bash
python run.py --seed 1 --steps 1000 --device cpu --validate_only
```

`--validate_only` is a hard gate for experiment validity. It prints a JSON
diagnostic report, then raises `RuntimeError` if any required check fails.
The required checks include:

- Drawer desired-goal source must come from MetaWorld environment metadata
  (`_target_pos`, `target_pos`, `_goal`, or `goal`). `obs_tail`,
  `synthetic_fallback`, and `unknown` are treated as invalid.
- B0 sanity path must produce unit actor/critic weights.
- B4 sanity path must produce finite support weights and finite `G_sup`.
- B3 runtime-opt path must produce finite critic weights and finite support
  diagnostics.
- Runtime-opt diagnostics must confirm that Numba support scan metrics,
  predicate early-exit metrics, and Numba scan timing are actually present.
- HER relabeling, replay indexing, raw reward finiteness, and local context
  timing diagnostics must be present.

This is intentional: if the Drawer-Open GoalEnv conversion or runtime-opt
support lookup silently falls back to an invalid path, the run should fail
before a long experiment starts.

Additional Drawer diagnostics are logged for analysis:

- `hand_handle_dist_mean`
- `handle_goal_dist_mean`
- `phase_approach_frac`
- `phase_contact_frac`
- `phase_opening_frac`
- `phase_open_frac`

For a short smoke run:

```bash
python run.py --seed 1 --steps 100000 --device cuda
```

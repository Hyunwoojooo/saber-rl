# FetchPickAndPlace DDPG+HER+SABER Actor-Only v4 LS512

This package is a strict matched one-factor loss-scope ablation of the completed
FetchPickAndPlace DDPG+HER+SABER v4 LS512 experiment.

## Scientific contract

The reference arm is:

`experiment_FPP/b3_critic_only_runtime_opt_v4_ls512`

Only these six `config.json` fields may differ from that reference:

| Field | Actor-only value |
|---|---|
| `folder` | `ddpg_saber_actor_only_v4_ls512` |
| `output_root` | `experiment_FPP/runs/ddpg_saber_actor_only_v4_ls512` |
| `purpose` | actor-only scope description |
| `reference` | `experiment_FPP/b3_critic_only_runtime_opt_v4_ls512` |
| `run_slug` | `ddpg_saber_actor_only_v4_ls512` |
| `weight_scope` | `actor_only` |

In particular, the trainer remains `b3_support_runtime_opt`, with
`local_search_max=512`, `lambda_support=1.0`, batch-normalized weights, HER ratio
0.8, one update per environment step, and seeds 1 through 5. Production Python
under `code/src` must remain byte-for-byte identical to the reference snapshot;
only package-local validation tests may be added.

Run `validate_config_parity.py` or `run_tests.sh` before launching. The parity
validator rejects any extra configuration or production-source change.

## Provenance

- Package preparation base: repository SHA `77b5e56eb0712497b55efac6dd0cfd6d841b2670`
- Canonical reference source/results origin: commit
  `c43d40502b84ae713924ebf995be623e7acc7ca8`

The first SHA records the repository state from which this new arm was prepared.
The second is the original commit containing the completed canonical v4 LS512
critic-only experiment copied into the current reference path.

## Paths and run names

- Package: `experiment_FPP/ddpg_saber_actor_only_v4_ls512`
- Frozen source: `experiment_FPP/ddpg_saber_actor_only_v4_ls512/code/src`
- Main results: `experiment_FPP/runs/ddpg_saber_actor_only_v4_ls512`
- 20k smoke results: `experiment_FPP/runs/ddpg_saber_actor_only_v4_ls512_smoke20k`
- Run name: `fetch_pick_and_place_{run_slug}_s{seed}_{steps}_ls{local_search_max}`

`run.py` discovers the repository root from `.git` plus `experiment_FPP`, honors
`PYTHON_BIN` before the repository virtual environment, accepts only registered
seeds 1 through 5, and refuses to create timestamp-suffixed duplicates when the
exact target run directory already exists.

## Tests

```bash
bash experiment_FPP/ddpg_saber_actor_only_v4_ls512/run_tests.sh
```

## 20k smoke run

This is a real 20,000-step run, not the trainer's `--smoke` mode:

```bash
DEVICE=cuda bash experiment_FPP/ddpg_saber_actor_only_v4_ls512/run_smoke_20k_seed1.sh
```

The strict runner intentionally has no `--smoke` option. Short runs must use
`--steps`, so the run-directory step tag always equals the actual training
budget.

## Main 1M runs, seeds 1-5

```bash
MAX_PARALLEL=2 DEVICE=cuda \
  bash experiment_FPP/ddpg_saber_actor_only_v4_ls512/run_main_1m_seeds1_5_parallel.sh
```

`MAX_PARALLEL` defaults to 2. The launcher fixes CPU math-library thread counts
to 1, writes one orchestration log per seed, preflights every target directory,
and validates the completed results after all five processes succeed. Immediately
before launching, it records `HEAD`, `origin/main`, and the pre-launch worktree
cleanliness flag in
`experiment_FPP/runs/ddpg_saber_actor_only_v4_ls512/_execution/launch_metadata.json`.
It refuses to start training unless `HEAD` exactly matches `origin/main`; it does
not fetch or update either ref. It also refuses a dirty pre-launch worktree after
recording that state in the same metadata file.

## Direct validation and dry-run

```bash
python3 experiment_FPP/ddpg_saber_actor_only_v4_ls512/run.py --seed 1 --validate_only
python3 experiment_FPP/ddpg_saber_actor_only_v4_ls512/run.py --seed 1 --steps 20000 --dry-run
python3 experiment_FPP/ddpg_saber_actor_only_v4_ls512/run.py --seed 1 --verify_runtime_optimization --dry-run
```

## No resume

Resume is intentionally unsupported for this arm. The frozen canonical trainer
does not preserve a fully exact resume trajectory and contains legacy step
bookkeeping that can skip a nominal transition. Do not add `--resume`, reuse a
partially populated run directory, or resume from another loss scope. Restart a
failed seed into a clean target only.

## Legacy launchers

`run_all_seeds.sh` and `run_screening_3seeds_parallel.sh` are retained only as
compatibility stubs. Both print the approved strict launcher paths and exit 2;
they cannot start arbitrary screening or tuning jobs. Use
`run_smoke_20k_seed1.sh` and `run_main_1m_seeds1_5_parallel.sh` instead.

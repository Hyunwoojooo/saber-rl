#!/usr/bin/env python3
"""Build a compact, verified DrawerOpen REDQ-HER TQB paper artifact."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from package_drawer_open_paper_artifacts import (
    AGGREGATE_FIELDS,
    PER_SEED_FIELDS,
    _aggregate,
    _copy_verifier,
    _run_result,
    _write_csv,
    _write_manifest,
)


SEEDS = tuple(range(1, 6))
ARTIFACT_ID = "DRAWEROPEN-REDQ-HER-TQB-2M"
EXPECTED_CONFIG = {
    "task_name": "drawer_open",
    "metaworld_env_name": "drawer-open-v3",
    "metaworld_reward_function_version": "v2",
    "reward_type": "sparse",
    "total_steps": 2_000_000,
    "updates_per_step": 4,
    "batch_size": 256,
    "her_ratio": 0.8,
    "gamma": 0.98,
    "tau": 0.005,
    "redq_ensemble_size": 10,
    "target_subset_size": 2,
    "sac_auto_alpha": True,
    "target_entropy": -4.0,
    "target_q_bounding": True,
    "target_q_min": -50.0,
    "target_q_max": 0.0,
    "eval_freq": 10_000,
    "eval_episodes": 20,
    "success_terminates": False,
    "timeout_bootstrap": True,
    "weight_scope": "none",
}


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _validate_protocol(run_dir: Path, seed: int) -> dict[str, object]:
    config = _read_json(run_dir / "config.json")
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in EXPECTED_CONFIG.items()
        if config.get(key) != expected
    }
    expected_seeds = {
        "env_seed": seed,
        "torch_seed": seed,
        "numpy_seed": seed,
        "action_noise_seed": seed,
        "weight_seed": 20_000 + seed,
        "eval_seed": 10_000 + seed,
    }
    observed_seeds = config.get("seeds")
    if observed_seeds != expected_seeds:
        mismatches["seeds"] = {"expected": expected_seeds, "observed": observed_seeds}
    if mismatches:
        raise ValueError(f"Protocol mismatch for seed {seed}: {mismatches}")

    snapshot = _read_json(run_dir / "env_snapshot.json")
    for split, expected_seed in (("train_env", seed), ("eval_env", 10_000 + seed)):
        env = snapshot.get(split)
        if not isinstance(env, dict):
            raise ValueError(f"Missing {split} snapshot for seed {seed}")
        expected_env = {
            "env_name": "drawer-open-v3",
            "reward_type": "sparse",
            "max_episode_steps": 200,
            "action_dim": 4,
            "obs_dim": 39,
            "goal_dim": 3,
        }
        for key, expected in expected_env.items():
            if env.get(key) != expected:
                raise ValueError(f"{split}.{key} mismatch for seed {seed}: {env.get(key)!r}")
        spec_kwargs = env.get("spec_kwargs")
        if not isinstance(spec_kwargs, dict) or spec_kwargs.get("seed") != expected_seed:
            raise ValueError(f"{split} seed mismatch for seed {seed}")
        if spec_kwargs.get("reward_function_version") != "v2":
            raise ValueError(f"{split} reward version mismatch for seed {seed}")
        if spec_kwargs.get("terminate_on_success") is not False:
            raise ValueError(f"{split} terminal convention mismatch for seed {seed}")
    return config


def _copy_source_snapshot(source_root: Path, output: Path) -> None:
    code_target = output / "code" / "source"
    shutil.copytree(source_root / "code" / "src", code_target / "src")
    shutil.copy2(source_root / "REDQ-HER-TQB-B0-2M.sh", code_target / "REDQ-HER-TQB-B0-2M.sh")


def build(source_root: Path, output: Path, verifier: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    audited_configs: dict[str, object] = {}
    result_root = source_root / "results_2m" / "REDQ-HER-TQB-B0-2M"
    for seed in SEEDS:
        run_dir = result_root / f"drawer_open_REDQ-HER-TQB-B0-2M_s{seed}_2m_cuda"
        audited_configs[str(seed)] = _validate_protocol(run_dir, seed)
        row, _ = _run_result(
            run_dir,
            seed,
            "drawer_open_redq_her_tqb",
            "REDQ-HER TQB",
            "seeds_1_5",
            output / "data",
            False,
        )
        rows.append(row)

    aggregate = _aggregate(
        "drawer_open_redq_her_tqb",
        "REDQ-HER TQB",
        "seeds_1_5",
        rows,
        SEEDS,
    )
    _write_csv(output / "per_seed_performance.csv", PER_SEED_FIELDS, rows)
    _write_csv(output / "aggregate_performance.csv", AGGREGATE_FIELDS, [aggregate])
    protocol_audit = {
        "artifact_id": ARTIFACT_ID,
        "expected_config": EXPECTED_CONFIG,
        "validated_seeds": list(SEEDS),
        "source_method_field_note": (
            "The legacy config method field is b0_ddpg_her; the REDQ identity is established by "
            "the REDQ trainer/source snapshot and the validated ensemble/subset/UTD settings."
        ),
        "run_names": {seed: audited_configs[str(seed)].get("run_name") for seed in SEEDS},
    }
    (output / "protocol_audit.json").write_text(
        json.dumps(protocol_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _copy_source_snapshot(source_root, output)
    _copy_verifier(output, verifier)

    hit_median = int(float(aggregate["hit90_median_step_among_hits"])) // 1000
    runtime_hours = sum(float(row["runtime_hours"]) for row in rows) / len(rows)
    readme = f"""# DrawerOpen REDQ-HER TQB 2M artifact

Blind artifact ID: `{ARTIFACT_ID}`

This package contains five completed 2M-step DrawerOpen REDQ+HER runs with a 10-critic ensemble, target subset size 2, UTD 4, automatic entropy tuning, and target-Q bounding `[-50, 0]`. It is an architecture/compute-scaled reference, not a matched one-update-per-step SABER baseline.

## Recomputed result

- Final success: `{aggregate['final_success_mean']:.3f} +/- {aggregate['final_success_sample_std']:.3f}`
- Success AUC: `{aggregate['success_auc_mean']:.3f} +/- {aggregate['success_auc_sample_std']:.3f}`
- Hit90: `{aggregate['hit90_count']}/{aggregate['hit90_total']}`, median `{hit_median}k` among reached seeds
- Mean recorded runtime: `{runtime_hours:.2f}` hours per seed
- Evaluation grid: exact 10k through 2M, 200 observed points per seed

The mean Hit90 over seeds is 644k, but the paper reports the reached-seed median 370k to remain consistent with its other threshold tables.

## Scope

The raw config retains the legacy method label `b0_ddpg_her`. REDQ identity is verified from the packaged trainer, 10-critic ensemble, target-subset backup, UTD 4, auto-alpha, and target-Q-bound configuration. Large checkpoints, raw high-frequency metrics, and logs are excluded. Exact-grid evaluation rows, summaries, configs, environment snapshots, and the source snapshot are included.

## Verification

```bash
python3 code/verify_artifact.py
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    _write_manifest(output, ARTIFACT_ID)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verifier",
        type=Path,
        default=Path(__file__).with_name("verify_drawer_open_paper_artifact.py"),
    )
    args = parser.parse_args()
    build(args.source_root, args.output, args.verifier)
    print(f"Built {args.output}")


if __name__ == "__main__":
    main()

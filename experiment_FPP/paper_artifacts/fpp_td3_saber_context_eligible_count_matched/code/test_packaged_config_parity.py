#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping


SEEDS = (1, 2, 3, 4, 5)
EVAL_STEPS = tuple(range(10_000, 1_000_001, 10_000))
PRIMARY_SUCCESS_FIELD = "eval/env_success_mean"
FALLBACK_SUCCESS_FIELD = "eval/success_mean"
METHOD_MODES = {
    "main": "current_policy",
    "control": "current_policy",
    "context_only": "context_count_only",
}
ASSIGNMENT_MODES = {"main": "support", "control": "context_eligible_count_matched", "context_only": "support"}
CONFIG_IGNORE_FIELDS = {
    "run_name",
    "output_root",
    "resume",
    "resume_path",
    "support_query_mode",
    "support_assignment_mode",
    "assignment_diagnostic_window_updates",
    "thread_hint",
}
COMMON_FIXED_FIELDS = {
    "total_steps": 1_000_000,
    "batch_size": 256,
    "her_ratio": 0.8,
    "updates_per_step": 1,
    "eval_freq": 10_000,
    "eval_episodes": 20,
    "env_id": "FetchPickAndPlace-v4",
    "reward_type": "sparse",
    "start_steps": 10_000,
    "warmup_steps": 5_000,
    "buffer_capacity": 300_000,
}
SABER_FIXED_FIELDS = {
    "local_search_max": 512,
    "M_B": 16,
    "M_Q_min": 4,
    "N_local_min": 4,
    "action_bins_per_dim": 5,
    "lambda_support": 1.0,
    "weight_scope": "critic_only",
    "weighting_mode": "batch_norm",
    "support_scan_mode": "deterministic_tail_v1",
}


def finite_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def row_step(row: Mapping[str, str]) -> int | None:
    for key in ("step", "global_step", "env_step"):
        value = finite_float(row.get(key))
        if value is not None:
            return int(value)
    return None


def row_success(row: Mapping[str, str]) -> float | None:
    value = finite_float(row.get(PRIMARY_SUCCESS_FIELD))
    return value if value is not None else finite_float(row.get(FALLBACK_SUCCESS_FIELD))


def seed_for_run(run_dir: Path) -> int:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return int(config["seeds"]["env_seed"])


def discover_runs(root: Path) -> dict[int, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"missing packaged method directory: {root}")
    found: dict[int, Path] = {}
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        required = (
            "config.json",
            "env_snapshot.json",
            "summary.json",
            "metrics_selected_10k.csv",
        )
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{run_dir}: missing {missing}")
        seed = seed_for_run(run_dir)
        if seed in found:
            raise AssertionError(f"{root}: duplicate seed {seed}")
        found[seed] = run_dir
    if set(found) != set(SEEDS):
        raise AssertionError(
            f"{root}: expected seeds {list(SEEDS)}, found {sorted(found)}"
        )
    return {seed: found[seed] for seed in SEEDS}


def validate_exact_grid(metrics_path: Path) -> None:
    values: dict[int, float] = {}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = row_step(row)
            success = row_success(row)
            if step is None or success is None:
                continue
            previous = values.get(step)
            if previous is not None and not math.isclose(
                previous, success, rel_tol=0.0, abs_tol=1e-12
            ):
                raise AssertionError(
                    f"{metrics_path}: evaluation rows disagree at step {step}"
                )
            values[step] = success
    if tuple(sorted(values)) != EVAL_STEPS:
        missing = sorted(set(EVAL_STEPS) - set(values))
        extra = sorted(set(values) - set(EVAL_STEPS))
        raise AssertionError(f"{metrics_path}: missing={missing}, extra={extra}")


def canonical(config: Mapping[str, object]) -> dict[str, object]:
    result = {
        key: value for key, value in config.items() if key not in CONFIG_IGNORE_FIELDS
    }
    seeds = dict(result.get("seeds", {}))
    seeds.pop("assignment_seed", None)
    result["seeds"] = seeds
    return result


def validate_seed_mapping(config: Mapping[str, object], seed: int, method: str) -> None:
    expected = {
        "env_seed": seed,
        "torch_seed": seed,
        "numpy_seed": seed,
        "action_noise_seed": seed,
        "weight_seed": 20_000 + seed,
        "eval_seed": 10_000 + seed,
    }
    actual = dict(config.get("seeds", {}))
    for field, value in expected.items():
        if actual.get(field) != value:
            raise AssertionError(
                f"{method} seed {seed}: {field}={actual.get(field)!r}, "
                f"expected {value!r}"
            )
    assignment_seed = actual.get("assignment_seed")
    if method == "control" and assignment_seed != 30_000 + seed:
        raise AssertionError(
            f"control seed {seed}: assignment_seed={assignment_seed!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate packaged TD3 SABER count-matched configs and grids."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()

    methods = {
        key: discover_runs(artifact_root / "data" / key)
        for key in ("main", "control", "context_only")
    }
    configs: dict[str, dict[int, dict[str, object]]] = {}
    for method, runs in methods.items():
        configs[method] = {}
        for seed, run_dir in runs.items():
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            configs[method][seed] = config
            validate_seed_mapping(config, seed, method)
            for field, expected in COMMON_FIXED_FIELDS.items():
                if config.get(field) != expected:
                    raise AssertionError(
                        f"{method} seed {seed}: {field}={config.get(field)!r}, "
                        f"expected {expected!r}"
                    )
            for field, expected in SABER_FIXED_FIELDS.items():
                if config.get(field) != expected:
                    raise AssertionError(
                        f"{method} seed {seed}: {field}={config.get(field)!r}, "
                        f"expected {expected!r}"
                    )
            validate_exact_grid(run_dir / "metrics_selected_10k.csv")
            diagnostics = run_dir / "assignment_diagnostics_1000.csv"
            if method == "control" and not diagnostics.is_file():
                raise FileNotFoundError(f"{run_dir}: missing {diagnostics.name}")

    for method, expected_mode in METHOD_MODES.items():
        for seed, config in configs[method].items():
            actual = config.get("support_query_mode", "current_policy")
            if actual != expected_mode:
                raise AssertionError(
                    f"{method} seed {seed}: mode={actual!r}, expected {expected_mode!r}"
                )
            assignment = config.get("support_assignment_mode", "support")
            if assignment != ASSIGNMENT_MODES[method]:
                raise AssertionError(
                    f"{method} seed {seed}: assignment={assignment!r}, "
                    f"expected {ASSIGNMENT_MODES[method]!r}"
                )

    for seed in SEEDS:
        reference = canonical(configs["main"][seed])
        for method in ("control", "context_only"):
            candidate = canonical(configs[method][seed])
            if candidate != reference:
                differences = sorted(
                    key
                    for key in set(reference) | set(candidate)
                    if reference.get(key) != candidate.get(key)
                )
                raise AssertionError(
                    f"{method} seed {seed}: non-query config differences {differences}"
                )

    print(
        "test_packaged_config_parity.py passed: "
        "3 methods x seeds 1-5, 100 exact checkpoints per run"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

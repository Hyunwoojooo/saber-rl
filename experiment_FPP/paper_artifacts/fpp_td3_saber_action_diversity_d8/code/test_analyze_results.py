#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load_module():
    path = HERE / "analyze_results.py"
    spec = importlib.util.spec_from_file_location("analyze_diversity_results", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_metrics(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "train/loss", "eval/env_success_mean", "eval/success_mean"])
        writer.writerows(rows)


def test_known_auc_and_duplicate_train_eval_rows() -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.csv"
        write_metrics(
            path,
            [
                [10, 1.0, "", ""],
                [10, "", 0.0, 0.9],
                [20, 0.5, "", ""],
                [20, "", 0.5, 0.1],
                [20, "", 0.5, 0.2],
                [30, "", 1.0, 0.0],
            ],
        )
        steps, values = module.load_eval_curve(path, (10, 20, 30))
    np.testing.assert_array_equal(steps, np.asarray([10.0, 20.0, 30.0]))
    np.testing.assert_array_equal(values, np.asarray([0.0, 0.5, 1.0]))
    assert module.normalized_auc(steps, values) == 0.5
    assert module.first_hit90(steps, values) == 30


def test_missing_checkpoint_raises() -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.csv"
        write_metrics(path, [[10, "", 0.0, ""], [30, "", 1.0, ""]])
        try:
            module.load_eval_curve(path, (10, 20, 30))
        except ValueError as exc:
            assert "missing exact evaluation checkpoints [20]" in str(exc)
        else:
            raise AssertionError("missing evaluation checkpoint must fail")


def synthetic_rows(module):
    rows = []
    methods = {
        "her": ("TD3+HER", 0.1, 1, "support"),
        "saber": ("TD3+HER+SABER", 0.2, 1, "support"),
        "diverse": ("TD3+HER+SABER-Diverse", 0.3, 4, "support"),
        "random_count": ("Random-Diversity-Count", 0.25, 4, "random_count"),
    }
    for key, (label, offset, dmin, assignment) in methods.items():
        for seed in module.SEEDS:
            rows.append(
                {
                    "method_key": key,
                    "method": label,
                    "seed": seed,
                    "recomputed_auc": offset + seed * 0.01,
                    "native_auc": offset + seed * 0.01,
                    "final_success": offset + seed * 0.02,
                    "hit90": 100_000 + seed * 10_000 if seed != 5 else None,
                    "support_min_unique_action_bins": dmin,
                    "support_assignment_mode": assignment,
                }
            )
    return rows


def test_exact_seed_pairing() -> None:
    module = load_module()
    rows = synthetic_rows(module)
    paired = module.paired_rows(rows)
    first = [row for row in paired if row["comparison"] == "Diverse - main SABER"]
    assert [row["seed"] for row in first] == [1, 2, 3, 4, 5]
    np.testing.assert_allclose([row["auc_difference"] for row in first], 0.1)
    broken = [row for row in rows if not (row["method_key"] == "saber" and row["seed"] == 3)]
    try:
        module.paired_rows(broken)
    except ValueError as exc:
        assert "paired seed coverage" in str(exc)
    else:
        raise AssertionError("inexact seed pairing must fail")


def test_condition_config_validation() -> None:
    module = load_module()
    rows = synthetic_rows(module)
    assert module.validate_condition_configs(rows) == 4
    broken = [dict(row) for row in rows]
    target = next(
        row
        for row in broken
        if row["method_key"] == "random_count" and row["seed"] == 3
    )
    target["support_min_unique_action_bins"] = 2
    try:
        module.validate_condition_configs(broken)
    except ValueError as exc:
        assert "one identical Dmin" in str(exc)
    else:
        raise AssertionError("mismatched production Dmin must fail")


def test_bootstrap_is_deterministic() -> None:
    module = load_module()
    differences = [0.01, 0.02, -0.01, 0.03, 0.00]
    first = module.paired_percentile_bootstrap(differences, resamples=1_000, rng_seed=27_027)
    second = module.paired_percentile_bootstrap(differences, resamples=1_000, rng_seed=27_027)
    assert first == second


def main() -> int:
    test_known_auc_and_duplicate_train_eval_rows()
    test_missing_checkpoint_raises()
    test_exact_seed_pairing()
    test_condition_config_validation()
    test_bootstrap_is_deterministic()
    print("test_analyze_results.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

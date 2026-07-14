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
    spec = importlib.util.spec_from_file_location(
        "analyze_count_matched_results",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_metrics(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "train/loss",
                "eval/env_success_mean",
                "eval/success_mean",
            ]
        )
        writer.writerows(rows)


def test_known_auc_and_duplicate_rows() -> None:
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
        before = path.read_bytes()
        steps, values = module.load_eval_curve(path, (10, 20, 30))
        assert path.read_bytes() == before
    np.testing.assert_array_equal(
        steps,
        np.asarray([10.0, 20.0, 30.0]),
    )
    np.testing.assert_array_equal(
        values,
        np.asarray([0.0, 0.5, 1.0]),
    )
    assert module.normalized_auc(steps, values) == 0.5
    assert module.first_hit90(steps, values) == 30


def test_missing_checkpoint_raises() -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.csv"
        write_metrics(
            path,
            [[10, "", 0.0, ""], [30, "", 1.0, ""]],
        )
        try:
            module.load_eval_curve(path, (10, 20, 30))
        except ValueError as exc:
            assert "missing exact evaluation checkpoints [20]" in str(exc)
        else:
            raise AssertionError("missing checkpoint did not fail")


def fixed_config(
    key: str,
    seed: int,
) -> dict[str, object]:
    module = load_module()
    config = dict(module.REQUIRED_PRODUCTION_CONFIG)
    config.update(
        {
            "run_name": f"synthetic_{key}_{seed}",
            "output_root": f"runs/{key}",
            "support_query_mode": module.QUERY_MODES[key],
            "support_assignment_mode": module.ASSIGNMENT_MODES[key],
            "seeds": {
                "env_seed": seed,
                "torch_seed": seed,
                "numpy_seed": seed,
                "action_noise_seed": seed,
                "weight_seed": 20_000 + seed,
                "eval_seed": 10_000 + seed,
            },
        }
    )
    if key == "control":
        config["assignment_diagnostic_window_updates"] = 1_000
        config["seeds"]["assignment_seed"] = 30_000 + seed
    return config


def synthetic_rows(module):
    rows = []
    methods = {
        "main": ("Main SABER", 0.30, False),
        "control": ("Count-matched", 0.20, False),
        "context_only": ("Context-only reference", 0.15, True),
    }
    for key, (label, offset, descriptive) in methods.items():
        for seed in module.SEEDS:
            rows.append(
                {
                    "method_key": key,
                    "method": label,
                    "descriptive_only": descriptive,
                    "seed": seed,
                    "recomputed_auc": offset + seed * 0.01,
                    "native_auc": offset + seed * 0.01,
                    "final_success": offset + seed * 0.02,
                    "best_success": offset + 0.2 + seed * 0.01,
                    "hit90": (
                        100_000 + seed * 10_000
                        if seed != 5
                        else None
                    ),
                    "support_query_mode": module.QUERY_MODES[key],
                    "support_assignment_mode": module.ASSIGNMENT_MODES[key],
                    "_config": fixed_config(key, seed),
                }
            )
    return rows


def test_exact_seed_pairing_and_config_parity() -> None:
    module = load_module()
    rows = synthetic_rows(module)
    assert module.validate_condition_configs(rows) is None
    paired = module.paired_rows(rows)
    assert [row["seed"] for row in paired] == [1, 2, 3, 4, 5]
    np.testing.assert_allclose(
        [row["auc_difference"] for row in paired],
        0.1,
    )

    missing = [
        row
        for row in rows
        if not (
            row["method_key"] == "control"
            and row["seed"] == 3
        )
    ]
    try:
        module.paired_rows(missing)
    except ValueError as exc:
        assert "paired seed coverage" in str(exc)
    else:
        raise AssertionError("inexact seed pairing did not fail")

    broken = [dict(row) for row in rows]
    target = next(
        row
        for row in broken
        if row["method_key"] == "control" and row["seed"] == 2
    )
    target["_config"] = dict(target["_config"])
    target["_config"]["gamma"] = 0.95
    try:
        module.validate_condition_configs(broken)
    except ValueError as exc:
        assert "production config mismatch" in str(exc)
    else:
        raise AssertionError("non-assignment config change did not fail")


def test_paired_inference_is_deterministic() -> None:
    module = load_module()
    differences = [0.01, 0.02, -0.01, 0.03, 0.00]
    first = module.paired_percentile_bootstrap(
        differences,
        resamples=1_000,
        rng_seed=27_027,
    )
    second = module.paired_percentile_bootstrap(
        differences,
        resamples=1_000,
        rng_seed=27_027,
    )
    assert first == second
    assert module.exact_sign_flip_pvalue([1, 1, 1, 1, 1]) == 0.0625

    paired = module.paired_rows(synthetic_rows(module))
    leave_one_out = module.leave_one_seed_out_rows(paired)
    assert [row["omitted_seed"] for row in leave_one_out] == [
        1,
        2,
        3,
        4,
        5,
    ]
    np.testing.assert_allclose(
        [row["mean_auc_difference"] for row in leave_one_out],
        0.1,
    )


def diagnostic_row(
    index: int,
    start: int,
    updates: int,
) -> dict[str, object]:
    end = start + updates - 1
    row: dict[str, object] = {
        "window_index": index,
        "update_start": start,
        "update_end": end,
        "updates": updates,
        "complete_window": int(updates == 1_000),
        "exact_k_mismatch_count": 0,
        "raw_histogram_mismatch_count": 0,
        "normalized_histogram_mismatch_count": 0,
        "invariant_failure_count": 0,
        "normalized_weight_sum_abs_diff_max": 0.0,
    }
    values = {
        "context_eligible_count_mean": 10.0,
        "main_positive_count_mean": 2.0,
        "assigned_positive_count_mean": 2.0,
        "context_eligible_rate_mean": 0.5,
        "main_positive_rate_mean": 0.1,
        "assigned_positive_rate_mean": 0.1,
        "overlap_count_mean": 0.4,
        "overlap_over_k_mean": 0.2,
        "jaccard_mean": 0.1,
        "reference_raw_weight_mean_mean": 1.1,
        "assigned_raw_weight_mean_mean": 1.1,
        "reference_raw_weight_std_mean": 0.3,
        "assigned_raw_weight_std_mean": 0.3,
        "reference_raw_weight_ESS_mean": 250.0,
        "assigned_raw_weight_ESS_mean": 250.0,
        "reference_normalized_weight_mean_mean": 1.0,
        "assigned_normalized_weight_mean_mean": 1.0,
        "reference_normalized_weight_std_mean": 0.2,
        "assigned_normalized_weight_std_mean": 0.2,
        "reference_normalized_weight_ESS_mean": 250.0,
        "assigned_normalized_weight_ESS_mean": 250.0,
    }
    row.update(values)
    return row


def write_diagnostics(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_diagnostic_windows() -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "assignment_diagnostics_1000.csv"
        rows = [
            diagnostic_row(1, 1, 1_000),
            diagnostic_row(2, 1_001, 500),
        ]
        write_diagnostics(path, rows)
        before = path.read_bytes()
        result = module.load_assignment_diagnostics(path, 1_500)
        assert path.read_bytes() == before
        assert result["window_count"] == 2
        assert result["total_updates"] == 1_500
        assert result["main_positive_count_mean"] == 2.0
        assert result["assigned_positive_count_mean"] == 2.0

        rows[1]["assigned_positive_count_mean"] = 3.0
        write_diagnostics(path, rows)
        try:
            module.load_assignment_diagnostics(path, 1_500)
        except ValueError as exc:
            assert "exact-k mismatch" in str(exc)
        else:
            raise AssertionError("diagnostic exact-k mismatch did not fail")


def main() -> int:
    test_known_auc_and_duplicate_rows()
    test_missing_checkpoint_raises()
    test_exact_seed_pairing_and_config_parity()
    test_paired_inference_is_deterministic()
    test_diagnostic_windows()
    print("test_analyze_results.py passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

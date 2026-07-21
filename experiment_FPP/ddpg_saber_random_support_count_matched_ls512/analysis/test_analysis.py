#!/usr/bin/env python3
"""Synthetic fail-closed tests for batch-wide Random-Support analysis."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from random_support_analysis import (
    CONTROL_KEY,
    EXPECTED_CONFIG,
    GRID,
    MAIN_KEY,
    SEEDS,
    TOTAL_STEPS,
    TOTAL_UPDATES,
    AnalysisError,
    analyze,
    auc,
    discover_run,
    exact_sign_flip,
    paired_bootstrap,
    verify_outputs,
    write_outputs,
)


DIAGNOSTIC_FIELDS = (
    "window_index",
    "update_start",
    "update_end",
    "updates",
    "complete_window",
    "batch_size_mean",
    "context_eligible_count_mean",
    "assignment_pool_count_mean",
    "main_positive_count_mean",
    "assigned_positive_count_mean",
    "batch_size_min",
    "batch_size_max",
    "context_eligible_count_min",
    "context_eligible_count_max",
    "assignment_pool_count_min",
    "assignment_pool_count_max",
    "main_positive_count_min",
    "main_positive_count_max",
    "assigned_positive_count_min",
    "assigned_positive_count_max",
    "context_eligible_rate_mean",
    "assignment_pool_rate_mean",
    "main_positive_rate_mean",
    "assigned_positive_rate_mean",
    "overlap_count_mean",
    "overlap_over_k_mean",
    "jaccard_mean",
    "reference_raw_weight_mean_mean",
    "reference_raw_weight_std_mean",
    "reference_raw_weight_ESS_mean",
    "assigned_raw_weight_mean_mean",
    "assigned_raw_weight_std_mean",
    "assigned_raw_weight_ESS_mean",
    "reference_normalized_weight_mean_mean",
    "reference_normalized_weight_std_mean",
    "reference_normalized_weight_ESS_mean",
    "assigned_normalized_weight_mean_mean",
    "assigned_normalized_weight_std_mean",
    "assigned_normalized_weight_ESS_mean",
    "reference_normalized_weight_sum_mean",
    "assigned_normalized_weight_sum_mean",
    "normalized_weight_sum_abs_diff_mean",
    "normalized_weight_sum_abs_diff_max",
    "overlap_count_sum",
    "k_zero_update_count",
    "exact_k_mismatch_count",
    "raw_histogram_mismatch_count",
    "normalized_histogram_mismatch_count",
    "invariant_failure_count",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def synthetic_curve(method_key: str, seed: int) -> tuple[float, ...]:
    offset = 0.02 if method_key == MAIN_KEY else 0.0
    return tuple(0.10 + 0.01 * seed + 0.004 * index + offset for index in range(len(GRID)))


def base_config(method_key: str, seed: int, run_dir: Path) -> dict[str, Any]:
    config = dict(EXPECTED_CONFIG)
    config.update(
        {
            "run_name": run_dir.name,
            "output_root": str(run_dir.parent),
            "smoke": False,
            "resume": False,
            "support_query_mode": "current_policy",
            "support_assignment_mode": (
                "support"
                if method_key == MAIN_KEY
                else "batch_wide_count_matched"
            ),
            "terminal_convention": {
                "success_terminates": False,
                "timeout_bootstrap": True,
            },
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
    if method_key == CONTROL_KEY:
        config["seeds"]["assignment_seed"] = 30_000 + seed
        config["assignment_diagnostic_window_updates"] = 1_000
    return config


def env_snapshot() -> dict[str, Any]:
    env = {
        "env_id": "FetchPickAndPlace-v4",
        "obs_dim": 25,
        "goal_dim": 3,
        "state_dim": 28,
        "action_dim": 4,
        "max_episode_steps": 50,
    }
    return {"train_env": dict(env), "eval_env": dict(env)}


def write_metrics(run_dir: Path, curve: tuple[float, ...]) -> None:
    path = run_dir / "metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("step", "eval/env_success_mean"),
            lineterminator="\n",
        )
        writer.writeheader()
        for step, value in zip(GRID, curve):
            writer.writerow({"step": step, "eval/env_success_mean": value})


def diagnostic_row(index: int, start: int, updates: int) -> dict[str, Any]:
    raw_mean = 1.25
    raw_std = 0.4330127018922193
    raw_ess = 240.0
    normalized_std = raw_std / raw_mean
    row = {
        "window_index": index,
        "update_start": start,
        "update_end": start + updates - 1,
        "updates": updates,
        "complete_window": int(updates == 1_000),
        "batch_size_mean": 256.0,
        "context_eligible_count_mean": 128.0,
        "assignment_pool_count_mean": 256.0,
        "main_positive_count_mean": 64.0,
        "assigned_positive_count_mean": 64.0,
        "batch_size_min": 256.0,
        "batch_size_max": 256.0,
        "context_eligible_count_min": 128.0,
        "context_eligible_count_max": 128.0,
        "assignment_pool_count_min": 256.0,
        "assignment_pool_count_max": 256.0,
        "main_positive_count_min": 64.0,
        "main_positive_count_max": 64.0,
        "assigned_positive_count_min": 64.0,
        "assigned_positive_count_max": 64.0,
        "context_eligible_rate_mean": 0.5,
        "assignment_pool_rate_mean": 1.0,
        "main_positive_rate_mean": 0.25,
        "assigned_positive_rate_mean": 0.25,
        "overlap_count_mean": 16.0,
        "overlap_over_k_mean": 0.25,
        "jaccard_mean": 16.0 / 112.0,
        "reference_raw_weight_mean_mean": raw_mean,
        "reference_raw_weight_std_mean": raw_std,
        "reference_raw_weight_ESS_mean": raw_ess,
        "assigned_raw_weight_mean_mean": raw_mean,
        "assigned_raw_weight_std_mean": raw_std,
        "assigned_raw_weight_ESS_mean": raw_ess,
        "reference_normalized_weight_mean_mean": 1.0,
        "reference_normalized_weight_std_mean": normalized_std,
        "reference_normalized_weight_ESS_mean": raw_ess,
        "assigned_normalized_weight_mean_mean": 1.0,
        "assigned_normalized_weight_std_mean": normalized_std,
        "assigned_normalized_weight_ESS_mean": raw_ess,
        "reference_normalized_weight_sum_mean": 256.0,
        "assigned_normalized_weight_sum_mean": 256.0,
        "normalized_weight_sum_abs_diff_mean": 0.0,
        "normalized_weight_sum_abs_diff_max": 0.0,
        "overlap_count_sum": 16.0 * updates,
        "k_zero_update_count": 0,
        "exact_k_mismatch_count": 0,
        "raw_histogram_mismatch_count": 0,
        "normalized_histogram_mismatch_count": 0,
        "invariant_failure_count": 0,
    }
    return row


def write_diagnostics(run_dir: Path) -> None:
    path = run_dir / "assignment_diagnostics_1000.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        covered = 0
        index = 0
        while covered < TOTAL_UPDATES:
            updates = min(1_000, TOTAL_UPDATES - covered)
            index += 1
            writer.writerow(diagnostic_row(index, covered + 1, updates))
            covered += updates


def write_run(root: Path, method_key: str, seed: int) -> Path:
    run_dir = root / f"synthetic_{method_key}_s{seed}_{TOTAL_STEPS}"
    run_dir.mkdir(parents=True)
    curve = synthetic_curve(method_key, seed)
    write_json(run_dir / "config.json", base_config(method_key, seed, run_dir))
    write_json(run_dir / "env_snapshot.json", env_snapshot())
    hit90 = next((step for step, value in zip(GRID, curve) if value >= 0.9), None)
    write_json(
        run_dir / "summary.json",
        {
            "method": "b3_support_runtime_opt",
            "total_steps": TOTAL_STEPS,
            "total_updates": TOTAL_UPDATES,
            "eval_AUC_success": auc(curve),
            "final_eval_success": curve[-1],
            "best_eval_success": max(curve),
            "first_hit_0.90": hit90,
            "run_dir": str(run_dir),
        },
    )
    write_metrics(run_dir, curve)
    if method_key == CONTROL_KEY:
        write_diagnostics(run_dir)
    return run_dir


def make_fixture(base: Path) -> tuple[Path, Path]:
    main_root = base / "main"
    control_root = base / "control"
    for root in (main_root, control_root):
        root.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        write_run(main_root, MAIN_KEY, seed)
        write_run(control_root, CONTROL_KEY, seed)
    return main_root, control_root


def mutate_csv(path: Path, mutate: Callable[[list[dict[str, str]]], None]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mutate(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class RandomSupportAnalysisTests(unittest.TestCase):
    def test_exact_analysis_outputs_and_inference_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            main_root, control_root = make_fixture(Path(temporary))
            first = analyze(main_root, control_root)
            second = analyze(main_root, control_root)
            self.assertEqual(first.inference_rows, second.inference_rows)
            self.assertEqual(len(first.per_seed_rows), 10)
            self.assertEqual(len(first.mean_curve_rows), 200)
            self.assertEqual(len(first.diagnostic_rows), 5)
            self.assertTrue(
                math.isclose(
                    first.inference_rows[0]["mean_delta"],
                    0.02,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            self.assertEqual(first.inference_rows[0]["positive_count"], 5)
            self.assertEqual(first.inference_rows[0]["exact_sign_flip_assignments"], 32)
            output = Path(temporary) / "results"
            write_outputs(first, output)
            verify_outputs(first, output)

    def test_missing_and_duplicate_grid_fail_closed(self) -> None:
        for operation, pattern in (("missing", "missing exact"), ("duplicate", "duplicate")):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                main_root, control_root = make_fixture(Path(temporary))
                run = discover_run(control_root, CONTROL_KEY, 1)
                metrics = run / "metrics.csv"
                if operation == "missing":
                    mutate_csv(metrics, lambda rows: rows.pop(10))
                else:
                    mutate_csv(metrics, lambda rows: rows.append(dict(rows[10])))
                with self.assertRaisesRegex(AnalysisError, pattern):
                    analyze(main_root, control_root)

    def test_forbidden_config_difference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            main_root, control_root = make_fixture(Path(temporary))
            run = discover_run(control_root, CONTROL_KEY, 3)
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            config["gamma"] = 0.97
            write_json(run / "config.json", config)
            with self.assertRaisesRegex(AnalysisError, "config.gamma|config differences"):
                analyze(main_root, control_root)

    def test_assignment_statistic_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            main_root, control_root = make_fixture(Path(temporary))
            run = discover_run(control_root, CONTROL_KEY, 2)
            diagnostics = run / "assignment_diagnostics_1000.csv"
            mutate_csv(
                diagnostics,
                lambda rows: rows[0].__setitem__(
                    "assigned_raw_weight_mean_mean", "1.2501"
                ),
            )
            with self.assertRaisesRegex(AnalysisError, "raw_weight mean mismatch"):
                analyze(main_root, control_root)

    def test_diagnostic_coverage_and_histogram_failures_fail(self) -> None:
        for field, value, pattern in (
            ("raw_histogram_mismatch_count", "1", "nonzero"),
            ("complete_window", "0", "complete_window"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                main_root, control_root = make_fixture(Path(temporary))
                run = discover_run(control_root, CONTROL_KEY, 4)
                diagnostics = run / "assignment_diagnostics_1000.csv"
                mutate_csv(diagnostics, lambda rows: rows[0].__setitem__(field, value))
                with self.assertRaisesRegex(AnalysisError, pattern):
                    analyze(main_root, control_root)

    def test_exact_k_counter_and_bounds_fail_closed(self) -> None:
        for field, value, pattern in (
            ("exact_k_mismatch_count", "1", "nonzero"),
            ("context_eligible_rate_mean", "1.1", "outside"),
            ("assignment_pool_rate_mean", "0.9", "pool rate"),
            ("assignment_pool_count_mean", "255", "full minibatch"),
            ("assignment_pool_count_min", "0", "full minibatch"),
            ("batch_size_min", "0", "full minibatch"),
            ("overlap_count_mean", "65", "overlap count"),
            ("reference_raw_weight_ESS_mean", "257", "ESS outside"),
            ("reference_raw_weight_mean_mean", "1", "raw-weight mean"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                main_root, control_root = make_fixture(Path(temporary))
                run = discover_run(control_root, CONTROL_KEY, 5)
                diagnostics = run / "assignment_diagnostics_1000.csv"
                def mutate(rows: list[dict[str, str]]) -> None:
                    rows[0][field] = value
                    if field == "reference_raw_weight_ESS_mean":
                        rows[0]["assigned_raw_weight_ESS_mean"] = value
                    if field == "reference_raw_weight_mean_mean":
                        rows[0]["assigned_raw_weight_mean_mean"] = value
                mutate_csv(diagnostics, mutate)
                with self.assertRaisesRegex(AnalysisError, pattern):
                    analyze(main_root, control_root)

    def test_primary_environment_success_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            main_root, control_root = make_fixture(Path(temporary))
            run = discover_run(control_root, CONTROL_KEY, 1)
            metrics = run / "metrics.csv"
            with metrics.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            with metrics.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("step", "eval/env_success_mean", "eval/success_mean"),
                    lineterminator="\n",
                )
                writer.writeheader()
                for index, row in enumerate(rows):
                    writer.writerow(
                        {
                            "step": row["step"],
                            "eval/env_success_mean": "" if index == 10 else row["eval/env_success_mean"],
                            "eval/success_mean": row["eval/env_success_mean"],
                        }
                    )
            with self.assertRaisesRegex(AnalysisError, "missing exact"):
                analyze(main_root, control_root)

    def test_bootstrap_and_exact_sign_flip(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.assertEqual(
            paired_bootstrap(values, resamples=1_000, seed=27_027),
            paired_bootstrap(values, resamples=1_000, seed=27_027),
        )
        self.assertEqual(exact_sign_flip([1.0] * 5), 2.0 / 32.0)

    def test_raw_optional_summary_nan_is_accepted_but_critical_nan_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            main_root, control_root = make_fixture(Path(temporary))
            run = discover_run(control_root, CONTROL_KEY, 1)
            summary_path = run / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["final_lite_metrics"] = {
                "b3/support_to_success_corr": float("nan")
            }
            summary_path.write_text(
                json.dumps(summary, allow_nan=True) + "\n",
                encoding="utf-8",
            )
            analyze(main_root, control_root)
            summary["eval_AUC_success"] = float("nan")
            summary_path.write_text(
                json.dumps(summary, allow_nan=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AnalysisError, "eval_AUC_success"):
                analyze(main_root, control_root)


if __name__ == "__main__":
    unittest.main()

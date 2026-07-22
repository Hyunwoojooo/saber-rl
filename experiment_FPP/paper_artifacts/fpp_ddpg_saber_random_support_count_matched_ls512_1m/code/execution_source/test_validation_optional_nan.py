#!/usr/bin/env python3
"""Regression tests for canonical optional raw NaN sentinels."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from validate_results import (
    GateError,
    inspect_jsonl,
    load_assignment_diagnostics,
    read_object,
)


def valid_assignment_window() -> dict[str, object]:
    batch = 256.0
    eligible = 128.0
    positive = 64.0
    raw_mean = 1.0 + positive / batch
    raw_std = math.sqrt((positive / batch) * (1.0 - positive / batch))
    raw_ess = (batch * raw_mean) ** 2 / (
        (batch - positive) + 4.0 * positive
    )
    normalized_std = raw_std / raw_mean
    return {
        "window_index": 1,
        "update_start": 1,
        "update_end": 1,
        "updates": 1,
        "complete_window": 0,
        "batch_size_mean": batch,
        "context_eligible_count_mean": eligible,
        "assignment_pool_count_mean": batch,
        "assignment_pool_rate_mean": 1.0,
        "main_positive_count_mean": positive,
        "assigned_positive_count_mean": positive,
        "context_eligible_rate_mean": eligible / batch,
        "main_positive_rate_mean": positive / batch,
        "assigned_positive_rate_mean": positive / batch,
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
        "reference_normalized_weight_sum_mean": batch,
        "assigned_normalized_weight_sum_mean": batch,
        "normalized_weight_sum_abs_diff_mean": 0.0,
        "batch_size_min": batch,
        "batch_size_max": batch,
        "context_eligible_count_min": eligible,
        "context_eligible_count_max": eligible,
        "assignment_pool_count_min": batch,
        "assignment_pool_count_max": batch,
        "main_positive_count_min": positive,
        "main_positive_count_max": positive,
        "assigned_positive_count_min": positive,
        "assigned_positive_count_max": positive,
        "overlap_count_sum": 16.0,
        "k_zero_update_count": 0,
        "exact_k_mismatch_count": 0,
        "raw_histogram_mismatch_count": 0,
        "normalized_histogram_mismatch_count": 0,
        "invariant_failure_count": 0,
        "normalized_weight_sum_abs_diff_max": 0.0,
    }


def write_assignment_window(path: Path, row: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


class OptionalRawNaNTests(unittest.TestCase):
    def test_assignment_window_rejects_full_batch_and_binary_count_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "assignment_diagnostics_1000.csv"
            baseline = valid_assignment_window()
            write_assignment_window(path, baseline)
            self.assertEqual(
                load_assignment_diagnostics(path, total_updates=1)["covered_updates"],
                1,
            )
            for field, value, pattern in (
                ("assignment_pool_count_min", 0, "full minibatch"),
                ("batch_size_min", 0, "full minibatch"),
            ):
                with self.subTest(field=field):
                    forged = dict(baseline)
                    forged[field] = value
                    write_assignment_window(path, forged)
                    with self.assertRaisesRegex(GateError, pattern):
                        load_assignment_diagnostics(path, total_updates=1)
            forged = dict(baseline)
            forged["reference_raw_weight_mean_mean"] = 1.0
            forged["assigned_raw_weight_mean_mean"] = 1.0
            write_assignment_window(path, forged)
            with self.assertRaisesRegex(GateError, "binary raw-weight mean"):
                load_assignment_diagnostics(path, total_updates=1)

    def test_summary_allows_only_known_optional_nan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "total_steps": 5_000,
                        "final_eval_success": 0.0,
                        "final_lite_metrics": {
                            "b3/support_to_success_corr": float("nan")
                        },
                    },
                    allow_nan=True,
                ),
                encoding="utf-8",
            )
            parsed = read_object(
                path, allow_optional_summary_nonfinite=True
            )
            self.assertNotEqual(
                parsed["final_lite_metrics"]["b3/support_to_success_corr"],
                parsed["final_lite_metrics"]["b3/support_to_success_corr"],
            )
            with self.assertRaisesRegex(GateError, "non-finite"):
                read_object(path)

            path.write_text(
                '{"final_eval_success": NaN, "final_lite_metrics": {}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(GateError, "final_eval_success"):
                read_object(path, allow_optional_summary_nonfinite=True)

    def test_jsonl_allows_optional_but_rejects_critical_nan_and_infinity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            path.write_text(
                '{"step": 1, "query/query_bin_matches_replay_action_rate": NaN}\n',
                encoding="utf-8",
            )
            self.assertEqual(inspect_jsonl(path, 1), 1)
            path.write_text(
                '{"step": 1, "pushT/obj_goal_xy_dist_mean": NaN}\n',
                encoding="utf-8",
            )
            self.assertEqual(inspect_jsonl(path, 1), 1)
            for key, constant in (
                ("loss/critic", "NaN"),
                ("eval/env_success_mean", "NaN"),
                ("assignment/invariant_failure_count", "NaN"),
                ("runtime/support_cache_hit_rate", "Infinity"),
                ("pushT/obj_goal_xy_dist_mean", "Infinity"),
            ):
                with self.subTest(key=key, constant=constant):
                    path.write_text(
                        f'{{"step": 1, "{key}": {constant}}}\n',
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(GateError, "critical non-finite"):
                        inspect_jsonl(path, 1)


if __name__ == "__main__":
    unittest.main()

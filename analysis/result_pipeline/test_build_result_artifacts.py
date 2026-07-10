#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_result_artifacts as pipeline


class MetricOverrideTests(unittest.TestCase):
    def test_common_grid_metrics_replace_only_configured_fields(self) -> None:
        records = []
        for method_id in ("method_a", "method_b"):
            for seed in (1, 2):
                records.append(
                    {
                        "method_id": method_id,
                        "seed": seed,
                        "final_success": 0.1,
                        "success_auc": 0.2,
                        "hit90_step": None,
                        "native_final_success": 0.1,
                        "native_success_auc": 0.2,
                        "native_hit90_step": None,
                    }
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "override.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "method_key",
                        "seed",
                        "common_grid_auc",
                        "common_grid_final_success",
                        "common_grid_hit90",
                    ],
                )
                writer.writeheader()
                for method_key in ("a", "b"):
                    for seed in (1, 2):
                        writer.writerow(
                            {
                                "method_key": method_key,
                                "seed": seed,
                                "common_grid_auc": 0.5 + seed / 10,
                                "common_grid_final_success": 0.8,
                                "common_grid_hit90": 100000 if seed == 1 else "",
                            }
                        )

            config = {
                "metric_overrides": [
                    {
                        "id": "test_grid",
                        "csv": "override.csv",
                        "method_key_to_id": {"a": "method_a", "b": "method_b"},
                        "metric_columns": {
                            "success_auc": "common_grid_auc",
                            "final_success": "common_grid_final_success",
                            "hit90_step": "common_grid_hit90",
                        },
                        "expected_seeds": [1, 2],
                        "curve_grid": {"start_step": 50000, "stop_step": 1000000, "step_size": 50000},
                    }
                ]
            }

            notes = pipeline.apply_metric_overrides(config, records, root)

        self.assertEqual(notes, ["metric override applied: test_grid (4 method-seed rows)"])
        self.assertEqual(records[0]["success_auc"], 0.6)
        self.assertEqual(records[0]["final_success"], 0.8)
        self.assertEqual(records[0]["hit90_step"], 100000)
        self.assertIsNone(records[1]["hit90_step"])
        self.assertEqual(records[0]["native_success_auc"], 0.2)
        self.assertEqual(records[0]["metric_source"], "test_grid")
        self.assertEqual(records[0]["curve_grid_step_size"], 50000)

    def test_paired_ci_override_uses_packaged_interval(self) -> None:
        paired_ci = [
            {
                "pair_id": "pair_a",
                "metric": "success_auc",
                "n": 5,
                "mean_delta": 0.0,
                "ci_low": -1.0,
                "ci_high": 1.0,
                "bootstrap_iterations": 100,
                "ci_source": "result_pipeline_bootstrap",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "paired.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "comparison",
                        "n_seed_pairs",
                        "paired_difference_mean",
                        "ci_lower_95",
                        "ci_upper_95",
                        "resamples",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "comparison": "A - B",
                        "n_seed_pairs": 5,
                        "paired_difference_mean": 0.112,
                        "ci_lower_95": 0.064,
                        "ci_upper_95": 0.161,
                        "resamples": 10000,
                    }
                )
            config = {
                "paired_ci_overrides": [
                    {
                        "id": "test_grid",
                        "csv": "paired.csv",
                        "metric": "success_auc",
                        "comparison_to_pair_id": {"A - B": "pair_a"},
                    }
                ]
            }

            notes = pipeline.apply_paired_ci_overrides(config, paired_ci, root)

        self.assertEqual(notes, ["paired CI override applied: test_grid (1 comparisons)"])
        self.assertEqual(paired_ci[0]["mean_delta"], 0.112)
        self.assertEqual(paired_ci[0]["ci_low"], 0.064)
        self.assertEqual(paired_ci[0]["ci_high"], 0.161)
        self.assertEqual(paired_ci[0]["bootstrap_iterations"], 10000)
        self.assertEqual(paired_ci[0]["ci_source"], "test_grid")

    def test_curve_grid_filter_is_exact(self) -> None:
        record = {
            "curve_grid_start_step": 50000,
            "curve_grid_stop_step": 1000000,
            "curve_grid_step_size": 50000,
        }
        self.assertTrue(pipeline.curve_step_allowed(record, 50000))
        self.assertTrue(pipeline.curve_step_allowed(record, 1000000))
        self.assertFalse(pipeline.curve_step_allowed(record, 75000))
        self.assertFalse(pipeline.curve_step_allowed(record, 0))


if __name__ == "__main__":
    unittest.main()

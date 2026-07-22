#!/usr/bin/env python3
"""Regression tests for canonical optional raw NaN sentinels."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from validate_results import GateError, inspect_jsonl, read_object


class OptionalRawNaNTests(unittest.TestCase):
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

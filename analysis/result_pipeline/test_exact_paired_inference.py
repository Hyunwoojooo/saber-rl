#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from build_result_artifacts import apply_paired_ci_overrides, exact_paired_inference, extract_seed


def rows(deltas):
    return [
        {
            "pair_id": "pair",
            "baseline_id": "baseline",
            "treatment_id": "treatment",
            "task": "task",
            "backbone": "backbone",
            "seed": seed,
            "metric": "success_auc",
            "delta": delta,
        }
        for seed, delta in enumerate(deltas, start=1)
    ]


class ExactPairedInferenceTest(unittest.TestCase):
    def test_five_positive_pairs_have_discrete_minimum_p_value(self):
        result = exact_paired_inference(rows([1.0, 2.0, 3.0, 4.0, 5.0]))[0]
        self.assertEqual(result["n_positive"], 5)
        self.assertEqual(result["sign_assignments"], 32)
        self.assertAlmostEqual(result["exact_sign_flip_p_two_sided"], 0.0625)
        self.assertTrue(result["leave_one_out_same_direction"])

    def test_leave_one_out_detects_direction_instability(self):
        result = exact_paired_inference(rows([4.0, -1.0]))[0]
        self.assertGreater(result["mean_delta"], 0)
        self.assertFalse(result["leave_one_out_same_direction"])
        self.assertLess(result["leave_one_out_mean_min"], 0)
        self.assertGreater(result["leave_one_out_mean_max"], 0)

    def test_extract_seed_uses_adjacent_run_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "6"
            run_dir.mkdir()
            summary_path = run_dir / "summary.json"
            summary_path.write_text("{}")
            (run_dir / "config.json").write_text(json.dumps({"seeds": {"env_seed": 6}}))
            self.assertEqual(extract_seed(summary_path, {}), 6)

    def test_analysis_json_overrides_frozen_paired_auc_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "analysis.json"
            source_path.write_text(
                json.dumps(
                    {
                        "n_pairs": 10,
                        "paired_auc_difference_mean": 0.0467,
                        "paired_bootstrap_ci_95": [0.0085, 0.0783],
                        "paired_bootstrap_resamples": 10000,
                    }
                )
            )
            paired_ci = [
                {
                    "pair_id": "pair",
                    "metric": "success_auc",
                    "n": 10,
                    "mean_delta": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "bootstrap_iterations": 10,
                }
            ]
            config = {
                "paired_ci_overrides": [
                    {
                        "id": "frozen",
                        "analysis_json": "analysis.json",
                        "metric": "success_auc",
                        "pair_id": "pair",
                    }
                ]
            }
            notes = apply_paired_ci_overrides(config, paired_ci, root=Path(tmp))
            self.assertEqual(notes, ["paired CI override applied: frozen (1 comparison)"])
            self.assertEqual(paired_ci[0]["n"], 10)
            self.assertAlmostEqual(paired_ci[0]["mean_delta"], 0.0467)
            self.assertAlmostEqual(paired_ci[0]["ci_low"], 0.0085)
            self.assertAlmostEqual(paired_ci[0]["ci_high"], 0.0783)
            self.assertEqual(paired_ci[0]["ci_source"], "frozen")


if __name__ == "__main__":
    unittest.main()

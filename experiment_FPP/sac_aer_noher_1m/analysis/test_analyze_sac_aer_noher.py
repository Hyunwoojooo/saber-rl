from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import analyze_sac_aer_noher as analysis


class AnalyzerTests(unittest.TestCase):
    def test_constant_curve_auc(self) -> None:
        curve = [0.37] * len(analysis.GRID)
        self.assertAlmostEqual(analysis.normalized_auc(curve), 0.37)

    def test_exact_sign_flip_all_positive_equal(self) -> None:
        differences = [1.0] * 10
        self.assertEqual(analysis.exact_sign_flip_p(differences), 2.0 / (1 << 10))

    def test_seed_spec(self) -> None:
        self.assertEqual(analysis.parse_seed_spec("1-3,7"), (1, 2, 3, 7))
        with self.assertRaises(ValueError):
            analysis.parse_seed_spec("3-1")

    def test_load_run_recomputes_exact_grid_auc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "arbitrary-name"
            run_dir.mkdir()
            config = {
                "method": "sac_aer",
                "task_name": "fetch_pick_and_place",
                "reward_type": "sparse",
                "env_id": "FetchPickAndPlace-v4",
                "total_steps": 1_000_000,
                "eval_freq": 10_000,
                "her_ratio": 0.0,
                "seeds": {"env_seed": 4},
            }
            (run_dir / "config.json").write_text(json.dumps(config))
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "method": "sac_aer",
                        "total_steps": 1_000_000,
                        "eval_AUC_success": 0.5,
                    }
                )
            )
            with (run_dir / "metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["step", "eval/success_mean"])
                writer.writeheader()
                for step in analysis.GRID:
                    writer.writerow({"step": step, "eval/success_mean": 0.5})

            loaded = analysis.load_run(root, "sac_aer", 4, 1_000_000)
            self.assertAlmostEqual(loaded.auc, 0.5)
            self.assertEqual(loaded.final, 0.5)


if __name__ == "__main__":
    unittest.main()

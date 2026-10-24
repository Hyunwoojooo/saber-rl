"""CPU-only regression tests for compact SAC publication, never training."""
from pathlib import Path
import tempfile
import unittest

from sac_results import STEPS, assert_close, auc, curve, inventory, paired, within


class ResultTests(unittest.TestCase):
    def rows(self):
        return [{"step": str(step), "eval/env_success_mean": "0.5"} for step in STEPS]

    def test_exact_grid_and_auc_denominator(self):
        self.assertEqual(len(curve(self.rows())), 200)
        self.assertEqual(auc([(10000, 0), (2000000, 1)]), .5)

    def test_missing_duplicate_and_reordered_checkpoints_rejected(self):
        rows = self.rows()
        for bad in (rows[1:], rows + rows[-1:], list(reversed(rows))):
            with self.assertRaises(ValueError):
                curve(bad)

    def test_invalid_success_rejected(self):
        for value in ("nan", "inf", "-0.01", "1.01"):
            rows = self.rows()
            rows[0]["eval/env_success_mean"] = value
            with self.assertRaises(ValueError):
                curve(rows)

    def test_zero_and_failed_to_learn_seeds_are_not_dropped(self):
        result = paired([0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(result["n_pairs"], 5)
        self.assertEqual(result["positive_pairs"], 0)
        for level in ("ci95", "ci97_5", "ci98_75"):
            self.assertEqual(result[level], [0.0, 0.0])
        with self.assertRaises(ValueError):
            paired([0.0] * 4)

    def test_bootstrap_is_deterministic_and_intervals_nested(self):
        values = [.1, -.3, .2, .0, .4]
        first = paired(values)
        self.assertEqual(first, paired(values))
        self.assertLessEqual(first["ci98_75"][0], first["ci97_5"][0])
        self.assertLessEqual(first["ci97_5"][0], first["ci95"][0])
        self.assertGreaterEqual(first["ci98_75"][1], first["ci97_5"][1])
        self.assertGreaterEqual(first["ci97_5"][1], first["ci95"][1])

    def test_recursive_numeric_check(self):
        assert_close({"x": [1.0, None]}, {"x": [1.0 + 1e-12, None]})
        for invalid in (1.1, float("nan")):
            with self.assertRaises(ValueError):
                assert_close({"x": [invalid]}, {"x": [1.0]})

    def test_relative_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            for path in ("../outside", "/etc/passwd"):
                with self.assertRaises(ValueError):
                    within(Path(tmp), path)

    def test_manifest_detects_changed_data_and_extra_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.csv"
            data.write_text("step,success\n10000,0.0\n")
            before = inventory(root)
            data.write_text("step,success\n10000,1.0\n")
            self.assertNotEqual(inventory(root), before)
            (root / "checkpoint.pt").write_bytes(b"not a real checkpoint")
            with self.assertRaises(ValueError):
                inventory(root)


if __name__ == "__main__":
    unittest.main()

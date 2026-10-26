"""CPU-only tests for exact-grid seed-extension reporting."""
import unittest

from ddpg_seed_extension_results import measure, paired, summarize


class ExtensionResultTests(unittest.TestCase):
    def rows(self):
        return [{"arm": arm, "seed": seed, "auc": .5, "final_quarter_auc": .5,
                 "final_success": .5, "hit90_step": None} for arm in ("her", "saber") for seed in range(1, 11)]

    def test_auc_excludes_unobserved_prefix(self):
        result = measure([(s, .5) for s in range(10000, 2000001, 10000)])
        self.assertEqual(result["auc"], .5)
        self.assertEqual(result["final_quarter_auc"], .5)
        self.assertIsNone(result["hit90_step"])

    def test_invalid_checkpoint_grid_and_nonfinite_rejected(self):
        points = [(s, .5) for s in range(10000, 2000001, 10000)]
        for invalid in (points[1:], points + points[-1:], points[::-1], [(10000, float("nan"))] + points[1:]):
            with self.assertRaises(ValueError):
                measure(invalid)

    def test_paired_resampling_is_deterministic_and_count_checked(self):
        for n in (5, 10):
            self.assertEqual(paired([1.] * n, [0.] * n)["ci95"], [1., 1.])
            self.assertEqual(paired(list(range(n)), [0.] * n), paired(list(range(n)), [0.] * n))
        with self.assertRaises(ValueError):
            paired([1.] * 4, [0.] * 4)
        with self.assertRaises(ValueError):
            paired([1.] * 5, [0.] * 10)

    def test_pairing_uses_seed_not_input_order(self):
        rows = self.rows()
        for r in rows:
            r["auc"] = r["seed"] / 10
        result = summarize(rows[::-1])
        self.assertEqual(result["cohorts"]["pooled_1_10"]["paired"]["auc"]["ci95"], [0., 0.])

    def test_missing_and_duplicate_seeds_rejected(self):
        rows = self.rows()
        for invalid in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                summarize(invalid)

    def test_zero_success_is_a_valid_completed_outcome(self):
        rows = self.rows()
        for row in rows:
            if row["seed"] == 7 and row["arm"] == "saber":
                row.update(auc=0., final_quarter_auc=0., final_success=0.)
        result = summarize(rows)
        self.assertEqual(result["cohorts"]["pooled_1_10"]["paired"]["auc"]["n"], 10)
        self.assertEqual(len(result["per_seed"]), 20)


if __name__ == "__main__":
    unittest.main()

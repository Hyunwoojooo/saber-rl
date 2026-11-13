"""Completed-cohort publication guards; no simulator, trainer or new training."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest

from sac_seed_extension_results import (
    BASELINE_SOURCE, SOURCE, LOSS_FIELDS, METRICS, append_method, comparison_groups,
    require_complete, require_complete_her, scan_training_csv, source_directories, summarize,
)
from sac_results import curve


def completed():
    return {"status": "completed", "seeds": {
        str(s): {"status": "completed", "exit_code": 0, "step": 2000000, "nonfinite_loss": []}
        for s in range(6, 11)}}


def rows(method="her"):
    return [{"method": method, "seed": s, **{k: 0.0 for k in METRICS}, "hit90": None}
            for s in range(1, 11)]


class PublicationTests(unittest.TestCase):
    def test_five_new_her_seeds_required(self):
        require_complete_her(completed())
        for key, value in (("status", "running"), ("step", 1990000), ("exit_code", 1),
                           ("nonfinite_loss", ["critic"])):
            state = completed()
            state["seeds"]["8"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_complete_her(state)

    def test_missing_completion_rejected(self):
        state = completed()
        del state["seeds"]["8"]
        with self.assertRaises(ValueError):
            require_complete_her(state)

    def test_three_cohorts_without_paired_ci(self):
        report = summarize(rows())
        self.assertEqual(report["comparisons"], {})
        self.assertEqual(report["published_methods"], ["her"])
        self.assertEqual([r["n"] for r in report["aggregates"]], [5, 5, 10])

    def test_other_methods_not_exported_even_if_completed(self):
        values = rows()
        values.append({**values[-1], "method": "saber"})
        with self.assertRaises(ValueError):
            summarize(values)

    def test_zero_success_not_dropped(self):
        values = rows()
        values[5]["auc"] = 1.0
        report = summarize(values)
        self.assertEqual(report["aggregates"][1]["auc_mean"], 0.2)
        self.assertEqual(report["aggregates"][1]["auc_sd"], 5 ** -0.5)
        with self.assertRaises(ValueError):
            summarize(values[:7] + values[8:])

    def test_duplicate_seed_rejected(self):
        values = rows()
        values[-1] = copy.deepcopy(values[-2])
        with self.assertRaises(ValueError):
            summarize(values)

    def test_complete_saber_enables_three_paired_cohorts(self):
        report = summarize(rows() + rows("saber"), ("her", "saber"))
        self.assertEqual(report["new_completed_runs"], 10)
        self.assertEqual(report["historical_reference_runs"], 10)
        self.assertEqual(report["published_methods"], ["her", "saber"])
        for cohort, n in (("historical_1_5", 5), ("extension_6_10", 5), ("pooled_1_10", 10)):
            value = report["comparisons"][cohort]["auc"]
            self.assertEqual(value["n"], n)
            self.assertEqual(value["ci95"], [0.0, 0.0])
            self.assertEqual(value["seed_differences"], [0.0] * n)

    def test_missing_or_duplicate_saber_seed_rejected(self):
        values = rows() + rows("saber")
        for bad in (values[:-1], values[:-1] + [values[-2]]):
            with self.assertRaises(ValueError):
                summarize(bad, ("her", "saber"))

    def test_hgr_and_per_not_authorized_by_saber_append(self):
        for method in ("hgr", "per"):
            with self.assertRaises(ValueError):
                summarize(rows() + rows(method), ("her", method))

    def test_hgr_append_requires_all_ten_seeds(self):
        values = rows() + rows("saber") + rows("hgr")
        report = summarize(values, ("her", "saber", "hgr"))
        self.assertEqual(report["new_completed_runs"], 15)
        self.assertEqual(report["historical_reference_runs"], 15)
        self.assertEqual(len(report["aggregates"]), 9)
        for bad in (values[:-1], values[:-1] + [values[-2]]):
            with self.assertRaises(ValueError):
                summarize(bad, ("her", "saber", "hgr"))

    def test_hgr_publication_does_not_enable_per(self):
        values = rows() + rows("saber") + rows("hgr")
        with self.assertRaises(ValueError):
            summarize(values + rows("per"), ("her", "saber", "hgr"))
        with self.assertRaises(ValueError):
            summarize(values + rows("per"), ("her", "saber", "hgr", "per"))
        with self.assertRaises(ValueError):
            append_method(Path("/nonexistent"), "per")

    def test_hgr_source_family_is_explicit(self):
        self.assertEqual(source_directories(("her", "saber")), {"her_saber": SOURCE})
        self.assertEqual(source_directories(("her", "saber", "hgr")),
                         {"her_saber": SOURCE, "hgr_per": BASELINE_SOURCE})

    def test_hgr_contrast_preserves_her_results_and_sign(self):
        values = rows() + rows("saber") + rows("hgr")
        for row in values:
            row["auc"] = row["seed"] / {"her": 100, "saber": 50, "hgr": 25}[row["method"]]
        old = summarize([r for r in values if r["method"] != "hgr"], ("her", "saber"))
        report = summarize(values[::-1], ("her", "saber", "hgr"))
        self.assertEqual(report["comparisons"], old["comparisons"])
        groups = comparison_groups(report)
        self.assertEqual(set(groups), {"saber_minus_her", "saber_minus_hgr"})
        result = groups["saber_minus_hgr"]["pooled_1_10"]["auc"]
        self.assertAlmostEqual(result["mean_difference"], -.11)
        self.assertEqual(result["positive_pairs"], 0)
        for seed, diff in enumerate(result["seed_differences"], 1):
            self.assertAlmostEqual(diff, -seed / 50)

    def test_hgr_bootstrap_retains_three_comparison_adjustment(self):
        import numpy as np
        values = rows() + rows("saber") + rows("hgr")
        diffs = np.asarray([.01, -.66, -.80, .08, .08, .07, .01, -.59, -.03, -.14])
        for row in values:
            row["auc"] = .9 - (diffs[row["seed"] - 1] if row["method"] == "hgr" else 0)
        report = summarize(values, ("her", "saber", "hgr"))
        result = report["baseline_comparisons"]["saber_minus_hgr"]["pooled_1_10"]["auc"]
        ids = np.random.default_rng(2027).integers(0, 10, size=(10000, 10))
        means = diffs[ids].mean(axis=1)
        np.testing.assert_allclose(result["ci95"], np.quantile(means, [.025, .975]), atol=1e-12)
        np.testing.assert_allclose(result["ci_bonferroni_three"], np.quantile(means, [.05/6, 1-.05/6]), atol=1e-12)

    def test_zero_hgr_success_is_retained(self):
        report = summarize(rows() + rows("saber") + rows("hgr"), ("her", "saber", "hgr"))
        pooled = report["baseline_comparisons"]["saber_minus_hgr"]["pooled_1_10"]["auc"]
        self.assertEqual(pooled["n"], 10)
        self.assertEqual(pooled["ci95"], [0.0, 0.0])

    def test_pairing_uses_seed_and_saber_minus_her_direction(self):
        values = rows() + rows("saber")
        for row in values:
            row["auc"] = row["seed"] / (100 if row["method"] == "saber" else 200)
        expected = summarize(values, ("her", "saber"))
        self.assertEqual(expected, summarize(values[::-1], ("her", "saber")))
        value = expected["comparisons"]["pooled_1_10"]["auc"]
        self.assertAlmostEqual(value["mean_difference"], .0275)
        self.assertEqual(value["positive_pairs"], 10)
        for seed, difference in enumerate(value["seed_differences"], 1):
            self.assertAlmostEqual(difference, seed / 200)

    def test_bootstrap_matches_predeclared_single_metric_calculation(self):
        import numpy as np
        values = rows() + rows("saber")
        diffs = np.asarray([-.1, .1, 0, .03, .04, -.02, .09, 0, -.21, .22])
        for row in values:
            row["auc"] = .3 + (diffs[row["seed"] - 1] if row["method"] == "saber" else 0)
        report = summarize(values, ("her", "saber"))
        value = report["comparisons"]["pooled_1_10"]["auc"]
        indices = np.random.default_rng(2027).integers(0, 10, size=(10000, 10))
        means = diffs[indices].mean(axis=1)
        np.testing.assert_allclose(value["ci95"], np.quantile(means, [.025, .975]), atol=1e-12)
        np.testing.assert_allclose(value["ci_bonferroni_three"], np.quantile(means, [.05/6, 1-.05/6]), atol=1e-12)
        self.assertLessEqual(value["ci_bonferroni_three"][0], value["ci95"][0])
        self.assertGreaterEqual(value["ci_bonferroni_three"][1], value["ci95"][1])

    def test_saber_completion_required_without_performance_filter(self):
        state = completed()
        require_complete(state)
        state["seeds"]["8"]["status"] = "running"
        with self.assertRaises(ValueError):
            require_complete(state)

    def test_exact_evaluation_grid(self):
        values = [{"step": s, "eval/env_success_mean": 0.0} for s in range(10000, 2000001, 10000)]
        self.assertEqual(len(curve(values)), 200)
        for bad in (values[:-1], values + [values[-1]], values[::-1]):
            with self.assertRaises(ValueError):
                curve(bad)

    def test_nonfinite_training_losses_are_not_sanitized_away(self):
        fields = ["step", "runtime/total_sec", "eval/env_success_mean", *LOSS_FIELDS]
        base = {"step": 2000000, "runtime/total_sec": 100.0, "eval/env_success_mean": "",
                **{k: 1.0 for k in LOSS_FIELDS}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.csv"
            for key, value in (("loss/critic", "nan"), ("loss/actor", "inf"), ("sac/alpha", "0")):
                with path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow({**base, key: value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    scan_training_csv(path)

    def test_training_reset_rejected(self):
        fields = ["step", "runtime/total_sec", "eval/env_success_mean", *LOSS_FIELDS]
        base = {"step": 10000, "runtime/total_sec": 100.0, "eval/env_success_mean": "",
                **{k: 1.0 for k in LOSS_FIELDS}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows([base, {**base, "step": 5000}])
            with self.assertRaises(ValueError):
                scan_training_csv(path)


if __name__ == "__main__":
    unittest.main()

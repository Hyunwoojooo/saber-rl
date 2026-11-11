"""HER-only publication guards; no simulator, trainer, GPU or new training."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest

from sac_seed_extension_results import (
    LOSS_FIELDS, METRICS, require_complete_her, scan_training_csv, summarize,
)
from sac_results import curve


def completed():
    return {"status": "completed", "seeds": {
        str(s): {"status": "completed", "exit_code": 0, "step": 2000000, "nonfinite_loss": []}
        for s in range(6, 11)}}


def rows():
    return [{"method": "her", "seed": s, **{k: 0.0 for k in METRICS}, "hit90": None}
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

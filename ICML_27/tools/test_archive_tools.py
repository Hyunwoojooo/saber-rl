"""Small CPU-only tests for publication helpers, independent of the ML environment."""
import json
from pathlib import Path
import tempfile
import unittest

from export_local_experiments import ASSIGN, PAL, DIAG, RETRIEVAL, RECOVERY, BASELINES, SPECS, auc, result_text, scrub
from verify_archive import check_experiment_index, check_hashes, close, digest


class ArchiveTests(unittest.TestCase):
    def test_export_includes_only_the_six_executed_families(self):
        self.assertEqual(set(SPECS), {ASSIGN, PAL, DIAG, RETRIEVAL, RECOVERY, BASELINES})
        self.assertEqual(sum(row[2] for row in SPECS.values()), 90)
        self.assertTrue(all(row[1] in {"completed", "completed_offline", "in_progress"} for row in SPECS.values()))
        with self.assertRaises(ValueError):
            result_text("unused_preparation", Path("not_read"), None)

    def test_index_rejects_preparation_and_unlisted_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiments = root / "experiments"
            (experiments / "production").mkdir(parents=True)
            index = {"experiments": [{"directory": "production", "status": "completed", "new_production_runs_planned": 10}],
                     "training_counts": {"planned": 10}}
            check_experiment_index(root, index)
            index["experiments"][0]["status"] = "superseded_before_production"
            with self.assertRaises(ValueError):
                check_experiment_index(root, index)
            index["experiments"][0]["status"] = "completed"
            (experiments / "preparation").mkdir()
            with self.assertRaises(ValueError):
                check_experiment_index(root, index)

    def test_auc_does_not_integrate_unobserved_zero_prefix(self):
        self.assertEqual(auc([(10000, 0), (2000000, 1)]), 0.5)

    def test_path_and_nonfinite_sanitization(self):
        value = {"path": "/private/work/run", "values": [float("nan"), float("inf"), 0.5]}
        got = scrub(value, Path("/private/work"))
        self.assertEqual(got, {"path": "<WORKSPACE>/run", "values": [None, None, 0.5]})
        json.dumps(got, allow_nan=False)

    def test_hash_change_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data.txt"
            path.write_text("original")
            hashes = {"data.txt": digest(path)}
            check_hashes(root, hashes)
            path.write_text("changed")
            with self.assertRaises(ValueError):
                check_hashes(root, hashes)

    def test_manifest_path_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                check_hashes(Path(tmp), {"../outside": "0" * 64})

    def test_summary_mismatch_is_detected(self):
        with self.assertRaises(ValueError):
            close(0.1, 0.2, "auc")


if __name__ == "__main__":
    unittest.main()

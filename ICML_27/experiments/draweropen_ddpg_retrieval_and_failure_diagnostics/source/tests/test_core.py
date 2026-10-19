"""Lightweight tests allowed while PAL is active; standard library only."""
import ast
import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import after_pal
from analyze_failures import (DIAGNOSTICS, curve_metrics, temporal_relation,
                              validate_windows, weighted_metric, training_episode_rows)
from common import PAL_SHA, SEEDS, TOTAL_UPDATES, write_json, output_directory
from freeze import verify_upstream


def make_pal(root, complete=True):
    state = {"source_manifest_sha256": PAL_SHA, "status": "complete" if complete else "production_running",
             "stage": "analysis" if complete else "production",
             "authorized_production": {"arm": "pal", "seeds": list(SEEDS),
                                       "steps_per_seed": 2_000_000, "parallel": 5},
             "stages": [{"name": "production", "returncode": 0}, {"name": "analysis", "returncode": 0}]}
    write_json(root / "outputs/automation/status.json", state)
    for seed in SEEDS:
        attempt = root / f"outputs/production/pal/seed_{seed:02d}/attempt_01"
        write_json(attempt / "status.json", {"source_manifest_sha256": PAL_SHA, "profile": "production",
                   "arm": "pal", "seed": seed, "checkpoint_continuation": False,
                   "status": "complete" if complete else "running"})
        write_json(attempt / "run/summary.json", {"total_steps": 2_000_000,
                                                   "total_updates": TOTAL_UPDATES})
    write_json(root / "outputs/analysis/report.json", {"source_manifest_sha256": PAL_SHA,
               "problems": [], "reference_reuse_audit_passed": True, "inference": {"mean": 0.0},
               "runs": [{"seed": s, "metrics": {"auc": 0.0}} for s in SEEDS]})
    return state


class QueueTests(unittest.TestCase):
    def test_running_pal_waits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_pal(root, complete=False)
            self.assertFalse(after_pal.inspect_pal(root)["ready"])

    def test_all_zero_results_do_not_block(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_pal(root)
            self.assertTrue(after_pal.inspect_pal(root)["ready"])

    def test_analysis_required_even_after_ten_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = make_pal(root)
            state["status"] = "validating"
            write_json(root / "outputs/automation/status.json", state)
            self.assertFalse(after_pal.inspect_pal(root)["ready"])

    def test_missing_seed_blocks_complete_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_pal(root)
            (root / "outputs/production/pal/seed_10/attempt_01/status.json").unlink()
            with self.assertRaises(ValueError):
                after_pal.inspect_pal(root)

    def test_failed_pal_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = make_pal(root)
            state["status"] = "blocked"
            write_json(root / "outputs/automation/status.json", state)
            with self.assertRaises(RuntimeError):
                after_pal.inspect_pal(root)

    def test_source_change_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = make_pal(root)
            state["source_manifest_sha256"] = "changed"
            write_json(root / "outputs/automation/status.json", state)
            with self.assertRaises(ValueError):
                after_pal.inspect_pal(root)

    def test_continuation_or_extra_attempt_blocks(self):
        for resumed in (True, False):
            with self.subTest(resumed=resumed), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                make_pal(root)
                path = root / "outputs/production/pal/seed_01/attempt_01/status.json"
                if resumed:
                    value = json.loads(path.read_text())
                    value["checkpoint_continuation"] = True
                    write_json(path, value)
                else:
                    (path.parents[1] / "attempt_02").mkdir()
                with self.assertRaises(ValueError):
                    after_pal.inspect_pal(root)

    def test_incomplete_budget_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_pal(root)
            write_json(root / "outputs/production/pal/seed_10/attempt_01/run/summary.json",
                       {"total_steps": 1_000_000, "total_updates": 995001})
            with self.assertRaises(ValueError):
                after_pal.inspect_pal(root)

    def test_analysis_failure_or_missing_inference_blocks(self):
        for changed in ({"problems": ["missing"]}, {"inference": None},
                        {"reference_reuse_audit_passed": False}):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                make_pal(root)
                path = root / "outputs/analysis/report.json"
                report = json.loads(path.read_text())
                report.update(changed)
                write_json(path, report)
                with self.assertRaises(ValueError):
                    after_pal.inspect_pal(root)

    def test_wait_module_does_not_import_ml(self):
        code = "import after_pal,sys; assert not any(k in sys.modules for k in ('torch','numpy','numba','mujoco'))"
        subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True)

    def test_pipeline_does_not_train(self):
        for name, command, timeout in after_pal.pipeline():
            self.assertFalse(any(Path(word).name in {"train.py", "launch.py"} for word in command))
            self.assertGreater(timeout, 0)

    def test_stage_failure_stops_pipeline(self):
        names = []

        def fail(name, command, timeout):
            names.append(name)
            raise RuntimeError("synthetic test failure")

        with self.assertRaises(RuntimeError):
            after_pal.run_pipeline(fail)
        self.assertEqual(names, ["tests"])

    def test_busy_upstream_does_not_spawn(self):
        runner = after_pal.Runner("snapshot")
        with mock.patch.object(after_pal, "verify", return_value="snapshot"), \
                mock.patch.object(after_pal, "inspect_pal", return_value={"ready": True}), \
                mock.patch.object(after_pal, "upstream_processes", return_value=[123]), \
                mock.patch.object(after_pal.subprocess, "Popen") as popen:
            with self.assertRaises(RuntimeError):
                runner.run_stage("tests", ["never"], 1)
            popen.assert_not_called()

    def test_idle_gpu_guard_rejects_unknown_output(self):
        result = subprocess.CompletedProcess([], 0, stdout="Not Supported\n", stderr="")
        with mock.patch.object(after_pal.subprocess, "run", return_value=result):
            with self.assertRaises(RuntimeError):
                after_pal.gpu_compute_pids()

    def test_stop_file_is_respected(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = after_pal.Runner("snapshot")
            runner.folder = Path(folder)
            (runner.folder / "STOP").touch()
            with self.assertRaises(InterruptedError):
                runner.check_stop()

    def test_upstream_copy_is_byte_identical(self):
        verify_upstream()

    def test_scripts_parse_without_executing(self):
        for path in ROOT.glob("*.py"):
            ast.parse(path.read_text(), filename=str(path))

    def test_no_outputs_into_original_experiments(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                output_directory(Path(folder) / "new")


def window(first, last, informative):
    count = last - first + 1
    row = {"start_update": str(first), "end_update": str(last), "step": str(last + 4999),
           "subset": "all", "n_updates": str(count)}
    for key in DIAGNOSTICS:
        row[key + "_n"] = str(count)
        row[key + "_mean"] = "0.5"
    row.update(empty_update_mean="0", informative_update_mean=str(informative / count),
               saturated_nonempty_update_mean=str(1 - informative / count))
    subset = copy.deepcopy(row)
    subset.update(subset="informative", n_updates=str(informative))
    for key in DIAGNOSTICS:
        subset[key + "_n"] = str(informative)
        subset[key + "_mean"] = "0.5" if informative else ""
    return row, subset


class AnalysisTests(unittest.TestCase):
    def test_eval_rows_do_not_duplicate_training_episodes(self):
        episode = {"step": "10000", "train/episode_success": "1", "train/episode_len": "200",
                   "eval/env_success_mean": ""}
        evaluation = {"step": "10000", "train/episode_success": "0", "train/episode_len": "0",
                      "eval/env_success_mean": "0.5"}
        self.assertEqual(training_episode_rows([episode, evaluation]), [episode])

    def test_weighted_means_use_counts_not_window_means(self):
        a, _ = window(1, 1000, 1000)
        b, _ = window(1001, 1001, 1)
        a["support_rate_mean"], b["support_rate_mean"] = "0", "1"
        self.assertAlmostEqual(weighted_metric([a, b], "support_rate"), 1 / 1001)

    def test_empty_denominators_are_none(self):
        _, subset = window(1, 10, 0)
        self.assertIsNone(weighted_metric([subset], "td_overlap_fraction"))

    def test_contiguous_windows_and_final_short_window(self):
        a, b = window(1, 1000, 1000)
        c, d = window(1001, 1001, 0)
        whole, informative = validate_windows([a, b, c, d], expected_updates=1001)
        self.assertEqual(len(whole), 2)
        self.assertEqual(int(informative[-1]["n_updates"]), 0)

    def test_missing_or_overlapping_windows_rejected(self):
        a, b = window(2, 10, 9)
        with self.assertRaises(ValueError):
            validate_windows([a, b], expected_updates=10)

    def test_nonfinite_is_not_imputed(self):
        a, _ = window(1, 10, 10)
        a["support_rate_mean"] = "nan"
        with self.assertRaises(ValueError):
            weighted_metric([a], "support_rate")

    def test_update_partition_rejected(self):
        a, b = window(1, 10, 10)
        a["empty_update_mean"] = "1"
        with self.assertRaises(ValueError):
            validate_windows([a, b], expected_updates=10)

    def test_temporal_order_is_interval_censored(self):
        self.assertEqual(temporal_relation(150, 100, 200), "same_window_order_unresolved")
        self.assertEqual(temporal_relation(None, 100, 200), "no_positive_evaluation_observed")
        self.assertEqual(temporal_relation(99, 100, 200), "positive_evaluation_before_first_saturated_window")

    def test_auc_exact_grid_and_denominator(self):
        points = [(s, (s - 10000) / 1990000) for s in range(10000, 2000001, 10000)]
        result = curve_metrics(points)
        self.assertAlmostEqual(result["auc"], .5)
        self.assertEqual(result["final_success"], 1)
        with self.assertRaises(ValueError):
            curve_metrics(points[1:])

    def test_zero_success_is_a_valid_outcome(self):
        result = curve_metrics([(s, 0.0) for s in range(10000, 2000001, 10000)])
        self.assertEqual(result["auc"], 0)
        self.assertTrue(result["zero_evaluation_success"])
        self.assertIsNone(result["hit90"])

    def test_evaluation_nan_fails(self):
        points = [(s, 0.0) for s in range(10000, 2000001, 10000)]
        points[0] = (10000, math.nan)
        with self.assertRaises(ValueError):
            curve_metrics(points)


if __name__ == "__main__":
    unittest.main()

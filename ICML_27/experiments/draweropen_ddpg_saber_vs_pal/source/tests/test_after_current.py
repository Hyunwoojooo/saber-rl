"""Standard-library-only tests of scheduling, never experiment execution."""
import json
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import after_current as queue


class QueueTests(unittest.TestCase):
    def fixture(self, root, complete=0):
        production = Path(root) / "outputs/production"
        queue.write_json(production / "plan.json", {
            "arms": list(queue.ARMS), "seeds": list(range(1, 11)), "profile": "production",
            "source_manifest_sha256": queue.REFERENCE_SHA})
        paths = []
        for seed in range(1, 11):
            for arm in queue.ARMS:
                p = production / arm / f"seed_{seed:02d}/attempt_01/status.json"
                paths.append(p)
                if len(paths) <= complete:
                    queue.write_json(p, {"arm": arm, "seed": seed, "profile": "production",
                        "source_manifest_sha256": queue.REFERENCE_SHA, "status": "complete",
                        "checkpoint_continuation": False, "metrics": {"auc": 0.0}})
        return paths

    def test_missing_jobs_wait_and_zero_success_is_complete(self):
        with tempfile.TemporaryDirectory() as root:
            self.fixture(root, 23)
            self.assertEqual(queue.inspect_reference(root),
                             {"complete": 23, "running": 0, "pending": 7, "total": 30})

    def test_all_thirty_required(self):
        with tempfile.TemporaryDirectory() as root:
            self.fixture(root, 30)
            self.assertEqual(queue.inspect_reference(root)["complete"], 30)

    def test_failure_or_source_change_blocks(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self.fixture(root, 30)
            original = queue.read_json(paths[0])
            for changed in ({"status": "failed_learning"}, {"source_manifest_sha256": "wrong"},
                            {"checkpoint_continuation": True}, {"arm": "wrong"}):
                queue.write_json(paths[0], {**original, **changed})
                with self.assertRaises(ValueError):
                    queue.inspect_reference(root)

    def test_undocumented_additional_attempt_blocks(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self.fixture(root, 30)
            (paths[0].parent.parent / "attempt_02").mkdir()
            with self.assertRaises(ValueError):
                queue.inspect_reference(root)

    def test_reference_process_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / "ref"
            reference.mkdir()
            proc = Path(tmp) / "proc"
            for pid, args, cwd in ((101, ["python", "launch.py"], reference),
                                   (102, ["python", str(reference / "train.py")], reference),
                                   (103, ["python", "other.py"], Path(tmp))):
                d = proc / str(pid)
                d.mkdir(parents=True)
                (d / "cmdline").write_bytes("\0".join(args).encode())
                (d / "cwd").symlink_to(cwd)
            self.assertEqual(queue.reference_processes(reference, proc), [101, 102])

    def test_no_gpu_work_before_wait_loop_finishes(self):
        names = [s[0] for s in queue.pipeline()]
        self.assertLess(names.index("cpu_tests"), names.index("cuda_loss"))
        self.assertLess(names.index("reference"), names.index("cuda_smoke"))
        self.assertLess(names.index("readiness"), names.index("production"))
        production = next(s for s in queue.pipeline() if s[0] == "production")
        self.assertIn("--execute", production[1])
        self.assertEqual(production[1][production[1].index("--parallel") + 1], "5")

    def test_failure_at_any_gate_never_reaches_production(self):
        stages = [s[0] for s in queue.pipeline()]
        for failed in stages[:stages.index("production")]:
            seen = []

            def fail_stage(name, *args):
                seen.append(name)
                if name == failed:
                    raise RuntimeError("synthetic failure")

            with self.assertRaises(RuntimeError):
                queue.run_pipeline(fail_stage)
            self.assertNotIn("production", seen)

    def test_occupied_or_unreadable_gpu_blocks(self):
        for output in ("123\n", "Not Supported\n"):
            result = subprocess.CompletedProcess([], 0, output, "")
            with patch.object(queue.subprocess, "run", return_value=result), self.assertRaises(RuntimeError):
                queue.require_idle_gpu()
        with patch.object(queue.subprocess, "run", side_effect=subprocess.CalledProcessError(1, [])):
            with self.assertRaises(subprocess.CalledProcessError):
                queue.require_idle_gpu()
        with patch.object(queue.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            queue.require_idle_gpu()

    def test_stop_file_and_no_automatic_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = queue.Runner(tmp, "synthetic")
            (Path(tmp) / "STOP").touch()
            with self.assertRaises(InterruptedError):
                runner.run_stage("production", [], 1, False)
            self.assertEqual(runner.state["stages"], [])

    def test_waiter_does_not_import_ml_or_simulator(self):
        import ast
        tree = ast.parse((queue.ROOT / "after_current.py").read_text())
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(imports & {"torch", "numpy", "gymnasium", "metaworld", "train", "protocol"})

    def test_early_checkpoint_check_is_not_a_performance_gate(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "pal/seed_01/attempt_01/run/metrics.csv"
            path.parent.mkdir(parents=True)
            fields = ["step", "eval/env_success_mean", "loss/critic", "loss/actor", "pal/base_loss",
                      "pal/normalizer", "critic/q_policy_mean", "critic/q_policy_std"]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for step in (10000, 20000):
                    writer.writerow(dict(zip(fields, [step, 0, 0.5, 1, 1, 2, -1, 0.5])))
            result = queue.audit_early_checkpoints(root, {})
            self.assertTrue(result["1"]["passed"])
            self.assertEqual([p["success"] for p in result["1"]["points"]], [0, 0])
            self.assertEqual(queue.audit_early_checkpoints(root, result), result)


if __name__ == "__main__":
    unittest.main()

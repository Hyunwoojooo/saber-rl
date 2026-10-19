import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_contract import ROOT
import dependency
import scheduled_run
import gates
from protocol import SPEC, SCHEDULE, make_config
from analyze import paired_intervals


class ScheduleContract(unittest.TestCase):
    def state(self, folder, status="running"):
        state = {"status": status, "pid": 12345, "stage": "production_drawer_close",
                 "source_manifest_sha256": SCHEDULE["predecessor_source_manifest_sha256"],
                 "recovery_manifest_sha256": SCHEDULE["predecessor_recovery_manifest_sha256"],
                 "stages": [{"name": n, "returncode": 0} for n in SCHEDULE["predecessor_required_stages"]]}
        target = folder / "recovery_state/status.json"
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(state))
        return state, target

    def test_running_waits_without_reading_partial_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.state(root)
            with patch.object(dependency, "alive", return_value=True), patch.object(dependency, "read_references") as read:
                self.assertEqual(dependency.predecessor_state(root)["state"], "waiting")
                read.assert_not_called()
            with patch.object(dependency, "alive", return_value=False), self.assertRaises(RuntimeError):
                dependency.predecessor_state(root)

    def test_completion_requires_all_stages_hashes_and_all_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, path = self.state(root, "complete")
            with patch.object(dependency, "alive", return_value=False), patch.object(dependency, "check_manifest") as manifest, patch.object(dependency, "read_references", return_value=[]) as read:
                self.assertEqual(dependency.predecessor_state(root)["state"], "ready")
                self.assertEqual(manifest.call_count, 2)
                read.assert_called_once_with(root)
                state["stages"][-1]["returncode"] = 1
                path.write_text(json.dumps(state))
                with self.assertRaises(RuntimeError):
                    dependency.predecessor_state(root)
            for status in ("failed", "blocked", "interrupted"):
                self.state(root, status)
                with self.assertRaises(RuntimeError):
                    dependency.predecessor_state(root)

    def test_manifest_and_identity_mismatch_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, path = self.state(root)
            state["source_manifest_sha256"] = "changed"
            path.write_text(json.dumps(state))
            with self.assertRaises(RuntimeError):
                dependency.predecessor_state(root)
            (root / "x.py").write_text("pass\n")
            m = root / "manifest.json"
            m.write_text(json.dumps({"sha256": {"x.py": hashlib.sha256(b"pass\n").hexdigest()}}))
            digest = hashlib.sha256(m.read_bytes()).hexdigest()
            dependency.check_manifest(root, m.name, digest)
            (root / "x.py").write_text("changed\n")
            with self.assertRaises(RuntimeError):
                dependency.check_manifest(root, m.name, digest)

    def test_incomplete_reference_cohort_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "analysis").mkdir()
            (root / "analysis/final.json").write_text(json.dumps({"status": "completed", "incomplete": [], "runs": []}))
            with self.assertRaises(RuntimeError):
                dependency.read_references(root)

    def test_reference_config_parity_rejects_scientific_change(self):
        for task in SPEC["tasks"]:
            for method in ("her", "saber"):
                cfg = json.loads(json.dumps(make_config(task, method, 1).as_dict()))
                cfg["output_root"] = "/old/results"
                cfg["experiment_id"] = "reference"
                dependency.reference_config_parity(cfg, task, method, 1)
                cfg["gamma"] = .99
                with self.assertRaises(ValueError):
                    dependency.reference_config_parity(cfg, task, method, 1)

    def test_validation_first_open_before_close_fixed_no_extension(self):
        plan = scheduled_run.stage_plan()
        self.assertEqual([n for n, _ in plan], ["cpu_revalidation", "gpu_validation",
            "production_drawer_open", "analysis_drawer_open", "production_drawer_close",
            "analysis_drawer_close", "final_analysis"])
        self.assertEqual(SPEC["seeds"], [1, 2, 3, 4, 5])
        self.assertEqual(SPEC["methods"], ["hgr", "per"])
        self.assertFalse(SPEC["automatic_seed_extension"])
        self.assertFalse(SCHEDULE["retry_failed_runs"])
        self.assertEqual(SCHEDULE["production_parallel"], 5)

    def test_busy_gpu_defers_and_query_error_blocks(self):
        from types import SimpleNamespace
        with patch.object(scheduled_run.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="123\n", stderr="")):
            self.assertEqual(scheduled_run.resource_state()["state"], "waiting")
        with patch.object(scheduled_run.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="", stderr="failed")):
            with self.assertRaises(RuntimeError):
                scheduled_run.resource_state()

    def test_cpu_gate_rejects_stale_hash_and_unit_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "validation/validation_report.json"
            gates.write_atomic(path, {"status": "unit_only_passed", "source_manifest_sha256": "x"})
            with self.assertRaises(RuntimeError):
                gates.require_cpu_gate("x", root)
            gates.write_atomic(path, {"status": "passed", "source_manifest_sha256": "x"})
            gates.require_cpu_gate("x", root)
            with self.assertRaises(RuntimeError):
                gates.require_cpu_gate("changed", root)

    def valid_gpu_report(self):
        return {"status": "passed", "source_manifest_sha256": "x",
                "cuda_update_parity": {"status": "passed", "updates": 3, "priority_modes": [None, "min_abs", "mean_abs"]},
                "reference_environment_audit": {"status": "passed", "reference_runs": 20},
                "runs": [{"profile": profile, "task": task, "method": method, "seed": 1,
                    "status": "passed", "steps": SCHEDULE[f"{profile}_steps"], "device": "cuda",
                    "checkpoint_verified": True, "sha256": {n: "a" * 64 for n in ("config.json", "summary.json", "metrics.csv", "checkpoint.pt")}}
                    for profile in ("gpu_smoke", "stability") for task in SPEC["tasks"] for method in SPEC["methods"]]}

    def test_gpu_gate_requires_all_eight_profiles_and_parity(self):
        good = self.valid_gpu_report()
        gates.validate_gpu_report(good, "x")
        for mutate in (lambda r: r["runs"].pop(), lambda r: r["runs"][0].update(status="failed"),
                       lambda r: r["cuda_update_parity"].update(priority_modes=[]),
                       lambda r: r["reference_environment_audit"].update(status="failed")):
            bad = copy.deepcopy(good)
            mutate(bad)
            with self.assertRaises(RuntimeError):
                gates.validate_gpu_report(bad, "x")

    def test_complete_seed_bootstrap_adjustment(self):
        ci = paired_intervals([.1, -.1, .2, -.3, .4])
        self.assertLessEqual(ci["ci98_75"][0], ci["ci95"][0])
        self.assertGreaterEqual(ci["ci98_75"][1], ci["ci95"][1])
        with self.assertRaises(ValueError):
            paired_intervals([.1, .2, .3, .4])


if __name__ == "__main__":
    unittest.main()

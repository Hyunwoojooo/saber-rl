import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gates
import gpu_validate
import scheduled_run as queue
from protocol import SPEC, SCHEDULE, make_config, make_gpu_validation_config


def gpu_report(root):
    rows = []
    for profile in ("gpu_smoke", "stability"):
        for task in SPEC["tasks"]:
            for method in SPEC["methods"]:
                folder = root / "validation" / f"{profile}_{task}_{method}"
                hashes = {}
                for name in ("config.json", "summary.json", "metrics.csv", "checkpoint.pt"):
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / name).write_bytes(b"test evidence\n")
                    hashes[name] = hashlib.sha256(b"test evidence\n").hexdigest()
                rows.append(dict(profile=profile, task=task, method=method,
                                 seed=SCHEDULE["validation_seed"], status="passed", device="cuda",
                                 steps=SCHEDULE[f"{profile}_steps"], checkpoint_verified=True,
                                 run_dir=str(folder), sha256=hashes, metrics={"auc": 0., "final": 0.}))
    return dict(status="passed", source_manifest_sha256="snapshot", runs=rows,
                cuda_update_parity={"status": "passed", "updates": 3})


def prior_fixture(root):
    (root / "trainer.py").write_text("# frozen test source\n")
    gates.write_atomic(root / "source_manifest.json", {"sha256": {
        "trainer.py": hashlib.sha256((root / "trainer.py").read_bytes()).hexdigest()}})
    digest = hashlib.sha256((root / "source_manifest.json").read_bytes()).hexdigest()
    gates.write_atomic(root / "outputs/automation/status.json", {
        "status": "complete", "pid": 99999999, "source_manifest_sha256": digest,
        "stages": [{"stage": s, "returncode": 0} for s in SCHEDULE["predecessor_required_stages"]]})
    for seed in range(1, 11):
        dest = root / f"outputs/production/random512/seed_{seed:02d}/attempt_01"
        gates.write_atomic(dest / "status.json", {"status": "complete", "source_manifest_sha256": digest})
        gates.write_atomic(dest / "run/summary.json", {"total_steps": 2000000, "total_updates": 1995001})
    gates.write_atomic(root / "outputs/analysis/report.json", {
        "failures_or_missing": [], "runs": [{}] * 20, "inference": {"test": 1}})
    gates.write_atomic(root / "outputs/final_retrieval/report.json", {
        "complete": True, "training_checkpoints": 20, "tail_parity_mismatches": 0})
    return digest


class ScheduleProfiles(unittest.TestCase):
    def test_five_seeds_only_and_no_automatic_extension(self):
        self.assertEqual(SPEC["seeds"], [1, 2, 3, 4, 5])
        self.assertIs(SPEC["automatic_seed_extension"], False)
        plan = subprocess.run([sys.executable, "-B", str(ROOT / "launch.py"), "--tasks", "drawer_open"],
                              capture_output=True, text=True, check=True, timeout=30)
        report = json.loads(plan.stdout)
        self.assertEqual((report["runs"], report["steps"]), (10, 20000000))
        self.assertFalse(any(f"_s{s}_" in name for s in range(6, 11) for name in report["names"]))
        for script, args in (("launch.py", ["--seeds", "6"]),
                             ("train.py", ["--task", "drawer_open", "--method", "her", "--seed", "6"])):
            result = subprocess.run([sys.executable, "-B", str(ROOT / script), *args],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid choice", result.stderr)

    def test_algorithm_and_common_settings_unchanged_from_v1(self):
        prior = ROOT.parent / "2026-10-08_drawer_sac_fpp_v1"
        for source in (ROOT / "code").rglob("*.py"):
            self.assertEqual(source.read_bytes(), (prior / source.relative_to(ROOT)).read_bytes())
        original = json.loads((prior / "protocol.json").read_text())
        for key in ("common", "runtime", "tasks", "methods", "statistics"):
            self.assertEqual(SPEC[key], original[key])

    def test_stability_retains_production_settings_except_validation_fields(self):
        for task in SPEC["tasks"]:
            for method in SPEC["methods"]:
                prod = make_config(task, method, 1)
                cfg = make_gpu_validation_config(task, method, 1, "stability", ROOT / "validation/test")
                a, b = prod.as_dict(), cfg.as_dict()
                difference = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
                self.assertEqual(difference, {"total_steps", "run_name", "output_root",
                                              "validation_only", "validation_profile"})
                self.assertEqual(cfg.total_steps, 100000)
                self.assertEqual(cfg.total_steps - cfg.warmup_steps + 1, 95001)
                self.assertFalse(cfg.resume)

    def test_gpu_smoke_is_separate_and_uses_cuda(self):
        cfg = make_gpu_validation_config("drawer_open", "saber", 1, "gpu_smoke", ROOT / "validation/test")
        self.assertEqual((cfg.device, cfg.total_steps, cfg.warmup_steps), ("cuda", 600, 256))
        self.assertTrue(cfg.validation_only)
        self.assertTrue(cfg.run_name.endswith("gpu_smoke"))
        with self.assertRaises(ValueError):
            make_gpu_validation_config("drawer_open", "saber", 2, "stability", ROOT)

    def test_stage_order_open_before_close_and_final_analysis(self):
        self.assertEqual([n for n, _ in queue.stage_plan()], [
            "cpu_revalidation", "gpu_validation", "production_drawer_open", "analysis_drawer_open",
            "production_drawer_close", "analysis_drawer_close", "final_analysis"])
        for name, args in queue.stage_plan():
            if name.startswith("production_"):
                self.assertEqual(args[args.index("--parallel") + 1], "5")
                self.assertNotIn("--seeds", args)
                self.assertNotIn("--methods", args)

    def test_waiter_import_does_not_import_torch_or_numpy(self):
        code = "import scheduled_run, sys; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True, timeout=30)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.report = gpu_report(self.root)
        gates.write_atomic(self.root / "validation/validation_report.json",
                           {"status": "passed", "source_manifest_sha256": "snapshot"})
        gates.write_atomic(self.root / "validation/gpu_validation_report.json", self.report)

    def test_complete_zero_success_report_is_accepted(self):
        self.assertEqual(gates.require_production_gates("snapshot", self.root), self.report)

    def test_wrong_source_and_unit_only_cpu_do_not_pass(self):
        with self.assertRaises(RuntimeError):
            gates.require_production_gates("changed", self.root)
        gates.write_atomic(self.root / "validation/validation_report.json",
                           {"status": "unit_only_passed", "source_manifest_sha256": "snapshot"})
        with self.assertRaises(RuntimeError):
            gates.require_production_gates("snapshot", self.root)

    def test_missing_duplicate_failed_stale_or_incomplete_gpu_evidence_rejected(self):
        variants = []
        a = copy.deepcopy(self.report)
        a["runs"].pop()
        variants.append(a)
        a = copy.deepcopy(self.report)
        a["runs"][-1] = copy.deepcopy(a["runs"][0])
        variants.append(a)
        for key, value in (("status", "failed"), ("source_manifest_sha256", "stale"), ("cuda_update_parity", {})):
            a = copy.deepcopy(self.report)
            a[key] = value
            variants.append(a)
        for key, value in (("device", "cpu"), ("steps", 500), ("seed", 2),
                           ("checkpoint_verified", False), ("status", "failed"), ("sha256", {})):
            a = copy.deepcopy(self.report)
            a["runs"][0][key] = value
            variants.append(a)
        for report in variants:
            with self.subTest(report=report), self.assertRaises(RuntimeError):
                gates.validate_gpu_report(report, "snapshot")

    def test_changed_checkpoint_is_rejected(self):
        (Path(self.report["runs"][0]["run_dir"]) / "checkpoint.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "evidence changed"):
            gates.require_production_gates("snapshot", self.root)

    def test_evidence_outside_validation_is_rejected(self):
        self.report["runs"][0]["run_dir"] = str(self.root)
        gates.write_atomic(self.root / "validation/gpu_validation_report.json", self.report)
        with self.assertRaisesRegex(RuntimeError, "experiment-local"):
            gates.require_production_gates("snapshot", self.root)


class PredecessorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.digest = prior_fixture(self.root)
        self.status = self.root / "outputs/automation/status.json"
        self.addCleanup(patch.stopall)
        self.alive = patch.object(queue, "alive", return_value=False).start()

    def read(self):
        return queue.predecessor_state(self.root, self.digest)

    def test_ready_requires_all_runs_and_both_post_analyses(self):
        self.assertEqual(self.read()["state"], "ready")
        gates.write_atomic(self.root / "outputs/final_retrieval/report.json", {"complete": False})
        with self.assertRaises(RuntimeError):
            self.read()

    def test_running_waits_but_dead_process_blocks(self):
        state = json.loads(self.status.read_text())
        state.update(status="running", stage="training")
        gates.write_atomic(self.status, state)
        self.alive.return_value = True
        self.assertEqual(self.read()["state"], "waiting")
        self.alive.return_value = False
        with self.assertRaisesRegex(RuntimeError, "orchestrator is absent"):
            self.read()

    def test_blocked_predecessor_and_failed_stage_block(self):
        state = json.loads(self.status.read_text())
        state["status"] = "blocked"
        gates.write_atomic(self.status, state)
        with self.assertRaises(RuntimeError):
            self.read()
        state["status"] = "complete"
        state["stages"][-1]["returncode"] = 1
        gates.write_atomic(self.status, state)
        with self.assertRaises(RuntimeError):
            self.read()

    def test_incomplete_seed_or_incomplete_analysis_blocks(self):
        gates.write_atomic(self.root / "outputs/production/random512/seed_10/attempt_01/run/summary.json",
                           {"total_steps": 1999999, "total_updates": 1995000})
        with self.assertRaises(RuntimeError):
            self.read()
        self.digest = prior_fixture(self.root)
        gates.write_atomic(self.root / "outputs/analysis/report.json", {"failures_or_missing": ["seed10"]})
        with self.assertRaises(RuntimeError):
            self.read()

    def test_modified_prior_source_blocks(self):
        (self.root / "trainer.py").write_text("# changed\n")
        with self.assertRaisesRegex(RuntimeError, "source changed"):
            self.read()

    def test_partial_status_write_waits_and_finishing_process_waits(self):
        self.alive.return_value = True
        self.assertEqual(self.read()["reason"], "predecessor finishing shutdown")
        self.status.write_text("{")
        self.assertEqual(self.read()["state"], "waiting")

    def test_duplicate_attempt_blocks(self):
        (self.root / "outputs/production/random512/seed_01/attempt_02").mkdir()
        with self.assertRaisesRegex(RuntimeError, "attempt count"):
            self.read()


class QueueExecutionTests(unittest.TestCase):
    def fake_queue(self, folder, codes):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(queue, "ROOT", folder))
        stack.enter_context(patch.object(queue, "verify", return_value="snapshot"))
        stack.enter_context(patch.object(queue, "require_cpu_gate"))
        gate = stack.enter_context(patch.object(queue, "require_production_gates"))
        stack.enter_context(patch.object(queue, "predecessor_state", return_value={"state": "ready"}))
        stack.enter_context(patch.object(queue, "resource_state", return_value={"state": "ready"}))
        stack.enter_context(patch.object(queue.signal, "signal"))
        stack.enter_context(patch.object(sys, "argv", ["scheduled_run.py", "--execute"]))
        processes = [SimpleNamespace(pid=123, returncode=c, poll=lambda c=c: c) for c in codes]
        popen = stack.enter_context(patch.object(queue.subprocess, "Popen", side_effect=processes))
        return popen, gate

    def test_gpu_failure_cannot_start_production(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            popen, gate = self.fake_queue(root, [0, 1])
            with self.assertRaisesRegex(RuntimeError, "gpu_validation failed"):
                queue.main()
            self.assertEqual(popen.call_count, 2)
            gate.assert_not_called()
            state = json.loads((root / "automation/status.json").read_text())
            self.assertEqual(state["status"], "blocked")

    def test_all_gates_pass_runs_open_then_close_then_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            popen, gate = self.fake_queue(root, [0] * 7)
            queue.main()
            self.assertEqual(popen.call_count, 7)
            self.assertEqual(gate.call_count, 2)
            state = json.loads((root / "automation/status.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual([s["name"] for s in state["stages"]], [n for n, _ in queue.stage_plan()])
            self.assertEqual(popen.call_args_list[0].kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(popen.call_args_list[1].kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
            with self.assertRaises(FileExistsError):
                queue.main()

    def test_gpu_probe_checks_predecessor_before_cuda(self):
        with patch.object(sys, "argv", ["gpu_validate.py", "--execute"]), \
             patch.object(gpu_validate, "verify", return_value="snapshot"), \
             patch.object(gpu_validate, "require_cpu_gate"), \
             patch.object(queue, "predecessor_state", return_value={"state": "waiting"}), \
             patch.object(gpu_validate, "cuda_update_parity") as probe:
            with self.assertRaisesRegex(RuntimeError, "Predecessor not complete"):
                gpu_validate.main()
            probe.assert_not_called()

    def test_busy_gpu_waits_and_unreadable_gpu_blocks(self):
        with patch.object(queue.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="123\n", stderr="")):
            self.assertEqual(queue.resource_state(), {
                "state": "waiting", "reason": "GPU has active compute processes", "pids": [123]})
        with patch.object(queue.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="N/A\n", stderr="")):
            with self.assertRaises(RuntimeError):
                queue.resource_state()


if __name__ == "__main__":
    unittest.main()

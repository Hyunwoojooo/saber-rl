import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("recovery_manage", Path(__file__).with_name("manage.py"))
manage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manage)


class RecoveryTests(unittest.TestCase):
    def test_exact_partition(self):
        manage.check_plan()
        self.assertEqual(len(manage.pending_jobs("drawer_open")), 7)
        self.assertEqual(len(manage.pending_jobs("drawer_close")), 10)

    def test_completed_poor_seeds_are_reused(self):
        for seed in (1, 2, 3):
            self.assertIn(["drawer_open", "her", seed], manage.PLAN["reuse"])
            self.assertNotIn(("drawer_open", "her", seed), manage.pending_jobs("drawer_open"))

    def test_original_order_without_reused_runs(self):
        self.assertEqual(manage.pending_jobs("drawer_open"), [
            ("drawer_open", "saber", 1), ("drawer_open", "saber", 2),
            ("drawer_open", "saber", 3), ("drawer_open", "her", 4),
            ("drawer_open", "saber", 4), ("drawer_open", "her", 5), ("drawer_open", "saber", 5)])

    def test_no_seed_extension(self):
        plan = copy.deepcopy(manage.PLAN)
        plan["fresh_unstarted"].append(["drawer_close", "her", 6])
        with self.assertRaises(ValueError):
            manage.check_plan(plan)

    def test_no_continuation(self):
        for key in ("checkpoint_continuation", "automatic_seed_extension", "performance_based_replacement"):
            plan = copy.deepcopy(manage.PLAN)
            plan[key] = True
            with self.assertRaises(ValueError):
                manage.check_plan(plan)

    def test_cannot_skip_an_interrupted_seed(self):
        plan = copy.deepcopy(manage.PLAN)
        plan["rerun_from_zero"].pop()
        with self.assertRaises(ValueError):
            manage.check_plan(plan)

    def test_cannot_reuse_an_interrupted_seed(self):
        plan = copy.deepcopy(manage.PLAN)
        item = plan["rerun_from_zero"].pop()
        plan["reuse"].append(item)
        with self.assertRaises(ValueError):
            manage.check_plan(plan)

    def test_environment_mismatch_blocks(self):
        manage.require_environment({"torch": "same"}, {"torch": "same"})
        with self.assertRaises(RuntimeError):
            manage.require_environment({"torch": "same"}, {"torch": "changed"})

    def test_partial_tail_read_is_non_destructive(self):
        original = b"step,x\n200,1\n400,2\n600," + b"\x00" * 10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_bytes(original)
            result = manage.scan_partial_csv(path)
            self.assertEqual(result["last_complete_step"], 400)
            self.assertEqual(result["partial_record_bytes"], 4)
            self.assertEqual(result["null_padding_bytes"], 10)
            self.assertEqual(path.read_bytes(), original)

    def test_internal_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_bytes(b"step,x\n200,1\n\x00400,2\n")
            with self.assertRaises(ValueError):
                manage.scan_partial_csv(path)

    def test_result_validation_uses_original_output_root(self):
        import protocol
        job = ("drawer_open", "her", 2)
        cfg = protocol.make_config(*job, output_root=manage.PARENT / "runs")
        self.assertFalse(cfg.resume)
        self.assertEqual(cfg.total_steps, 2000000)
        self.assertEqual(cfg.seeds.env_seed, 2)
        self.assertEqual(Path(cfg.output_root), manage.PARENT / "runs")

    def test_missing_pair_withholds_inference(self):
        import analyze
        with self.assertRaises(ValueError):
            analyze.paired_intervals([0.1, 0.2, 0.3, 0.4])

    def test_fresh_config_only_changes_output_location(self):
        import json
        import protocol
        for job in manage.PLAN["rerun_from_zero"]:
            old = json.loads((manage.PARENT / "runs" / manage.name(job) / "config.json").read_text())
            new = json.loads(json.dumps(protocol.make_config(*job).as_dict()))
            metadata = {"thread_hint", "terminal_convention", "sac_target_entropy_effective"}
            self.assertEqual(set(old) - set(new), metadata)
            self.assertEqual(old["sac_target_entropy_effective"], new["sac_target_entropy"])
            self.assertEqual(old["terminal_convention"], {
                "success_terminates": new["success_terminates"], "timeout_bootstrap": new["timeout_bootstrap"]})
            self.assertTrue(all(v == "1" for v in old["thread_hint"].values()))
            differences = {k for k in new if new[k] != old.get(k)}
            self.assertEqual(differences, {"output_root"})
            self.assertFalse(new["resume"])
            self.assertIsNone(new["resume_path"])

    def test_running_status_cannot_be_analyzed_as_complete(self):
        with patch.object(manage, "read", return_value={"status": "running", "source_manifest_sha256": "s"}):
            with self.assertRaises(ValueError):
                manage.validate_result(("drawer_open", "saber", 1), "s", {})

    def test_wrong_source_cannot_be_analyzed(self):
        with patch.object(manage, "read", return_value={"status": "completed", "source_manifest_sha256": "other"}):
            with self.assertRaises(ValueError):
                manage.validate_result(("drawer_open", "saber", 1), "expected", {})

    def test_incomplete_analysis_does_not_bootstrap(self):
        import analyze
        import gates
        catalog = {"environment": {}, "original_attempts": []}
        with patch.object(manage, "verify_all", return_value=("digest", catalog)), \
             patch.object(manage, "sha", return_value="recovery"), \
             patch.object(manage, "validate_result", side_effect=ValueError("missing")), \
             patch.object(gates, "write_atomic") as save, \
             patch.object(analyze, "paired_intervals") as inference:
            with self.assertRaises(RuntimeError):
                manage.analyze_tasks(["drawer_open"])
            payload = save.call_args.args[1]
            self.assertEqual(payload["status"], "incomplete")
            self.assertEqual(len(payload["incomplete"]), 10)
            self.assertNotIn("comparisons", payload)
            inference.assert_not_called()


if __name__ == "__main__":
    unittest.main()

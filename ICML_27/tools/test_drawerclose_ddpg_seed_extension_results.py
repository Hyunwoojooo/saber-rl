"""Publication checks; no training imports or live-output writes."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ddpg_seed_extension_results import measure
from drawerclose_ddpg_seed_extension_results import (
    export, require_completed_her, require_her_config, summarize,
)


class DrawerClosePublicationTests(unittest.TestCase):
    def queue(self):
        return {"protocol": {"task": "DrawerClose-v3", "backbone": "DDPG",
                             "seeds": list(range(6, 11)), "steps_per_run": 2000000},
                "arms": {"her": {"status": "completed", "seeds": {
                    str(s): {"status": "completed", "exit_code": 0,
                             "step": 2000000, "nonfinite_loss": []} for s in range(6, 11)}}}}

    def rows(self):
        return [{"arm": "her", "seed": s, "auc": .2, "final_quarter_auc": .3,
                 "final_success": 0., "hit90_step": None} for s in range(1, 11)]

    def config(self):
        return {"task_name": "drawer_close", "metaworld_env_name": "drawer-close-v3",
                "metaworld_reward_function_version": "v2", "reward_type": "sparse",
                "method": "b0_ddpg_her", "weight_scope": "none", "total_steps": 2000000,
                "updates_per_step": 1, "her_ratio": .8, "max_episode_steps": 200,
                "eval_freq": 10000, "eval_episodes": 20, "resume": False, "resume_path": None,
                "seeds": {"env_seed": 6, "torch_seed": 6, "numpy_seed": 6,
                          "action_noise_seed": 6, "weight_seed": 20006, "eval_seed": 10006}}

    def test_all_five_new_runs_required(self):
        require_completed_her(self.queue())
        for changes in ({"status": "running"}, {"exit_code": 1}, {"step": 1990000},
                        {"nonfinite_loss": ["loss/critic"]}):
            q = self.queue()
            q["arms"]["her"]["seeds"]["9"].update(changes)
            with self.assertRaises(ValueError):
                require_completed_her(q)

    def test_wrong_task_backbone_budget_rejected(self):
        for key, value in (("task", "DrawerOpen-v3"), ("backbone", "SAC"),
                           ("steps_per_run", 1000000), ("seeds", list(range(1, 6)))):
            q = self.queue()
            q["protocol"][key] = value
            with self.assertRaises(ValueError):
                require_completed_her(q)

    def test_missing_or_duplicate_runs_rejected(self):
        rows = self.rows()
        for invalid in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                summarize(invalid)

    def test_other_method_cannot_be_included(self):
        rows = self.rows()
        rows[5]["arm"] = "saber"
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_low_success_retained_and_no_paired_claim(self):
        report = summarize(self.rows())
        self.assertEqual(report["new_completed_runs"], 5)
        self.assertEqual(report["historical_reference_runs"], 5)
        self.assertEqual(report["paired_comparisons"], {})
        self.assertEqual(report["cohorts"]["pooled_1_10"]["n"], 10)
        self.assertEqual(report["cohorts"]["extension_6_10"]["final_success"]["mean"], 0.)

    def test_results_independent_of_input_order(self):
        self.assertEqual(summarize(self.rows()), summarize(self.rows()[::-1]))

    def test_nonfinite_or_invalid_metrics_rejected(self):
        for value in (float("nan"), float("inf"), -.1, 1.1):
            rows = self.rows()
            rows[0]["auc"] = value
            with self.assertRaises(ValueError):
                summarize(rows)

    def test_native_her_contract(self):
        require_her_config(self.config(), 6)
        for key, value in (("resume", True), ("resume_path", "checkpoint.pt"),
                           ("method", "b3_support_runtime_opt"), ("updates_per_step", 4),
                           ("weight_scope", "critic_only"), ("eval_freq", 50000)):
            cfg = self.config()
            cfg[key] = value
            with self.assertRaises(ValueError):
                require_her_config(cfg, 6)

    def test_seed_and_evaluation_seed_mapping(self):
        for key in self.config()["seeds"]:
            cfg = self.config()
            cfg["seeds"][key] += 1
            with self.assertRaises(ValueError):
                require_her_config(cfg, 6)

    def test_complete_evaluation_grid_required(self):
        curve = [(s, .5) for s in range(10000, 2000001, 10000)]
        self.assertEqual(measure(curve)["auc"], .5)
        for invalid in (curve[1:], curve[:-1], curve + [curve[-1]]):
            with self.assertRaises(ValueError):
                measure(invalid)

    def test_export_never_overwrites_destination_or_live_workspace(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for workspace, dest in ((root / "workspace", root), (root, root / "artifact")):
                with self.assertRaisesRegex(ValueError, "new artifact destination"):
                    export(workspace, dest)


if __name__ == "__main__":
    unittest.main()

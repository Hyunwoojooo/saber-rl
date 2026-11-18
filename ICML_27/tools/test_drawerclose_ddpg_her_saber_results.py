"""CPU-only publication regression checks; no training imports."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import ddpg_seed_extension_results as pair
from drawerclose_ddpg_her_saber_results import (
    export, require_completed, require_config, status_snapshot,
)


class DrawerClosePairPublicationTests(unittest.TestCase):
    def queue(self):
        return {"protocol": {"task": "DrawerClose-v3", "backbone": "DDPG",
                             "seeds": list(range(6, 11)), "steps_per_run": 2000000},
                "arms": {a: {"status": "completed", "seeds": {
                    str(s): {"status": "completed", "exit_code": 0, "step": 2000000,
                             "nonfinite_loss": [], "result": {"auc": .3}}
                    for s in range(6, 11)}} for a in ("her", "saber", "hgr")}}

    def config(self):
        return {"task_name": "drawer_close", "metaworld_env_name": "drawer-close-v3",
                "metaworld_reward_function_version": "v2", "reward_type": "sparse",
                "total_steps": 2000000, "updates_per_step": 1, "her_ratio": .8,
                "max_episode_steps": 200, "eval_freq": 10000, "eval_episodes": 20,
                "resume": False, "resume_path": None, "smoke": False,
                "method": "b3_support_runtime_opt", "weight_scope": "critic_only",
                "lambda_support": 1., "weighting_mode": "batch_norm", "action_bins_per_dim": 5,
                "N_local_min": 4, "M_Q_min": 4, "M_B": 16, "local_search_max": 512,
                "policy_support_mode": "bin", "support_scan_mode": "deterministic_tail_v1",
                "seeds": {"env_seed": 6, "numpy_seed": 6, "torch_seed": 6,
                          "action_noise_seed": 6, "eval_seed": 10006, "weight_seed": 20006}}

    def rows(self):
        return [{"arm": a, "seed": s, "auc": s / 20 + (0.1 if a == "saber" else 0),
                 "final_quarter_auc": .3, "final_success": 0., "hit90_step": None}
                for a in pair.ARMS for s in range(1, 11)]

    def test_all_ten_completed_extension_runs_required(self):
        require_completed(self.queue())
        for arm in pair.ARMS:
            for changes in ({"status": "running"}, {"exit_code": 1}, {"step": 1990000},
                            {"nonfinite_loss": ["loss/critic"]}):
                q = self.queue()
                q["arms"][arm]["seeds"]["9"].update(changes)
                with self.assertRaises(ValueError):
                    require_completed(q)
            q = self.queue()
            del q["arms"][arm]["seeds"]["10"]
            with self.assertRaises(ValueError):
                require_completed(q)

    def test_wrong_task_or_backbone_rejected(self):
        for key, value in (("task", "DrawerOpen-v3"), ("backbone", "SAC")):
            q = self.queue()
            q["protocol"][key] = value
            with self.assertRaises(ValueError):
                require_completed(q)

    def test_unpublished_methods_keep_only_status_even_if_complete(self):
        q = self.queue()
        original = deepcopy(q)
        clean = status_snapshot(q)
        self.assertEqual(q, original)
        self.assertEqual(clean["arms"]["saber"], q["arms"]["saber"])
        self.assertEqual(clean["arms"]["hgr"]["seeds"]["6"], {"status": "completed", "step": 2000000})
        self.assertEqual(clean, status_snapshot(clean))

    def test_saber_contract_and_fresh_execution(self):
        require_config(self.config(), "saber", 6)
        for key, value in (("weight_scope", "actor_critic"), ("method", "b0_ddpg_her"),
                           ("resume", True), ("resume_path", "checkpoint.pt"), ("smoke", True),
                           ("local_search_max", 8192), ("action_bins_per_dim", 7),
                           ("lambda_support", 0.), ("updates_per_step", 4)):
            config = self.config()
            config[key] = value
            with self.assertRaises(ValueError):
                require_config(config, "saber", 6)

    def test_seed_mapping(self):
        for key in self.config()["seeds"]:
            config = self.config()
            config["seeds"][key] += 1
            with self.assertRaises(ValueError):
                require_config(config, "saber", 6)

    def test_other_arm_and_seed_rejected(self):
        for arm, seed in (("hgr", 6), ("saber", 5), ("saber", 11)):
            with self.assertRaises(ValueError):
                require_config(self.config(), arm, seed)

    def test_seed_matching_independent_of_order(self):
        left = pair.summarize(self.rows())["cohorts"]
        right = pair.summarize(self.rows()[::-1])["cohorts"]
        self.assertEqual(left, right)
        self.assertAlmostEqual(left["pooled_1_10"]["paired"]["auc"]["mean_difference"], .1)

    def test_no_duplicate_missing_or_other_method(self):
        rows = self.rows()
        extra = {**rows[-1], "arm": "hgr"}
        for invalid in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]], rows[:-1] + [extra]):
            with self.assertRaises(ValueError):
                pair.summarize(invalid)

    def test_zero_success_retained(self):
        report = pair.summarize(self.rows())
        self.assertEqual(len(report["per_seed"]), 20)
        self.assertEqual(report["cohorts"]["pooled_1_10"]["summary"]["saber"]["final_success"]["mean"], 0.)

    def test_export_cannot_overwrite_or_write_live_workspace(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for workspace, dest in ((root / "workspace", root), (root, root / "artifact")):
                with self.assertRaisesRegex(ValueError, "new artifact destination"):
                    export(workspace, dest)


if __name__ == "__main__":
    unittest.main()

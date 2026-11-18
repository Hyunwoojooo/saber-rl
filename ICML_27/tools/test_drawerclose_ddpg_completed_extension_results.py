"""Verify primary-cohort publication boundaries without importing trainers."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from drawerclose_ddpg_completed_extension_results import (
    ARMS, ARTIFACT, PUBLISHED_ARMS, completed_keys, export, require_config,
    status_snapshot, summarize,
)


class DrawerClosePrimaryPublicationTests(unittest.TestCase):
    def fixture(self):
        queue = {"updated_utc": "2026-11-18T00:00:00+00:00",
                 "protocol": {"order": list(ARMS), "seeds": list(range(6, 11)),
                              "task": "DrawerClose-v3", "backbone": "DDPG", "steps_per_run": 2000000},
                 "arms": {a: {"status": "completed", "seeds": {
                     str(s): {"status": "completed", "step": 2000000, "exit_code": 0,
                              "nonfinite_loss": [], "result": {"auc": .4}}
                     for s in range(6, 11)}} for a in ARMS}}
        rows = [{"arm": a, "seed": s, "auc": .2 + s / 30,
                 "final_quarter_auc": .3, "final_success": 0., "hit90_step": None}
                for a in PUBLISHED_ARMS for s in range(1, 11)]
        return queue, rows

    def config(self, arm):
        config = {"task_name": "drawer_close", "metaworld_env_name": "drawer-close-v3",
                  "metaworld_reward_function_version": "v2", "reward_type": "sparse",
                  "total_steps": 2000000, "updates_per_step": 1, "her_ratio": .8,
                  "max_episode_steps": 200, "eval_freq": 10000, "eval_episodes": 20,
                  "resume": False, "resume_path": None, "smoke": False,
                  "seeds": {"env_seed": 6, "numpy_seed": 6, "torch_seed": 6,
                            "action_noise_seed": 6, "eval_seed": 10006, "weight_seed": 20006}}
        if arm == "hgr":
            config.update(method="hgr", weight_scope="none", hgr_alpha_episode=.6, hgr_alpha_goal=.6,
                          hgr_beta_episode_initial=.4, hgr_beta_episode_final=1.,
                          hgr_beta_goal_initial=.4, hgr_beta_goal_final=1., hgr_priority_eps=1e-6,
                          hgr_storage={"enabled": True, "actor_loss_weight": "all_ones"})
        else:
            config.update(method="ddpg_her_per_full", weight_scope="critic_only", lambda_support=0.,
                          per_alpha=.6, per_beta_start=.4, per_beta_end=1., per_beta_steps=2000000)
        return config

    def test_destination_is_drawerclose(self):
        self.assertEqual(ARTIFACT.parents[1].name, "metaworld_Drawer_Close_DDPG")

    def test_exactly_twenty_extensions_even_if_other_methods_complete(self):
        q, rows = self.fixture()
        self.assertEqual(completed_keys(q), {(a, s) for a in PUBLISHED_ARMS for s in range(6, 11)})
        report = summarize(rows, q)
        self.assertEqual(report["new_completed_runs"], 20)
        self.assertEqual(report["historical_reference_runs"], 20)
        self.assertEqual(report["newly_published_runs"], 10)
        self.assertEqual(report["complete_arms"], list(PUBLISHED_ARMS))
        self.assertEqual(report["partial_arms"], [])
        for c in report["cohorts"].values():
            self.assertEqual(set(c["paired"]), {"saber_minus_her", "saber_minus_hgr", "saber_minus_per"})

    def test_all_primary_seeds_must_complete(self):
        for arm in PUBLISHED_ARMS:
            for change in ({"status": "running"}, {"step": 1000000}, {"exit_code": 1},
                           {"nonfinite_loss": ["loss/critic"]}):
                q, _ = self.fixture()
                q["arms"][arm]["seeds"]["7"].update(change)
                with self.assertRaises(ValueError):
                    completed_keys(q)

    def test_nonprimary_results_scrubbed_without_changing_status(self):
        q, _ = self.fixture()
        original = deepcopy(q)
        clean = status_snapshot(q)
        self.assertEqual(q, original)
        self.assertEqual(clean, status_snapshot(clean))
        self.assertEqual(clean["arms"]["hgr"], q["arms"]["hgr"])
        self.assertEqual(clean["arms"]["replay_query"]["seeds"]["6"], {"status": "completed", "step": 2000000})
        self.assertEqual(completed_keys(clean), completed_keys(q))

    def test_missing_duplicate_or_control_rows_rejected(self):
        q, rows = self.fixture()
        for invalid in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]],
                        rows[:-1] + [{**rows[-1], "arm": "replay_query"}]):
            with self.assertRaises(ValueError):
                summarize(invalid, q)

    def test_seed_order_and_zero_success(self):
        q, rows = self.fixture()
        report = summarize(rows, q)
        self.assertEqual(report, summarize(rows[::-1], q))
        self.assertEqual(report["cohorts"]["pooled_1_10"]["summary"]["per"]["final_success"]["mean"], 0.)
        self.assertEqual(report["cohorts"]["pooled_1_10"]["paired"]["saber_minus_per"]["auc"]["ci95"], [0., 0.])

    def test_nonfinite_and_out_of_range_metrics(self):
        for value in (float("nan"), float("inf"), -.1, 1.1):
            q, rows = self.fixture()
            rows[0]["auc"] = value
            with self.assertRaises(ValueError):
                summarize(rows, q)

    def test_wrong_task_backbone_or_budget(self):
        for key, value in (("task", "DrawerOpen-v3"), ("backbone", "TD3"), ("steps_per_run", 1000000)):
            q, _ = self.fixture()
            q["protocol"][key] = value
            with self.assertRaises(ValueError):
                completed_keys(q)

    def test_native_baseline_contracts(self):
        for arm in ("hgr", "per"):
            require_config(self.config(arm), arm, 6)
            for key, value in (("method", "per_style_loss_weighting"), ("resume", True),
                               ("updates_per_step", 4), ("task_name", "drawer_open")):
                config = self.config(arm)
                config[key] = value
                with self.assertRaises(ValueError):
                    require_config(config, arm, 6)

    def test_eval_seed_mapping(self):
        for arm in ("hgr", "per"):
            config = self.config(arm)
            config["seeds"]["eval_seed"] += 1
            with self.assertRaises(ValueError):
                require_config(config, arm, 6)

    def test_no_overwrite_or_live_workspace_export(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for workspace, dest in ((root / "workspace", root), (root, root / "artifact")):
                with self.assertRaisesRegex(ValueError, "new artifact destination"):
                    export(workspace, dest)


if __name__ == "__main__":
    unittest.main()

"""Verify DrawerClose mechanism-control publication boundaries without importing trainers."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from drawerclose_ddpg_replay_query_results import (
    ARMS, ARTIFACT, PUBLISHED_ARMS, RANDOM_SUPPORT, REPLAY_QUERY, SHUFFLED_QUERY,
    completed_keys, export, require_config,
    status_snapshot, summarize,
)


class DrawerCloseReplayPublicationTests(unittest.TestCase):
    def fixture(self, control=REPLAY_QUERY):
        queue = {"updated_utc": "2026-11-19T00:00:00+00:00",
                 "protocol": {"order": list(ARMS), "seeds": list(range(6, 11)),
                              "task": "DrawerClose-v3", "backbone": "DDPG", "steps_per_run": 2000000},
                 "arms": {a: {"status": "completed", "seeds": {
                     str(s): {"status": "completed", "step": 2000000, "exit_code": 0,
                              "nonfinite_loss": [], "result": {"auc": .4}}
                     for s in range(6, 11)}} for a in ARMS}}
        rows = [{"arm": a, "seed": s, "auc": .2 + s / 30,
                 "final_quarter_auc": .3, "final_success": 0., "hit90_step": None}
                for a in control.arms for s in range(1, 11)]
        return queue, rows

    def config(self, arm, control=REPLAY_QUERY):
        config = {"task_name": "drawer_close", "metaworld_env_name": "drawer-close-v3",
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
        if arm == control.arm:
            config["method"] = control.method
            if control.mode is not None:
                config["support_query_mode"] = control.mode
            if control == RANDOM_SUPPORT:
                config["random_fixed_count_k"] = 0
        return config

    def test_destination_is_drawerclose(self):
        self.assertEqual(ARTIFACT.parents[1].name, "metaworld_Drawer_Close_DDPG")

    def test_exact_two_method_scope_even_if_other_methods_complete(self):
        q, rows = self.fixture()
        self.assertEqual(completed_keys(q), {(a, s) for a in PUBLISHED_ARMS for s in range(6, 11)})
        report = summarize(rows, q)
        self.assertEqual(report["new_completed_runs"], 10)
        self.assertEqual(report["historical_reference_runs"], 10)
        self.assertEqual(report["newly_published_runs"], 5)
        self.assertEqual(report["reused_previously_published_extension_runs"], 5)
        self.assertEqual(report["complete_arms"], list(PUBLISHED_ARMS))
        self.assertEqual(report["partial_arms"], [])
        for c in report["cohorts"].values():
            self.assertEqual(set(c["paired"]), {"saber_minus_replay_query"})

    def test_incomplete_or_failed_seed_rejected(self):
        for arm in PUBLISHED_ARMS:
            for change in ({"status": "running"}, {"step": 1000000}, {"exit_code": 1},
                           {"nonfinite_loss": ["loss/critic"]}):
                q, _ = self.fixture()
                q["arms"][arm]["seeds"]["7"].update(change)
                with self.assertRaises(ValueError):
                    completed_keys(q)

    def test_other_performance_scrubbed_without_changing_status(self):
        q, _ = self.fixture()
        original = deepcopy(q)
        clean = status_snapshot(q)
        self.assertEqual(q, original)
        self.assertEqual(clean, status_snapshot(clean))
        self.assertEqual(clean["arms"]["replay_query"], q["arms"]["replay_query"])
        for arm in set(ARMS) - set(PUBLISHED_ARMS):
            self.assertEqual(clean["arms"][arm]["seeds"]["6"], {"status": "completed", "step": 2000000})
        self.assertEqual(completed_keys(clean), completed_keys(q))

    def test_missing_duplicate_or_unrelated_rows_rejected(self):
        q, rows = self.fixture()
        for invalid in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]],
                        rows[:-1] + [{**rows[-1], "arm": "shuffled_query"}]):
            with self.assertRaises(ValueError):
                summarize(invalid, q)

    def test_seed_order_and_zero_success(self):
        q, rows = self.fixture()
        report = summarize(rows, q)
        self.assertEqual(report, summarize(rows[::-1], q))
        c = report["cohorts"]["pooled_1_10"]
        self.assertEqual(c["summary"]["replay_query"]["final_success"]["mean"], 0.)
        self.assertEqual(c["paired"]["saber_minus_replay_query"]["auc"]["ci95"], [0., 0.])

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

    def test_query_modes_and_shared_contract(self):
        for arm in PUBLISHED_ARMS:
            require_config(self.config(arm), arm, 6)
            for key, value in (("resume", True), ("updates_per_step", 4), ("task_name", "drawer_open"),
                               ("weight_scope", "actor_critic"), ("lambda_support", 0.),
                               ("support_query_mode", "shuffled_policy")):
                config = self.config(arm)
                config[key] = value
                with self.assertRaises(ValueError):
                    require_config(config, arm, 6)

    def test_query_mode_cannot_be_swapped_or_omitted(self):
        for arm, mode in (("saber", "replay_action"), ("replay_query", "current_policy")):
            config = self.config(arm)
            config["support_query_mode"] = mode
            with self.assertRaises(ValueError):
                require_config(config, arm, 6)
        config = self.config("replay_query")
        del config["support_query_mode"]
        with self.assertRaises(ValueError):
            require_config(config, "replay_query", 6)

    def test_eval_seed_mapping(self):
        for arm in PUBLISHED_ARMS:
            config = self.config(arm)
            config["seeds"]["eval_seed"] += 1
            with self.assertRaises(ValueError):
                require_config(config, arm, 6)

    def test_shuffled_scope_excludes_replay_and_incomplete_control(self):
        q, rows = self.fixture(SHUFFLED_QUERY)
        q["arms"]["random_support"].update(status="running")
        q["arms"]["random_support"]["seeds"]["6"].update(status="running", step=1000000)
        clean = status_snapshot(q, SHUFFLED_QUERY)
        self.assertEqual(clean["arms"]["shuffled_query"], q["arms"]["shuffled_query"])
        self.assertEqual(clean["arms"]["replay_query"]["seeds"]["6"],
                         {"status": "completed", "step": 2000000})
        self.assertEqual(clean["arms"]["random_support"]["seeds"]["6"],
                         {"status": "running", "step": 1000000})
        report = summarize(rows, clean, SHUFFLED_QUERY)
        self.assertEqual(report["complete_arms"], ["saber", "shuffled_query"])
        self.assertEqual(report["newly_published_runs"], 5)
        for cohort in report["cohorts"].values():
            self.assertEqual(set(cohort["paired"]), {"saber_minus_shuffled_query"})
        with self.assertRaises(ValueError):
            summarize(rows + [{**rows[0], "arm": "replay_query"}], clean, SHUFFLED_QUERY)

    def test_shuffled_requires_all_five_completed_seeds(self):
        q, _ = self.fixture(SHUFFLED_QUERY)
        q["arms"]["shuffled_query"]["seeds"]["6"]["step"] = 1900000
        with self.assertRaises(ValueError):
            completed_keys(q, SHUFFLED_QUERY)

    def test_shuffled_query_mode_is_not_exchangeable_with_other_modes(self):
        for arm in SHUFFLED_QUERY.arms:
            require_config(self.config(arm, SHUFFLED_QUERY), arm, 6, SHUFFLED_QUERY)
        config = self.config("shuffled_query", SHUFFLED_QUERY)
        for mode in ("replay_action", "current_policy"):
            config["support_query_mode"] = mode
            with self.assertRaises(ValueError):
                require_config(config, "shuffled_query", 6, SHUFFLED_QUERY)

    def test_random_support_scope_excludes_count_matched_control(self):
        q, rows = self.fixture(RANDOM_SUPPORT)
        q["arms"]["strict_random"].update(status="running")
        q["arms"]["strict_random"]["seeds"]["6"].update(status="running", step=1500000)
        clean = status_snapshot(q, RANDOM_SUPPORT)
        self.assertEqual(clean["arms"]["strict_random"]["seeds"]["6"],
                         {"status": "running", "step": 1500000})
        self.assertEqual(clean["arms"]["random_support"], q["arms"]["random_support"])
        report = summarize(rows, clean, RANDOM_SUPPORT)
        self.assertEqual(report["complete_arms"], ["saber", "random_support"])
        self.assertEqual(report["newly_published_runs"], 5)
        for cohort in report["cohorts"].values():
            self.assertEqual(set(cohort["paired"]), {"saber_minus_random_support"})
        with self.assertRaises(ValueError):
            summarize(rows + [{**rows[0], "arm": "strict_random"}], clean, RANDOM_SUPPORT)

    def test_random_support_requires_completed_seeds(self):
        q, _ = self.fixture(RANDOM_SUPPORT)
        q["arms"]["random_support"]["seeds"]["9"]["exit_code"] = 1
        with self.assertRaises(ValueError):
            completed_keys(q, RANDOM_SUPPORT)

    def test_random_support_requires_own_matched_count_preset(self):
        for arm in RANDOM_SUPPORT.arms:
            require_config(self.config(arm, RANDOM_SUPPORT), arm, 6, RANDOM_SUPPORT)
        correct = self.config("random_support", RANDOM_SUPPORT)
        for change in ({"method": "b3_support_runtime_opt"},
                       {"method": "b3_fully_random_fixed_count"},
                       {"random_fixed_count_k": 32},
                       {"support_query_mode": "shuffled_policy"}):
            with self.assertRaises(ValueError):
                require_config({**correct, **change}, "random_support", 6, RANDOM_SUPPORT)

    def test_no_overwrite_or_live_workspace_export(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for workspace, dest in ((root / "workspace", root), (root, root / "artifact")):
                with self.assertRaisesRegex(ValueError, "new artifact destination"):
                    export(workspace, dest)


if __name__ == "__main__":
    unittest.main()

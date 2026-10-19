import copy
import contextlib
import csv
import io
import json
import importlib
import importlib.util
import itertools
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import ROOT, make_config, evaluation_seed, scientific_config
from metacog_lite_v0.agent import DDPGAgent
from metacog_lite_v0.assignment_comparison import (
    AssignmentSelector, AssignmentWindows, random_mask, td_mask,
    validate_masks, mean_one_weights, overlap_statistics,
)
from analyze import joint_bootstrap, evaluation_curve, curve_metrics, validate_assignment_trace


def load_upstream():
    name = "assignment_upstream"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "upstream/code/src/metacog_lite_v0/__init__.py",
            submodule_search_locations=[str(ROOT / "upstream/code/src/metacog_lite_v0")])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(name + ".agent").DDPGAgent


def agent(cls=DDPGAgent):
    return cls(state_dim=5, action_dim=2, hidden_dim=16, actor_lr=3e-4,
               critic_lr=3e-4, gamma=0.98, tau=0.005,
               action_low=np.full(2, -1, np.float32), action_high=np.full(2, 1, np.float32),
               device=torch.device("cpu"), l2_action_penalty=0)


def batch():
    g = torch.Generator().manual_seed(713)
    return dict(state=torch.randn(8, 5, generator=g),
                action_exec=torch.randn(8, 2, generator=g).clamp(-1, 1),
                reward=-torch.ones(8), next_state=torch.randn(8, 5, generator=g),
                nonterminal=torch.tensor([1, 1, 0, 1, 1, 1, 0, 1.0]),
                omega_critic=torch.ones(8), omega_actor=torch.ones(8))


class AssignmentTests(unittest.TestCase):
    def test_exact_k_and_same_weight_distribution(self):
        rng = np.random.default_rng(3)
        for n in (1, 8, 256):
            for m in range(n + 1):
                e = np.arange(n) < m
                for k in {0, m, m // 2}:
                    s = np.arange(n) < k
                    r = random_mask(e, k, rng)
                    t = td_mask(e, k, rng.random(n), rng)
                    for mask in (r, t):
                        self.assertEqual(int(mask.sum()), k)
                        self.assertFalse(np.any(mask & ~e))
                        np.testing.assert_array_equal(np.sort(mean_one_weights(mask)[1]),
                                                      np.sort(mean_one_weights(s)[1]))
                        self.assertAlmostEqual(float(mean_one_weights(mask)[1].mean()), 1, places=6)
                    if k in (0, m):
                        np.testing.assert_array_equal(s, r)
                        np.testing.assert_array_equal(s, t)

    def test_random_selection_parity_with_frozen_strict(self):
        load_upstream()
        old = importlib.import_module("assignment_upstream.weights")
        old_rng, new_rng = np.random.default_rng(30001), np.random.default_rng(30001)
        data = np.random.default_rng(27)
        for _ in range(500):
            e = data.random(40) < 0.7
            s = e & (data.random(40) < 0.4)
            expected = old.context_eligible_count_matched_assignment(e, s, old_rng)
            np.testing.assert_array_equal(expected, random_mask(e, int(s.sum()), new_rng))
        self.assertEqual(old_rng.bit_generator.state, new_rng.bit_generator.state)

    def test_random_uniformity_and_expected_overlap(self):
        rng = np.random.default_rng(42)
        e, s = np.ones(4, bool), np.array([1, 1, 0, 0], bool)
        counts = {ids: 0 for ids in itertools.combinations(range(4), 2)}
        for _ in range(12000):
            counts[tuple(np.flatnonzero(random_mask(e, 2, rng)))] += 1
        self.assertTrue(all(abs(n - 2000) < 180 for n in counts.values()))
        overlaps = [len(set(ids) & {0, 1}) for ids in counts]
        self.assertEqual(np.mean(overlaps), 1.0)
        stats = overlap_statistics(e, s, s, s, s)
        self.assertEqual(stats["random_expected_intersection"], 1)
        self.assertEqual(stats["random_expected_overlap_fraction"], .5)
        self.assertNotAlmostEqual(np.mean([x / (4 - x) for x in overlaps]), 1 / 3)

    def test_td_topk_ties_and_nonfinite(self):
        e = np.array([1, 1, 0, 1, 1], bool)
        scores = np.array([8, 5, 100, 5, 2.0])
        rng = np.random.default_rng(10)
        counts = np.zeros(5)
        for _ in range(4000):
            t = td_mask(e, 2, scores, rng)
            self.assertTrue(t[0])
            self.assertFalse(t[2] or t[4])
            counts += t
        self.assertLess(abs(counts[1] - 2000), 130)
        for value in (np.nan, np.inf, -np.inf):
            bad = scores.copy()
            bad[2] = value
            with self.assertRaises(FloatingPointError):
                td_mask(e, 2, bad, rng)

    def test_rng_and_logging_do_not_change_training_streams(self):
        replay = np.random.default_rng(18)
        before = copy.deepcopy(replay.bit_generator.state)
        global_before = copy.deepcopy(np.random.get_state())
        torch_before = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            windows = AssignmentWindows(Path(tmp) / "trace.csv", 3)
            for log in (False, True):
                random_rng, tie_rng = np.random.default_rng(30), np.random.default_rng(40)
                masks = []
                for step in range(1, 9):
                    wb = SimpleNamespace(context_eligible_mask=np.ones(8, bool),
                                         support_mask=np.array([1, 1, 0, 0, 0, 0, 0, 0], bool))
                    select = AssignmentSelector(wb, "random", random_rng, tie_rng)
                    masks.append(select(np.ones(8)))
                    if log:
                        windows.add(step, step, select.stats)
                if not log:
                    reference = masks
                else:
                    for a, b in zip(reference, masks):
                        np.testing.assert_array_equal(a, b)
            windows.close()
        self.assertEqual(before, replay.bit_generator.state)
        np.testing.assert_array_equal(global_before[1], np.random.get_state()[1])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))

    def test_na_and_informative_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            log = AssignmentWindows(path, 2)
            e = np.ones(4, bool)
            for step, k in enumerate((0, 2, 4), 1):
                m = np.arange(4) < k
                stats = overlap_statistics(e, m, m, m, m)
                if not k:
                    self.assertIsNone(stats["td_overlap_fraction"])
                    self.assertIsNone(stats["td_jaccard"])
                log.add(step, step * 10, stats)
            log.close()
            result = validate_assignment_trace(path, 3)
            self.assertEqual(result["all"]["td_overlap_fraction"]["valid_updates"], 2)
            self.assertEqual(result["informative"]["td_overlap_fraction"]["valid_updates"], 1)

    def test_invalid_eligibility(self):
        with self.assertRaises(ValueError):
            validate_masks([0, 1], [1, 0])


class UpdateTests(unittest.TestCase):
    def assert_tree_equal(self, a, b):
        if isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for k in a:
                self.assert_tree_equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        elif torch.is_tensor(a):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            self.assertEqual(a, b)

    def test_update_parity_including_optimizer_state_and_targets(self):
        old, new = agent(load_upstream()), agent()
        new.load_state_dict(copy.deepcopy(old.state_dict()))
        data = batch()
        for step in range(3):
            mask = np.arange(8) % (step + 2) == 0
            weight = mean_one_weights(mask)[1]
            old_data = {**data, "omega_critic": torch.tensor(weight)}
            expected = old.update(**old_data)
            observed = new.update(**data, critic_weight_selector=lambda td: weight,
                                  fail_on_nonfinite=True)
            self.assertEqual(expected.critic_loss, observed.critic_loss)
            self.assertEqual(expected.actor_loss, observed.actor_loss)
            self.assert_tree_equal(old.state_dict(), new.state_dict())

    def test_td_ranking_reuses_single_preupdate_target_and_replay_action(self):
        a, data = agent(), batch()
        with torch.no_grad():
            target = data["reward"] + a.gamma * data["nonterminal"] * a.critic_target(
                data["next_state"], a.actor_target(data["next_state"]))
            td = a.critic(data["state"], data["action_exec"]) - target
        wb = SimpleNamespace(context_eligible_mask=np.ones(8, bool),
                             support_mask=np.arange(8) < 3)
        selector = AssignmentSelector(wb, "td_topk", np.random.default_rng(1), np.random.default_rng(2))
        calls, actions = [], []
        h1 = a.critic_target.register_forward_hook(lambda *args: calls.append(1))
        h2 = a.critic.register_forward_pre_hook(lambda module, args: actions.append(args[1].clone()))
        result = a.update(**data, critic_weight_selector=selector, fail_on_nonfinite=True)
        h1.remove()
        h2.remove()
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(actions[0], data["action_exec"], rtol=0, atol=0)
        np.testing.assert_array_equal(result.td_error_abs_values, td.abs().numpy())
        self.assertEqual(result.critic_loss, float((torch.tensor(wb.omega_critic) * td.square()).mean()))
        self.assertTrue(np.all(wb.omega_actor == 1))

    def test_nonfinite_is_a_failure_not_zero(self):
        a, data = agent(), batch()
        data["reward"][0] = float("nan")
        before = copy.deepcopy(a.critic.state_dict())
        with self.assertRaises(FloatingPointError):
            a.update(**data, fail_on_nonfinite=True)
        self.assert_tree_equal(before, a.critic.state_dict())


class AnalysisTests(unittest.TestCase):
    def test_attempt_discovery_ignores_sibling_logs(self):
        from analyze import list_attempts, accepted_attempt
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "attempt_01").mkdir()
            (root / "attempt_01/status.json").write_text(json.dumps({"status": "complete"}))
            (root / "attempt_01.log").write_text("completed run log")
            self.assertEqual(list_attempts(root), [root / "attempt_01"])
            self.assertEqual(accepted_attempt(root), root / "attempt_01")

    def test_full_analysis_pipeline_and_failed_seed_reporting(self):
        import analyze
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            production = root / "production"
            production.mkdir()
            (production / "plan.json").write_text(json.dumps({"source_manifest_sha256": "fixture"}))
            for seed in range(1, 11):
                for j, arm in enumerate(("support", "random", "td_topk")):
                    attempt = production / arm / f"seed_{seed:02d}" / "attempt_01"
                    run = attempt / "run"
                    run.mkdir(parents=True)
                    cfg = make_config(arm, seed, Path("/original/remote/location")).as_dict()
                    (run / "config.json").write_text(json.dumps(cfg))
                    state = {"arm": arm, "seed": seed, "profile": "production",
                             "status": "complete", "checkpoint_continuation": False,
                             "source_manifest_sha256": "fixture"}
                    (attempt / "status.json").write_text(json.dumps(state))
                    env = {k: {} for k in ("python", "packages", "torch_cuda_build", "cudnn", "gpu", "threads")}
                    (attempt / "environment.json").write_text(json.dumps(env))
                    y = .5 + seed * .001 - j * .05
                    (run / "metrics.csv").write_text("step,eval/env_success_mean\n" +
                                                     "".join(f"{s},{y}\n" for s in range(10000, 2000001, 10000)))
                    summary = {"total_steps": 2000000, "total_updates": 1995001,
                               "final_eval_env_success": y, "runtime_total_sec": 10000}
                    (run / "summary.json").write_text(json.dumps(summary))
                    (run / "checkpoint.pt").touch()
                    fields = ["start_update", "end_update", "step", "subset", "n_updates",
                              "informative_update_mean", "informative_update_n", "td_overlap_fraction_mean", "td_overlap_fraction_n"]
                    with (run / "assignment_windows.csv").open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fields)
                        writer.writeheader()
                        for start in range(1, 1995002, 1000):
                            end = min(start + 999, 1995001)
                            for group in ("all", "informative"):
                                n = end - start + 1 if group == "all" else 0
                                writer.writerow(dict(start_update=start, end_update=end, step=end + 4999,
                                                     subset=group, n_updates=n,
                                                     informative_update_mean=0 if n else None,
                                                     informative_update_n=n, td_overlap_fraction_mean=None,
                                                     td_overlap_fraction_n=0))
            with patch.object(sys, "argv", ["analyze.py", "--output-root", str(root)]), contextlib.redirect_stdout(io.StringIO()):
                analyze.main()
                report = json.loads((root / "analysis/report.json").read_text())
                self.assertAlmostEqual(report["inference"]["support_minus_td_topk"]["mean"], .1)
                target = production / "td_topk/seed_04/attempt_01/status.json"
                status = json.loads(target.read_text())
                status["status"] = "failed_learning"
                target.write_text(json.dumps(status))
                with self.assertRaises(SystemExit):
                    analyze.main()
                report = json.loads((root / "analysis/report.json").read_text())
                self.assertIsNone(report["inference"])
                self.assertEqual(len(report["runs"]), 30)
                self.assertEqual(report["failures_or_missing"][0]["seed"], 4)
                status["status"], status["seed"] = "complete", 5
                target.write_text(json.dumps(status))
                with self.assertRaises(SystemExit):
                    analyze.main()
                report = json.loads((root / "analysis/report.json").read_text())
                self.assertIn("identity", report["failures_or_missing"][0]["problem"])

    def test_completed_or_learning_failed_seed_cannot_be_replaced(self):
        from analyze import accepted_attempt
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for number in (1, 2):
                p = root / f"attempt_{number:02d}"
                p.mkdir()
                (p / "status.json").write_text(json.dumps({"status": "complete", "retry_reason": "test"}))
            with self.assertRaises(ValueError):
                accepted_attempt(root)
            p = root / "attempt_01/status.json"
            p.write_text(json.dumps({"status": "failed_learning"}))
            with self.assertRaises(ValueError):
                accepted_attempt(root)
            p.write_text(json.dumps({"status": "failed_infrastructure"}))
            self.assertEqual(accepted_attempt(root).name, "attempt_02")

    def test_joint_bootstrap_quantiles(self):
        v = np.random.default_rng(80).random((10, 3))
        result = joint_bootstrap(v, 2000, 99)
        rows = np.random.default_rng(99).integers(0, 10, (2000, 10))
        diffs = v[:, [0]] - v[:, 1:]
        draws = diffs[rows].mean(axis=1)
        for j, key in enumerate(result):
            np.testing.assert_array_equal(result[key]["ci95"], np.quantile(draws[:, j], [.025, .975]))
            np.testing.assert_array_equal(result[key]["ci97_5_bonferroni"], np.quantile(draws[:, j], [.0125, .9875]))
        with self.assertRaises(ValueError):
            joint_bootstrap(v[:-1])

    def test_auc_denominator_and_missing_checkpoint(self):
        grid = list(range(10000, 2000001, 10000))
        metrics = curve_metrics(grid, np.ones(200), 1500000)
        self.assertEqual(metrics["auc"], 1)
        self.assertEqual(metrics["fq_auc"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "curve.csv"
            p.write_text("step,eval/env_success_mean\n" + "".join(f"{s},1\n" for s in grid[1:]))
            with self.assertRaises(ValueError):
                evaluation_curve(p, grid)
            p.write_text("step,eval/env_success_mean\n" + "".join(f"{s},1\n" for s in grid + [grid[-1]]))
            with self.assertRaises(ValueError):
                evaluation_curve(p, grid)

    def test_config_parity_and_evaluation_seeds(self):
        configs = [make_config(a, 1, Path("/tmp/test-assignment")) for a in ("support", "random", "td_topk")]
        for cfg in configs:
            self.assertEqual(scientific_config(cfg.as_dict()), scientific_config(configs[0].as_dict()))
            self.assertEqual(cfg.eval_freq, 10000)
            self.assertEqual(cfg.total_steps, 2000000)
            self.assertEqual(cfg.seeds.eval_seed + 10000 * 10 + 19, evaluation_seed(1, 10000, 19))
            cfg.resume = True
            with self.assertRaises(ValueError):
                cfg.normalize()
            cfg.resume = False

    def test_frozen_her_and_replay_are_byte_identical(self):
        for filename in ("replay.py", "task_drawer_open.py", "envs.py", "networks.py",
                         "numba_support_scan_drawer_open.py"):
            self.assertEqual((ROOT / "code/src/metacog_lite_v0" / filename).read_bytes(),
                             (ROOT / "upstream/code/src/metacog_lite_v0" / filename).read_bytes())


class RetrievalTests(unittest.TestCase):
    def test_eligibility_and_support_match_frozen_strict_after_wrap(self):
        from metacog_lite_v0.replay import HERReplayBuffer
        from metacog_lite_v0.task_drawer_open import MetaWorldDrawerOpenTaskSpec
        from metacog_lite_v0.weights import LiteWeightComputer, GapStats
        load_upstream()
        old_weights = importlib.import_module("assignment_upstream.weights")
        task = MetaWorldDrawerOpenTaskSpec(np.full(4, -1, np.float32), np.ones(4, np.float32), epsilon_g=.05)
        replay = HERReplayBuffer(1024, 7, 3, 4)
        for i in range(1400):
            obs = np.array([0, 0, 0, .2, .25, .1, 0], np.float32)
            if i % 3 == 0:
                obs[4] = .8
            action = np.full(4, 0 if i % 3 == 0 else .9, np.float32)
            goal = np.array([.6, .1, 0], np.float32)
            replay.add_transition(obs, obs[4:7], goal, action, -1, obs, obs[4:7], False,
                                  i % 200 == 199, 20, i // 200, i % 200,
                                  task.coarse_hash(obs, 20), int(task.action_bin_batch(action[None], 5)[0]))
        b = replay.sample(256, .8, np.random.default_rng(18))
        actor = SimpleNamespace(policy_actions=lambda state: torch.zeros((len(state), 4)))
        new_cfg = make_config("support", 1, Path("/tmp/test-assignment"), device="cpu")
        old_cfg = copy.deepcopy(new_cfg)
        old_cfg.assignment_experiment = False
        old_cfg.support_assignment_mode = "context_eligible_count_matched"
        old = old_weights.LiteWeightComputer(old_cfg, task, old_weights.GapStats(50000, 128, .1, .9, 1e-6))
        new = LiteWeightComputer(new_cfg, task, GapStats(50000, 128, .1, .9, 1e-6))
        before = copy.deepcopy(replay.state_dict())
        old_wb = old.compute(b, replay, actor, np.random.default_rng(20001), torch.device("cpu"),
                             assignment_rng=np.random.default_rng(30001))
        replay.load_state_dict(before)
        wb = new.compute(b, replay, actor, np.random.default_rng(20001), torch.device("cpu"),
                         assignment_rng=np.random.default_rng(30001))
        np.testing.assert_array_equal(wb.support_mask, old_wb.psi_q_rank > .5)
        self.assertEqual(int(wb.context_eligible_mask.sum()), old_wb.extra_metrics["strict/context_eligible_count"])
        self.assertGreater(int(wb.context_eligible_mask.sum()), 0)
        self.assertGreater(int((wb.context_eligible_mask & ~wb.support_mask).sum()), 0)
        selector = AssignmentSelector(wb, "random", np.random.default_rng(30001), np.random.default_rng(40001))
        selected = selector(np.arange(256, dtype=np.float32))
        np.testing.assert_array_equal(selected, old_wb.omega_critic)


def load_tests(loader, tests, pattern):
    # Retained checks run from the independent copy, including original scanner parity.
    for name in ("test_strict_context_eligible_count_matched", "test_strict_ddpg_loss_scope",
                 "test_strict_drawer_open_scanner", "test_strict_config_resume_contract"):
        module = importlib.import_module(name)
        for key in sorted(vars(module)):
            if key.startswith("test_") and callable(getattr(module, key)):
                tests.addTest(unittest.FunctionTestCase(getattr(module, key)))
    return tests


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

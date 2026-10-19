"""CPU-only tests. Author loss is extracted from the untouched vendored AST."""
import ast
import copy
import importlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import ROOT, make_config, expected_reference_config, PAL_ONLY_FIELDS
from metacog_lite_v0.agent import DDPGAgent
from metacog_lite_v0.pal import prioritized_approximation_loss, PALLoss
from metacog_lite_v0.replay import HERReplayBuffer


def official_loss(delta, alpha=0.4, threshold=1.0):
    tree = ast.parse((ROOT / "third_party/lap_pal/PAL_TD3.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PAL_TD3")
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "PAL")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, "official_PAL_method", "exec"), namespace)
    base = namespace["PAL"](SimpleNamespace(alpha=alpha, min_priority=threshold), delta)
    normalizer = torch.max(delta.abs(), delta.abs()).clamp(min=threshold).pow(alpha).mean().detach()
    return base / normalizer, normalizer, base


def upstream(module):
    name = "pal_frozen_upstream"
    if name not in sys.modules:
        directory = ROOT / "upstream/code/src/metacog_lite_v0"
        spec = importlib.util.spec_from_file_location(name, directory / "__init__.py",
                                                     submodule_search_locations=[str(directory)])
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
    return importlib.import_module(name + "." + module)


def agent(cls=DDPGAgent, **kwargs):
    return cls(state_dim=5, action_dim=2, hidden_dim=16, actor_lr=3e-4,
               critic_lr=3e-4, gamma=0.98, tau=0.005,
               action_low=np.full(2, -1, np.float32), action_high=np.full(2, 1, np.float32),
               device=torch.device("cpu"), l2_action_penalty=0, **kwargs)


def batch():
    g = torch.Generator().manual_seed(730)
    return dict(state=torch.randn(8, 5, generator=g),
                action_exec=torch.randn(8, 2, generator=g).clamp(-1, 1),
                reward=-torch.linspace(0, 5, 8), next_state=torch.randn(8, 5, generator=g),
                nonterminal=torch.tensor([1, 1, 0, 1, 1, 1, 0, 1.0]),
                omega_critic=torch.ones(8), omega_actor=torch.ones(8))


class LossTests(unittest.TestCase):
    def test_official_loss_and_gradient_parity(self):
        for dtype in (torch.float32, torch.float64):
            for alpha in (0.0, 0.4, 1.0):
                for threshold in (0.3, 1.0, 2.0):
                    x = torch.tensor([-10, -threshold, -0.01, 0, 0.01, threshold, 10],
                                     dtype=dtype, requires_grad=True)
                    y = x.detach().clone().requires_grad_(True)
                    result = prioritized_approximation_loss(x, alpha, threshold)
                    expected, z, base = official_loss(y, alpha, threshold)
                    torch.testing.assert_close(result.loss, expected, rtol=0, atol=0)
                    torch.testing.assert_close(result.normalizer, z, rtol=0, atol=0)
                    torch.testing.assert_close(result.base_loss, base, rtol=0, atol=0)
                    result.loss.backward()
                    expected.backward()
                    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)

    def test_normalizer_is_detached_and_small_residual_has_half_factor(self):
        x = torch.tensor([-0.3, 0, 0.7], dtype=torch.float64, requires_grad=True)
        r = prioritized_approximation_loss(x)
        self.assertFalse(r.normalizer.requires_grad)
        torch.testing.assert_close(r.loss, 0.5 * x.square().mean())
        x = torch.tensor([-3, -0.2, 0, 0.8, 4], dtype=torch.float64, requires_grad=True)
        r = prioritized_approximation_loss(x)
        r.loss.backward()
        expected = torch.where(x.detach().abs() < 1, x.detach(), x.detach().sign() * x.detach().abs().pow(0.4))
        torch.testing.assert_close(x.grad, expected / (len(x) * r.normalizer))

    def test_official_boundary_values_are_preserved(self):
        x = torch.tensor([1.0, -1.0], dtype=torch.float64, requires_grad=True)
        r = prioritized_approximation_loss(x)
        self.assertAlmostEqual(r.loss.item(), 1 / 1.4)
        self.assertEqual(r.large_error_fraction.item(), 1)

    def test_nonfinite_and_invalid_inputs_fail(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(FloatingPointError):
                prioritized_approximation_loss(torch.tensor([value]))
        for values in (torch.tensor([]), torch.tensor([1], dtype=torch.int64)):
            with self.assertRaises(ValueError):
                prioritized_approximation_loss(values)
        for alpha, threshold in ((-1, 1), (2, 1), (float("nan"), 1), (0.4, 0)):
            with self.assertRaises(ValueError):
                prioritized_approximation_loss(torch.ones(3), alpha, threshold)


class UpdateTests(unittest.TestCase):
    def assert_tree_equal(self, a, b):
        if isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        elif torch.is_tensor(a):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            self.assertEqual(a, b)

    def test_mse_path_optimizer_actor_and_target_match_frozen_source(self):
        old, new = agent(upstream("agent").DDPGAgent), agent()
        new.load_state_dict(copy.deepcopy(old.state_dict()))
        data = batch()
        for _ in range(3):
            a, b = old.update(**data, fail_on_nonfinite=True), new.update(**data, fail_on_nonfinite=True)
            self.assertEqual(a.critic_loss, b.critic_loss)
            self.assertEqual(a.actor_loss, b.actor_loss)
            self.assert_tree_equal(old.state_dict(), new.state_dict())

    def test_pal_update_matches_official_single_critic_reference(self):
        actual, expected = agent(critic_objective="pal"), agent(critic_objective="pal")
        expected.load_state_dict(copy.deepcopy(actual.state_dict()))

        def reference(delta, alpha, threshold):
            loss, z, base = official_loss(delta, alpha, threshold)
            return PALLoss(loss, z, base, (delta.detach().abs() >= threshold).float().mean())

        for _ in range(3):
            a = actual.update(**batch())
            with patch("metacog_lite_v0.agent.prioritized_approximation_loss", reference):
                b = expected.update(**batch())
            self.assertEqual(a.critic_loss, b.critic_loss)
            self.assertEqual(a.actor_loss, b.actor_loss)
            self.assert_tree_equal(actual.state_dict(), expected.state_dict())

    def test_single_preupdate_target_and_replay_action(self):
        a, data = agent(critic_objective="pal"), batch()
        calls, actions = [], []
        h1 = a.critic_target.register_forward_hook(lambda *args: calls.append(1))
        h2 = a.critic.register_forward_pre_hook(lambda module, args: actions.append(args[1].clone()))
        a.update(**data)
        h1.remove()
        h2.remove()
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(actions[0], data["action_exec"], rtol=0, atol=0)

    def test_pal_rejects_weight_masks_and_nonfinite_before_parameter_update(self):
        a, data = agent(critic_objective="pal"), batch()
        with self.assertRaises(ValueError):
            a.update(**data, critic_weight_selector=lambda x: x)
        with self.assertRaises(ValueError):
            a.update(**{**data, "omega_critic": torch.full((8,), 2.0)})
        with self.assertRaises(ValueError):
            a.update(**{**data, "omega_actor": torch.full((8,), 2.0)})
        before = copy.deepcopy(a.critic.state_dict())
        data["reward"][0] = float("nan")
        with self.assertRaises(FloatingPointError):
            a.update(**data)
        self.assert_tree_equal(before, a.critic.state_dict())


class ReplayAndProtocolTests(unittest.TestCase):
    def test_replay_and_env_sources_are_unchanged(self):
        for name in ("replay.py", "envs.py", "networks.py", "task_drawer_open.py", "io_utils.py"):
            self.assertEqual((ROOT / "code/src/metacog_lite_v0" / name).read_bytes(),
                             (ROOT / "upstream/code/src/metacog_lite_v0" / name).read_bytes())

    def test_no_index_has_no_effect_on_her_samples_after_wrap(self):
        indexed = HERReplayBuffer(12, 2, 1, 2)
        plain = HERReplayBuffer(12, 2, 1, 2)
        for t in range(40):
            data = dict(observation=np.array([t, 0]), achieved_goal=np.array([t]),
                        desired_goal=np.array([-1]), action_exec=np.zeros(2), reward_ext=-1,
                        next_observation=np.array([t+1, 0]), next_achieved_goal=np.array([t+1]),
                        done=t % 5 == 4, timeout=t % 5 == 4, h=5-t % 5,
                        episode_id=t // 5, episode_step=t % 5)
            indexed.add_transition(**data, coarse_key=t % 3, action_bin=t % 4)
            plain.add_transition(**data)
        r1, r2 = np.random.default_rng(71), np.random.default_rng(71)
        for _ in range(20):
            x, y = indexed.sample(8, 0.8, r1), plain.sample(8, 0.8, r2)
            for key in x.__dataclass_fields__:
                np.testing.assert_array_equal(getattr(x, key), getattr(y, key))
        self.assertEqual(plain.indexed_slot_count(), 0)

    def test_future_her_excludes_immediate_next_achieved_goal(self):
        r = HERReplayBuffer(10, 2, 1, 2)
        for t in range(5):
            r.add_transition(observation=np.zeros(2), achieved_goal=np.array([t]),
                desired_goal=np.array([-1]), action_exec=np.zeros(2), reward_ext=-1,
                next_observation=np.zeros(2), next_achieved_goal=np.array([t+1]),
                done=t==4, timeout=t==4, h=5-t, episode_id=0, episode_step=t)
        rng = np.random.default_rng(9)
        for slot in range(4):
            for _ in range(20):
                self.assertGreaterEqual(float(r._sample_future_goal(slot, rng)[0]), slot+2)
        self.assertIsNone(r._sample_future_goal(4, rng))

    def test_frozen_upstream_protocol_parity_for_all_seeds(self):
        # The frozen protocol resolves the local copied dataclass, whose new fields
        # have inert defaults; remove only these fields to compare old records.
        spec = importlib.util.spec_from_file_location("pal_reference_protocol", ROOT / "upstream/protocol.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for seed in range(1, 11):
            old = module.make_config("support", seed, "/tmp/not-a-run").as_dict()
            for key in PAL_ONLY_FIELDS:
                old.pop(key)
            self.assertEqual(old, expected_reference_config(seed, "/tmp/not-a-run"))

    def test_profile_and_scope_guards(self):
        cfg = make_config("pal", 1, "/tmp/not-a-run")
        self.assertEqual((cfg.total_steps, cfg.updates_per_step, cfg.batch_size), (2_000_000, 1, 256))
        self.assertFalse(cfg.assignment_experiment)
        self.assertFalse(cfg.numba_support_scan)
        smoke = make_config("pal", 1, "/tmp/not-a-run", "smoke", "cpu")
        self.assertEqual(smoke.total_steps, 6000)
        for key, value in (("resume", True), ("lambda_support", 1), ("pal_alpha", 0.6),
                           ("assignment_experiment", True), ("weight_scope", "actor_critic")):
            changed = copy.deepcopy(cfg)
            setattr(changed, key, value)
            with self.assertRaises(ValueError):
                changed.normalize()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

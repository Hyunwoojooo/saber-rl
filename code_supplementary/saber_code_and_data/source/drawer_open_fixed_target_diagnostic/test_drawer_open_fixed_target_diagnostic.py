from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from metacog_lite_v0.agent import DDPGAgent
from metacog_lite_v0.trainer import _b3_residual_diagnostic_metrics


class DrawerOpenFixedTargetDiagnosticContractTest(unittest.TestCase):
    @staticmethod
    def _agent() -> DDPGAgent:
        return DDPGAgent(
            state_dim=3,
            action_dim=2,
            hidden_dim=8,
            actor_lr=3e-4,
            critic_lr=3e-4,
            gamma=0.98,
            tau=0.005,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            device=torch.device("cpu"),
            l2_action_penalty=0.0,
        )

    @staticmethod
    def _batch() -> tuple[torch.Tensor, ...]:
        generator = torch.Generator().manual_seed(19)
        state = torch.randn(8, 3, generator=generator)
        action = torch.tanh(torch.randn(8, 2, generator=generator))
        reward = torch.randn(8, generator=generator)
        next_state = torch.randn(8, 3, generator=generator)
        nonterminal = torch.ones(8)
        weights = torch.ones(8)
        return state, action, reward, next_state, nonterminal, weights

    def test_metrics_use_fixed_target_and_preserve_live_target_audit(self) -> None:
        td_before = np.arange(1.0, 21.0, dtype=np.float32)
        support = np.zeros(20, dtype=np.float32)
        support[-4:] = 1.0
        td_after_fixed = td_before - np.where(support > 0.5, 2.0, 0.5)
        td_after_live = td_before - np.where(support > 0.5, 1.5, 0.25)
        weights = np.where(support > 0.5, 2.0, 1.0).astype(np.float32)
        weights /= np.mean(weights)
        wb = SimpleNamespace(psi_q_rank=support, omega_critic=weights)

        metrics = _b3_residual_diagnostic_metrics(
            wb=wb,
            td_before=td_before,
            td_after_fixed=td_after_fixed,
            td_after_live=td_after_live,
        )

        self.assertEqual(metrics["diagnostic/fixed_target_from_pre_update"], 1.0)
        self.assertEqual(metrics["diagnostic/same_minibatch_one_update"], 1.0)
        self.assertEqual(metrics["diagnostic/snapshot_only_collection"], 1.0)
        self.assertAlmostEqual(metrics["diagnostic/residual_reduction_support_pos_mean"], 2.0)
        self.assertAlmostEqual(metrics["diagnostic/residual_reduction_support_neg_mean"], 0.5)
        self.assertAlmostEqual(
            metrics["diagnostic/live_target_residual_reduction_support_pos_mean"], 1.5
        )
        self.assertAlmostEqual(
            metrics["diagnostic/live_target_residual_reduction_support_neg_mean"], 0.25
        )
        self.assertEqual(metrics["diagnostic/td_error_top10_count"], 2.0)
        self.assertAlmostEqual(metrics["diagnostic/td_error_top10_fraction"], 0.1)
        self.assertAlmostEqual(metrics["diagnostic/td_error_top10_enrichment_ratio"], 5.0)

    def test_top10_uses_exact_stable_k_when_td_errors_tie(self) -> None:
        td = np.ones(20, dtype=np.float32)
        support = np.zeros(20, dtype=np.float32)
        support[-2:] = 1.0
        wb = SimpleNamespace(psi_q_rank=support, omega_critic=np.ones(20, dtype=np.float32))
        metrics = _b3_residual_diagnostic_metrics(wb, td, td, td)

        self.assertEqual(metrics["diagnostic/td_error_top10_count"], 2.0)
        self.assertAlmostEqual(metrics["diagnostic/td_error_top10_fraction"], 0.1)
        self.assertAlmostEqual(metrics["diagnostic/td_error_top10_support_rate"], 0.0)

    def test_supplied_pre_update_target_is_used_without_mutation(self) -> None:
        torch.manual_seed(7)
        agent = self._agent()
        state, action, reward, next_state, nonterminal, weights = self._batch()
        fixed_target = torch.linspace(-1.0, 1.0, state.shape[0])
        fixed_copy = fixed_target.clone()

        update = agent.update(
            state=state,
            action_exec=action,
            reward=reward,
            next_state=next_state,
            nonterminal=nonterminal,
            omega_critic=weights,
            omega_actor=weights,
            fixed_target=fixed_target,
        )

        self.assertAlmostEqual(update.target_mean, float(fixed_target.mean()), places=7)
        torch.testing.assert_close(fixed_target, fixed_copy)

    def test_instrumentation_does_not_change_parameter_update(self) -> None:
        torch.manual_seed(11)
        baseline = self._agent()
        instrumented = self._agent()
        instrumented.load_state_dict(copy.deepcopy(baseline.state_dict()))
        state, action, reward, next_state, nonterminal, weights = self._batch()
        _, fixed_target, _ = instrumented.td_error_components(
            state, action, reward, next_state, nonterminal
        )

        baseline.update(state, action, reward, next_state, nonterminal, weights, weights)
        instrumented.update(
            state,
            action,
            reward,
            next_state,
            nonterminal,
            weights,
            weights,
            fixed_target=fixed_target,
        )

        for name in ("actor", "actor_target", "critic", "critic_target"):
            left = getattr(baseline, name).state_dict()
            right = getattr(instrumented, name).state_dict()
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0.0, atol=0.0)

    def test_snapshot_diff_is_limited_to_agent_and_trainer(self) -> None:
        source_root = Path(__file__).resolve().parent
        repo_root = next(parent for parent in source_root.parents if (parent / ".git").exists())
        main_root = repo_root / "metaworld_Drawer_Open_DDPG" / "code" / "src"
        unchanged = (
            "train_drawer_open_lite_v0.py",
            "metacog_lite_v0/config.py",
            "metacog_lite_v0/envs.py",
            "metacog_lite_v0/io_utils.py",
            "metacog_lite_v0/networks.py",
            "metacog_lite_v0/numba_support_scan_drawer_open.py",
            "metacog_lite_v0/replay.py",
            "metacog_lite_v0/task_drawer_open.py",
            "metacog_lite_v0/weights.py",
        )
        for relative in unchanged:
            with self.subTest(relative=relative):
                self.assertEqual(
                    (source_root / relative).read_bytes(),
                    (main_root / relative).read_bytes(),
                )

    def test_resume_is_disabled_and_snapshot_collection_is_explicit(self) -> None:
        source_root = Path(__file__).resolve().parent
        experiment_root = source_root.parents[1]
        trainer = (source_root / "metacog_lite_v0" / "trainer.py").read_text(encoding="utf-8")
        launcher = (experiment_root / "run_5seeds_parallel.sh").read_text(encoding="utf-8")
        self.assertIn("Resume is disabled for DrawerOpen", trainer)
        self.assertIn("diagnostic_snapshot_due", trainer)
        self.assertIn("fixed_target=diagnostic_target_before", trainer)
        self.assertNotIn("--resume", launcher)
        self.assertNotIn("RESUME=", launcher)


if __name__ == "__main__":
    unittest.main()

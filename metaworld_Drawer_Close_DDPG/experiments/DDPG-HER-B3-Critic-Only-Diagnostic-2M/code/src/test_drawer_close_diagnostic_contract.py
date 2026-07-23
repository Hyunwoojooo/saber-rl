from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from metacog_lite_v0.agent import DDPGAgent
from metacog_lite_v0.trainer import _b3_residual_diagnostic_metrics


PHASES = ("approach", "contact", "closing", "closed")


class DrawerCloseDiagnosticContractTest(unittest.TestCase):
    def test_four_diagnostics_use_fixed_and_live_targets(self) -> None:
        td_before = np.arange(1.0, 21.0, dtype=np.float32)
        support = np.zeros(20, dtype=np.float32)
        support[-4:] = 1.0
        fixed_reduction = np.where(
            support > 0.5,
            2.0,
            0.5,
        ).astype(np.float32)
        live_reduction = np.where(
            support > 0.5,
            1.5,
            0.25,
        ).astype(np.float32)
        td_after_fixed = td_before - fixed_reduction
        td_after_live = td_before - live_reduction
        phases = np.repeat(np.arange(4, dtype=np.int64), 5)
        weights = np.where(
            support > 0.5,
            2.0,
            1.0,
        ).astype(np.float32)
        weights /= np.mean(weights)
        wb = SimpleNamespace(
            psi_q_rank=support.copy(),
            omega_critic=weights.copy(),
            phase=phases.copy(),
        )

        before_copy = td_before.copy()
        fixed_copy = td_after_fixed.copy()
        live_copy = td_after_live.copy()
        metrics = _b3_residual_diagnostic_metrics(
            wb,
            td_before,
            td_after_fixed,
            td_after_live,
            PHASES,
        )

        self.assertEqual(
            metrics["diagnostic/fixed_target_from_pre_update"],
            1.0,
        )
        self.assertAlmostEqual(
            metrics["diagnostic/support_positive_rate"],
            0.2,
        )
        self.assertGreater(
            metrics["diagnostic/support_to_td_error_corr"],
            0.0,
        )
        self.assertEqual(
            metrics["diagnostic/td_error_top10_count"],
            2.0,
        )
        self.assertAlmostEqual(
            metrics["diagnostic/td_error_top10_fraction"],
            0.1,
        )
        self.assertAlmostEqual(
            metrics["diagnostic/td_error_top10_support_rate"],
            1.0,
        )
        self.assertAlmostEqual(
            metrics["diagnostic/td_error_top10_enrichment_ratio"],
            5.0,
        )
        self.assertAlmostEqual(
            metrics[
                "diagnostic/residual_reduction_support_pos_mean"
            ],
            2.0,
        )
        self.assertAlmostEqual(
            metrics[
                "diagnostic/residual_reduction_support_neg_mean"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            metrics[
                "diagnostic/"
                "live_target_residual_reduction_support_pos_mean"
            ],
            1.5,
        )
        self.assertAlmostEqual(
            metrics[
                "diagnostic/"
                "live_target_residual_reduction_support_neg_mean"
            ],
            0.25,
        )
        self.assertEqual(
            metrics["diagnostic/same_minibatch_one_update"],
            1.0,
        )
        for phase in PHASES:
            self.assertAlmostEqual(
                metrics[f"diagnostic/phase_{phase}_batch_rate"],
                0.25,
            )
            self.assertIn(
                f"diagnostic/phase_{phase}_support_positive_rate",
                metrics,
            )
        np.testing.assert_array_equal(td_before, before_copy)
        np.testing.assert_array_equal(td_after_fixed, fixed_copy)
        np.testing.assert_array_equal(td_after_live, live_copy)

    def test_top10_uses_exact_stable_k_when_td_errors_tie(self) -> None:
        td = np.ones(20, dtype=np.float32)
        support = np.zeros(20, dtype=np.float32)
        support[-2:] = 1.0
        wb = SimpleNamespace(
            psi_q_rank=support,
            omega_critic=np.ones(20, dtype=np.float32),
            phase=np.zeros(20, dtype=np.int64),
        )
        metrics = _b3_residual_diagnostic_metrics(
            wb,
            td,
            td,
            td,
            PHASES,
        )

        self.assertEqual(
            metrics["diagnostic/td_error_top10_count"],
            2.0,
        )
        self.assertAlmostEqual(
            metrics["diagnostic/td_error_top10_fraction"],
            0.1,
        )
        self.assertAlmostEqual(
            metrics["diagnostic/td_error_top10_support_rate"],
            0.0,
        )

    def test_agent_update_uses_the_supplied_pre_update_target(self) -> None:
        torch.manual_seed(7)
        device = torch.device("cpu")
        agent = DDPGAgent(
            state_dim=3,
            action_dim=2,
            hidden_dim=8,
            actor_lr=3e-4,
            critic_lr=3e-4,
            gamma=0.98,
            tau=0.005,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            device=device,
            l2_action_penalty=0.0,
        )
        state = torch.randn(4, 3)
        action = torch.tanh(torch.randn(4, 2))
        reward = torch.randn(4)
        next_state = torch.randn(4, 3)
        nonterminal = torch.ones(4)
        fixed_target = torch.tensor(
            [-0.25, -0.50, -0.75, -1.00],
            dtype=torch.float32,
        )
        fixed_copy = fixed_target.clone()

        update = agent.update(
            state=state,
            action_exec=action,
            reward=reward,
            next_state=next_state,
            nonterminal=nonterminal,
            omega_critic=torch.ones(4),
            omega_actor=torch.ones(4),
            fixed_target=fixed_target,
        )

        self.assertAlmostEqual(
            update.target_mean,
            float(fixed_target.mean()),
            places=7,
        )
        torch.testing.assert_close(fixed_target, fixed_copy)

    def test_non_instrumented_snapshot_matches_drawer_close_main(self) -> None:
        source_root = Path(__file__).resolve().parent
        repo_root = Path(__file__).resolve().parents[5]
        main_root = (
            repo_root / "metaworld_Drawer_Close_DDPG" / "code" / "src"
        )
        paths = (
            "train_drawer_close_lite_v0.py",
            "metacog_lite_v0/config.py",
            "metacog_lite_v0/envs.py",
            "metacog_lite_v0/replay.py",
            "metacog_lite_v0/task_drawer_close.py",
            "metacog_lite_v0/weights.py",
        )
        for relative in paths:
            with self.subTest(relative=relative):
                self.assertEqual(
                    (source_root / relative).read_bytes(),
                    (main_root / relative).read_bytes(),
                )

    def test_resume_is_disabled_in_trainer_and_launcher(self) -> None:
        source_root = Path(__file__).resolve().parent
        experiment_root = source_root.parents[1]
        trainer_text = (
            source_root / "metacog_lite_v0" / "trainer.py"
        ).read_text(encoding="utf-8")
        launcher_text = (
            experiment_root
            / "DDPG-HER-B3-Critic-Only-Diagnostic-2M.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("Resume is disabled for DrawerClose", trainer_text)
        self.assertNotIn("load_checkpoint(", trainer_text)
        self.assertNotIn("RESUME=", launcher_text)
        self.assertNotIn("--resume", launcher_text)


if __name__ == "__main__":
    unittest.main()

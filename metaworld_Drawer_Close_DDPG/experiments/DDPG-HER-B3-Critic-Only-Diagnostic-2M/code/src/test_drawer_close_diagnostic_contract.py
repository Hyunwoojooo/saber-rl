from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from metacog_lite_v0.trainer import _b3_residual_diagnostic_metrics


PHASES = ("approach", "contact", "closing", "closed")


class DrawerCloseDiagnosticContractTest(unittest.TestCase):
    def test_four_diagnostics_are_computed_from_one_batch(self) -> None:
        td_before = np.arange(1.0, 21.0, dtype=np.float32)
        support = np.zeros(20, dtype=np.float32)
        support[-4:] = 1.0
        reduction = np.where(support > 0.5, 2.0, 0.5).astype(np.float32)
        td_after = td_before - reduction
        phases = np.repeat(np.arange(4, dtype=np.int64), 5)
        weights = np.where(support > 0.5, 2.0, 1.0).astype(np.float32)
        weights /= np.mean(weights)
        wb = SimpleNamespace(
            psi_q_rank=support.copy(),
            omega_critic=weights.copy(),
            phase=phases.copy(),
        )

        before_copy = td_before.copy()
        after_copy = td_after.copy()
        metrics = _b3_residual_diagnostic_metrics(
            wb,
            td_before,
            td_after,
            PHASES,
        )

        self.assertAlmostEqual(metrics["diagnostic/support_positive_rate"], 0.2)
        self.assertGreater(metrics["diagnostic/support_to_td_error_corr"], 0.0)
        self.assertAlmostEqual(metrics["diagnostic/td_error_top10_support_rate"], 1.0)
        self.assertAlmostEqual(metrics["diagnostic/td_error_top10_enrichment_ratio"], 5.0)
        self.assertAlmostEqual(metrics["diagnostic/residual_reduction_support_pos_mean"], 2.0)
        self.assertAlmostEqual(metrics["diagnostic/residual_reduction_support_neg_mean"], 0.5)
        self.assertEqual(metrics["diagnostic/same_minibatch_one_update"], 1.0)
        for phase in PHASES:
            self.assertAlmostEqual(metrics[f"diagnostic/phase_{phase}_batch_rate"], 0.25)
            self.assertIn(f"diagnostic/phase_{phase}_support_positive_rate", metrics)
        np.testing.assert_array_equal(td_before, before_copy)
        np.testing.assert_array_equal(td_after, after_copy)

    def test_non_trainer_snapshot_matches_drawer_close_main(self) -> None:
        source_root = Path(__file__).resolve().parent
        repo_root = Path(__file__).resolve().parents[5]
        main_root = repo_root / "metaworld_Drawer_Close_DDPG" / "code" / "src"
        paths = (
            "train_drawer_close_lite_v0.py",
            "metacog_lite_v0/agent.py",
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


if __name__ == "__main__":
    unittest.main()

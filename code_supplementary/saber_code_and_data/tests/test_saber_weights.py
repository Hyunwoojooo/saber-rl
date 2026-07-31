from __future__ import annotations

import unittest

import numpy as np
import torch

from _source_import import add_fpp_ddpg

add_fpp_ddpg()
from metacog_lite_v0.weights import GapStats, LiteWeightComputer  # noqa: E402
from test_b3_runtime_opt_lookup import (  # noqa: E402
    FakeAgent,
    MiniTask,
    make_batch,
    make_config,
    make_replay,
    naive_deterministic,
)


class SABERWeightContractTest(unittest.TestCase):
    def test_exact_support_and_mean_one_critic_only_weights(self) -> None:
        np.random.seed(0)
        torch.manual_seed(0)
        config = make_config()
        task = MiniTask()
        replay = make_replay(task, config)
        batch = make_batch(replay)
        agent = FakeAgent()
        expected = naive_deterministic(config, task, batch, replay, agent)
        computer = LiteWeightComputer(
            config, task, GapStats(16, 4, 0.6, 0.95, 1e-6)
        )
        actual = computer.compute(
            batch, replay, agent, np.random.default_rng(123), torch.device("cpu")
        )

        np.testing.assert_array_equal(actual.psi_q_rank, expected["psi_q_rank"])
        np.testing.assert_array_equal(actual.psi_a_policy, expected["psi_a_policy"])
        np.testing.assert_array_equal(actual.w_p_raw, expected["w_p_raw"])
        np.testing.assert_allclose(
            actual.omega_critic, expected["omega_critic"], atol=0.0, rtol=0.0
        )
        np.testing.assert_array_equal(actual.omega_actor, np.ones(batch.indices.size))
        self.assertAlmostEqual(float(np.mean(actual.omega_critic)), 1.0, places=7)
        np.testing.assert_array_equal(
            actual.w_p_raw,
            1.0 + config.lambda_support * actual.psi_q_rank,
        )


if __name__ == "__main__":
    unittest.main()

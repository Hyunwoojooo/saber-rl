from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parent / "code" / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from metacog_lite_v0.replay import HERReplayBuffer  # noqa: E402


class FirstChoiceRng:
    def integers(self, low: int, high: int | None = None, **_: object) -> int:
        if high is None:
            return 0
        return int(low)


def build_buffer() -> HERReplayBuffer:
    replay = HERReplayBuffer(capacity=8, obs_dim=1, goal_dim=1, action_dim=1)
    for step in range(4):
        value = np.array([float(step)], dtype=np.float32)
        replay.add_transition(
            observation=value,
            achieved_goal=value,
            desired_goal=np.array([-99.0], dtype=np.float32),
            action_exec=np.zeros(1, dtype=np.float32),
            reward_ext=-1.0,
            next_observation=value + 1.0,
            next_achieved_goal=value + 1.0,
            done=False,
            timeout=False,
            h=4 - step,
            episode_id=7,
            episode_step=step,
        )
    return replay


def test_current_transition_is_an_eligible_future_goal_source() -> None:
    replay = build_buffer()
    first = FirstChoiceRng()
    np.testing.assert_array_equal(replay._sample_future_goal(1, first), np.array([2.0], dtype=np.float32))
    np.testing.assert_array_equal(replay._sample_future_goal(3, first), np.array([4.0], dtype=np.float32))


def test_all_transitions_are_relabelable_at_unit_probability() -> None:
    replay = build_buffer()
    batch = replay.sample(batch_size=4, her_ratio=1.0, rng=np.random.default_rng(314159))
    assert batch.her_relabel_ratio == 1.0
    assert np.all(batch.desired_goal_train > -99.0)


if __name__ == "__main__":
    test_current_transition_is_an_eligible_future_goal_source()
    test_all_transitions_are_relabelable_at_unit_probability()
    print("inclusive future-goal HER tests passed")

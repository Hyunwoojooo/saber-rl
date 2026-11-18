from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class UnitreeAppleTaskSpec:
    action_low: np.ndarray
    action_high: np.ndarray
    epsilon_g: float = 0.05
    d_reach: float = 0.10
    d_goal_near: float = 0.07
    hand_apple_bins: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    apple_goal_bins: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    apple_z_bins: Tuple[float, ...] = (0.75, 0.85, 1.00)
    horizon_bins: Tuple[int, ...] = (10, 25, 50, 100)

    right_tcp_slice: Tuple[int, int] = (0, 3)
    apple_slice: Tuple[int, int] = (3, 6)
    is_grasped_index: int = 9

    PHASE_APPROACH: int = 0
    PHASE_GRASP: int = 1
    PHASE_TRANSPORT: int = 2
    PHASE_RELEASE_PLACE: int = 3
    phase_names: Tuple[str, ...] = ("approach", "grasp", "transport", "release_place")

    def make_state(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, goal], axis=-1).astype(np.float32)

    def parse_features(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        right_tcp = obs[:, self.right_tcp_slice[0]:self.right_tcp_slice[1]]
        apple = obs[:, self.apple_slice[0]:self.apple_slice[1]]
        apple_z = apple[:, 2]
        is_grasped = obs[:, self.is_grasped_index]
        return right_tcp, apple, apple_z, is_grasped

    def phase_batch(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        right_tcp, apple, _apple_z, is_grasped = self.parse_features(obs)
        d_hand_apple = np.linalg.norm(right_tcp - apple, axis=1)
        d_apple_goal = np.linalg.norm(apple - goal, axis=1)
        phase = np.full(obs.shape[0], self.PHASE_RELEASE_PLACE, dtype=np.int64)
        phase[d_apple_goal > self.d_goal_near] = self.PHASE_TRANSPORT
        phase[is_grasped < 0.5] = self.PHASE_GRASP
        phase[d_hand_apple > self.d_reach] = self.PHASE_APPROACH
        return phase

    def coarse_hash_batch(self, obs: np.ndarray, h: np.ndarray) -> np.ndarray:
        right_tcp, apple, apple_z, is_grasped = self.parse_features(obs)
        d_hand_apple = np.linalg.norm(right_tcp - apple, axis=1)
        hand_bin = self._bin_array(d_hand_apple, self.hand_apple_bins)
        z_bin = self._bin_array(apple_z, self.apple_z_bins)
        grasp_bin = (is_grasped >= 0.5).astype(np.int64)
        h_bin = self._bin_array(h.astype(np.float32), tuple(float(v) for v in self.horizon_bins))
        return (hand_bin * 1_000 + z_bin * 100 + grasp_bin * 10 + h_bin).astype(np.int64)

    def coarse_hash(self, obs: np.ndarray, h: int) -> int:
        return int(self.coarse_hash_batch(obs[None, :], np.asarray([h], dtype=np.int64))[0])

    def full_context_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        right_tcp, apple, apple_z, is_grasped = self.parse_features(obs)
        d_hand_apple = np.linalg.norm(right_tcp - apple, axis=1)
        d_apple_goal = np.linalg.norm(apple - goal, axis=1)
        phase = self.phase_batch(obs, goal)
        hand_bin = self._bin_array(d_hand_apple, self.hand_apple_bins)
        goal_bin = self._bin_array(d_apple_goal, self.apple_goal_bins)
        z_bin = self._bin_array(apple_z, self.apple_z_bins)
        grasp_bin = (is_grasped >= 0.5).astype(np.int64)
        h_bin = self._bin_array(h.astype(np.float32), tuple(float(v) for v in self.horizon_bins))
        return (
            phase * 100_000
            + hand_bin * 10_000
            + goal_bin * 1_000
            + z_bin * 100
            + grasp_bin * 10
            + h_bin
        ).astype(np.int64)

    def full_context_hash(self, obs: np.ndarray, goal: np.ndarray, h: int) -> int:
        return int(
            self.full_context_hash_batch(
                obs[None, :],
                goal[None, :],
                np.asarray([h], dtype=np.int64),
            )[0]
        )

    def action_bin_batch(self, action: np.ndarray, bins_per_dim: int) -> np.ndarray:
        norm = self.normalize_action(action)
        scaled = (norm + 1.0) * 0.5 * float(bins_per_dim)
        bins = np.floor(np.clip(scaled, 0.0, float(bins_per_dim - 1))).astype(np.int64)
        # High-dimensional humanoid action bins are used only as a stored signature
        # for non-radius ablations. Use a bounded rolling hash to avoid int overflow.
        out = np.zeros(action.shape[0], dtype=np.int64)
        for dim in range(bins.shape[1]):
            out = (out * 131 + bins[:, dim] + 1) % 2_147_483_647
        return out.astype(np.int64)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        center = 0.5 * (self.action_high + self.action_low)
        scale = np.maximum(1e-6, 0.5 * (self.action_high - self.action_low))
        return ((action - center) / scale).astype(np.float32)

    def metadata(self) -> Dict[str, object]:
        return {
            "task": "UnitreeApple",
            "env_id": "UnitreeG1PlaceAppleInBowl-v1",
            "coarse_index_key": "hand-apple distance bin, apple z bin, grasp bin, horizon bin",
            "full_context_key": "phase, hand-apple distance bin, apple-goal distance bin, apple z bin, grasp bin, horizon bin",
            "phase_labels": {
                self.PHASE_APPROACH: "approach",
                self.PHASE_GRASP: "grasp",
                self.PHASE_TRANSPORT: "transport",
                self.PHASE_RELEASE_PLACE: "release_place",
            },
            "observation_layout": {
                "right_tcp_pos": list(self.right_tcp_slice),
                "apple_pos": list(self.apple_slice),
                "is_grasped_index": self.is_grasped_index,
            },
            "epsilon_g": self.epsilon_g,
            "d_reach": self.d_reach,
            "d_goal_near": self.d_goal_near,
            "hand_apple_bins": list(self.hand_apple_bins),
            "apple_goal_bins": list(self.apple_goal_bins),
            "apple_z_bins": list(self.apple_z_bins),
            "horizon_bins": list(self.horizon_bins),
        }

    @staticmethod
    def _bin_array(values: np.ndarray, boundaries: Tuple[float, ...]) -> np.ndarray:
        out = np.zeros(values.shape[0], dtype=np.int64)
        for boundary in boundaries:
            out += values > float(boundary)
        return out

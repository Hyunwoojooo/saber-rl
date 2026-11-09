from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class FetchSlideTaskSpec:
    action_low: np.ndarray
    action_high: np.ndarray
    epsilon_g: float
    d_contact: float = 0.08
    d_goal_near: float = 0.05
    v_slide_min: float = 0.05
    v_fast: float = 0.15
    ee_obj_bins: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    obj_goal_bins: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.70, 1.00)
    obj_speed_bins: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.40)
    vel_align_bins: Tuple[float, ...] = (-0.50, 0.0, 0.50, 0.80)
    horizon_bins: Tuple[int, ...] = (10, 25, 50, 100)

    ee_slice: Tuple[int, int] = (0, 3)
    object_slice: Tuple[int, int] = (3, 6)
    object_velocity_slice: Tuple[int, int] = (14, 17)
    ee_velocity_slice: Tuple[int, int] = (20, 23)

    PHASE_APPROACH: int = 0
    PHASE_STRIKE: int = 1
    PHASE_SLIDE: int = 2
    PHASE_NEAR_GOAL: int = 3
    phase_names: Tuple[str, ...] = ("approach", "strike", "slide", "near_goal")

    def make_state(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, goal], axis=-1).astype(np.float32)

    def parse_features(
        self,
        obs: np.ndarray,
        goal: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim != 2:
            raise ValueError(f"Expected batched observation with ndim=2, got shape={obs_arr.shape}")
        required_dim = max(self.ee_velocity_slice[1], self.object_velocity_slice[1], self.object_slice[1])
        if obs_arr.shape[1] < required_dim:
            raise ValueError(
                "FetchSlide observation must expose Fetch-style object and velocity slices; "
                f"need obs_dim>={required_dim}, got obs_dim={obs_arr.shape[1]}"
            )
        p_ee = obs_arr[:, self.ee_slice[0]:self.ee_slice[1]]
        p_obj = obs_arr[:, self.object_slice[0]:self.object_slice[1]]
        v_obj = obs_arr[:, self.object_velocity_slice[0]:self.object_velocity_slice[1]]
        v_ee = obs_arr[:, self.ee_velocity_slice[0]:self.ee_velocity_slice[1]]
        d_ee_obj = np.linalg.norm(p_ee - p_obj, axis=1)
        obj_speed_xy = np.linalg.norm(v_obj[:, :2], axis=1)
        if goal is None:
            vel_align = np.zeros(obs_arr.shape[0], dtype=np.float32)
        else:
            goal_arr = np.asarray(goal, dtype=np.float32)
            goal_xy = goal_arr[:, :2] - p_obj[:, :2]
            goal_norm = np.linalg.norm(goal_xy, axis=1, keepdims=True)
            goal_dir = goal_xy / np.maximum(goal_norm, 1e-8)
            vel_align = np.sum(v_obj[:, :2] * goal_dir, axis=1) / np.maximum(obj_speed_xy, 1e-8)
            vel_align = np.clip(vel_align, -1.0, 1.0).astype(np.float32)
        return p_ee, p_obj, v_obj, v_ee, d_ee_obj.astype(np.float32), obj_speed_xy.astype(np.float32), vel_align

    def phase_batch(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        _, p_obj, _, _, d_ee_obj, obj_speed_xy, _ = self.parse_features(obs, goal)
        d_obj_goal = np.linalg.norm(p_obj - goal, axis=1)
        phase = np.full(obs.shape[0], self.PHASE_APPROACH, dtype=np.int64)
        phase[d_ee_obj <= self.d_contact] = self.PHASE_STRIKE
        phase[obj_speed_xy >= self.v_slide_min] = self.PHASE_SLIDE
        phase[d_obj_goal <= self.d_goal_near] = self.PHASE_NEAR_GOAL
        return phase

    def bucket_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        return self.full_context_hash_batch(obs, goal, h)

    def bucket_hash(self, obs: np.ndarray, goal: np.ndarray, h: int) -> int:
        return self.full_context_hash(obs, goal, h)

    def coarse_hash_batch(self, obs: np.ndarray, h: np.ndarray) -> np.ndarray:
        _, _, _, _, d_ee_obj, obj_speed_xy, _ = self.parse_features(obs, None)
        ee_bin = self._bin_array(d_ee_obj, self.ee_obj_bins)
        speed_bin = self._bin_array(obj_speed_xy, self.obj_speed_bins)
        h_bin = self._bin_array(h.astype(np.float32), tuple(float(v) for v in self.horizon_bins))
        return (ee_bin * 1_000 + speed_bin * 100 + h_bin).astype(np.int64)

    def coarse_hash(self, obs: np.ndarray, h: int) -> int:
        return int(self.coarse_hash_batch(obs[None, :], np.asarray([h], dtype=np.int64))[0])

    def full_context_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        _, p_obj, _, _, d_ee_obj, obj_speed_xy, vel_align = self.parse_features(obs, goal)
        d_obj_goal = np.linalg.norm(p_obj - goal, axis=1)
        phase = self.phase_batch(obs, goal)
        ee_bin = self._bin_array(d_ee_obj, self.ee_obj_bins)
        goal_bin = self._bin_array(d_obj_goal, self.obj_goal_bins)
        speed_bin = self._bin_array(obj_speed_xy, self.obj_speed_bins)
        align_bin = self._bin_array(vel_align, self.vel_align_bins)
        h_bin = self._bin_array(h.astype(np.float32), tuple(float(v) for v in self.horizon_bins))
        return (
            phase * 1_000_000
            + ee_bin * 100_000
            + goal_bin * 10_000
            + speed_bin * 1_000
            + align_bin * 100
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
        multipliers = (bins_per_dim ** np.arange(action.shape[1], dtype=np.int64))[None, :]
        return np.sum(bins * multipliers, axis=1).astype(np.int64)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        center = 0.5 * (self.action_high + self.action_low)
        scale = np.maximum(1e-6, 0.5 * (self.action_high - self.action_low))
        return ((action - center) / scale).astype(np.float32)

    def diagnostic_metrics(self, obs: np.ndarray, goal: np.ndarray) -> Dict[str, float]:
        _, _, _, _, _, obj_speed_xy, vel_align = self.parse_features(obs, goal)
        phase = self.phase_batch(obs, goal)
        metrics: Dict[str, float] = {
            "lite/slide_mean_obj_speed": float(np.mean(obj_speed_xy)) if obj_speed_xy.size else 0.0,
            "lite/slide_mean_vel_align": float(np.mean(vel_align)) if vel_align.size else 0.0,
        }
        for phase_id, phase_name in enumerate(self.phase_names):
            metrics[f"lite/slide_phase_distribution_{phase_name}"] = float(np.mean(phase == phase_id)) if phase.size else 0.0
        return metrics

    def metadata(self) -> Dict[str, object]:
        return {
            "task": "FetchSlide",
            "coarse_index_key": "ee-object distance bin, object speed bin, horizon bin",
            "full_context_key": (
                "phase, ee-object distance bin, object-goal distance bin, "
                "object speed bin, velocity alignment bin, horizon bin"
            ),
            "phase_labels": {idx: name for idx, name in enumerate(self.phase_names)},
            "phase_definition": (
                "near_goal if d_obj_goal<=d_goal_near; slide if obj_speed_xy>=v_slide_min; "
                "strike if d_ee_obj<=d_contact; else approach"
            ),
            "observation_layout_assumption": {
                "ee_pos": list(self.ee_slice),
                "object_pos": list(self.object_slice),
                "object_velocity": list(self.object_velocity_slice),
                "ee_velocity": list(self.ee_velocity_slice),
            },
            "epsilon_g": self.epsilon_g,
            "d_contact": self.d_contact,
            "d_goal_near": self.d_goal_near,
            "v_slide_min": self.v_slide_min,
            "v_fast": self.v_fast,
            "ee_obj_bins": list(self.ee_obj_bins),
            "obj_goal_bins": list(self.obj_goal_bins),
            "obj_speed_bins": list(self.obj_speed_bins),
            "vel_align_bins": list(self.vel_align_bins),
            "horizon_bins": list(self.horizon_bins),
        }

    @staticmethod
    def _bin_array(values: np.ndarray, boundaries: Tuple[float, ...]) -> np.ndarray:
        out = np.zeros(values.shape[0], dtype=np.int64)
        for boundary in boundaries:
            out += values > float(boundary)
        return out

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class PushTTaskSpec:
    action_low: np.ndarray
    action_high: np.ndarray
    pos_success_thresh: float = 0.05
    rot_success_thresh: float = 0.2617993877991494
    contact_thresh: float = 0.06
    pos_near_thresh: float = 0.05
    rot_near_thresh: float = 0.2617993877991494
    rot_large_thresh: float = 0.5235987755982988
    speed_thresh: float = 0.03
    ang_vel_thresh: float = 0.10
    tcp_obj_bins: Tuple[float, ...] = (0.03, 0.06, 0.10, 0.20, 0.40)
    obj_goal_dist_bins: Tuple[float, ...] = (0.03, 0.05, 0.10, 0.20, 0.40, 0.70)
    rot_err_bins_rad: Tuple[float, ...] = (0.15, 0.30, 0.52, 0.78, 1.57, 3.14)
    obj_speed_bins: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.10, 0.20)
    ang_vel_bins: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.50, 1.00)
    horizon_bins: Tuple[int, ...] = (10, 25, 50, 100)

    tcp_slice: Tuple[int, int] = (0, 3)
    obj_slice: Tuple[int, int] = (3, 6)
    obj_cos_index: int = 6
    obj_sin_index: int = 7
    tcp_to_obj_slice: Tuple[int, int] = (8, 11)
    obj_velocity_xy_slice: Tuple[int, int] = (11, 13)
    obj_ang_vel_z_index: int = 13
    tcp_velocity_xy_slice: Tuple[int, int] = (14, 16)

    PHASE_APPROACH: int = 0
    PHASE_CONTACT: int = 1
    PHASE_PUSH_TRANSLATE: int = 2
    PHASE_PUSH_ROTATE: int = 3
    PHASE_NEAR_GOAL: int = 4
    phase_names: Tuple[str, ...] = ("approach", "contact", "push_translate", "push_rotate", "near_goal")

    def make_state(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, goal], axis=-1).astype(np.float32)

    def extract_features(self, obs: np.ndarray, goal: np.ndarray) -> Dict[str, np.ndarray]:
        obs_arr = np.asarray(obs, dtype=np.float32)
        goal_arr = np.asarray(goal, dtype=np.float32)
        if obs_arr.ndim != 2:
            raise ValueError(f"Expected batched observation with ndim=2, got shape={obs_arr.shape}")
        if goal_arr.ndim != 2:
            raise ValueError(f"Expected batched goal with ndim=2, got shape={goal_arr.shape}")
        if obs_arr.shape[1] < self.tcp_velocity_xy_slice[1]:
            raise ValueError(
                "PushT compact observation must match the documented 16-D layout; "
                f"got obs_dim={obs_arr.shape[1]}"
            )
        if goal_arr.shape[1] < 4:
            raise ValueError(f"PushT desired_goal must be [x,y,cos(theta),sin(theta)], got {goal_arr.shape}")
        tcp = obs_arr[:, self.tcp_slice[0] : self.tcp_slice[1]]
        obj = obs_arr[:, self.obj_slice[0] : self.obj_slice[1]]
        obj_cos = obs_arr[:, self.obj_cos_index]
        obj_sin = obs_arr[:, self.obj_sin_index]
        tcp_to_obj = obs_arr[:, self.tcp_to_obj_slice[0] : self.tcp_to_obj_slice[1]]
        obj_vel_xy = obs_arr[:, self.obj_velocity_xy_slice[0] : self.obj_velocity_xy_slice[1]]
        obj_ang_vel_z = obs_arr[:, self.obj_ang_vel_z_index]
        tcp_vel_xy = obs_arr[:, self.tcp_velocity_xy_slice[0] : self.tcp_velocity_xy_slice[1]]
        pos_vec = goal_arr[:, :2] - obj[:, :2]
        pos_dist = np.linalg.norm(pos_vec, axis=1).astype(np.float32)
        rot_err = self.angle_error_from_cossin(obj_cos, obj_sin, goal_arr[:, 2], goal_arr[:, 3])
        tcp_obj_dist = np.linalg.norm(tcp - obj, axis=1).astype(np.float32)
        obj_speed_xy = np.linalg.norm(obj_vel_xy, axis=1).astype(np.float32)
        pos_norm = np.linalg.norm(pos_vec, axis=1, keepdims=True)
        goal_dir = pos_vec / np.maximum(pos_norm, 1e-8)
        vel_align = np.sum(obj_vel_xy * goal_dir, axis=1) / np.maximum(obj_speed_xy, 1e-8)
        vel_align = np.clip(vel_align, -1.0, 1.0).astype(np.float32)
        return {
            "tcp": tcp,
            "obj": obj,
            "obj_cos": obj_cos,
            "obj_sin": obj_sin,
            "tcp_to_obj": tcp_to_obj,
            "goal": goal_arr,
            "pos_dist": pos_dist,
            "rot_err": rot_err,
            "tcp_obj_dist": tcp_obj_dist,
            "obj_vel_xy": obj_vel_xy,
            "obj_speed_xy": obj_speed_xy,
            "obj_ang_vel_z": obj_ang_vel_z,
            "tcp_vel_xy": tcp_vel_xy,
            "vel_align_to_goal": vel_align,
        }

    def pose_error(self, obs: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = self.extract_features(obs, goal)
        return features["pos_dist"], features["rot_err"]

    def phase_batch(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        features = self.extract_features(obs, goal)
        pos_dist = features["pos_dist"]
        rot_err = features["rot_err"]
        tcp_obj_dist = features["tcp_obj_dist"]
        phase = np.full(obs.shape[0], self.PHASE_CONTACT, dtype=np.int64)
        phase[pos_dist > self.pos_near_thresh] = self.PHASE_PUSH_TRANSLATE
        phase[rot_err > self.rot_large_thresh] = self.PHASE_PUSH_ROTATE
        phase[tcp_obj_dist > self.contact_thresh] = self.PHASE_APPROACH
        near_goal = np.logical_and(pos_dist <= self.pos_near_thresh, rot_err <= self.rot_near_thresh)
        phase[near_goal] = self.PHASE_NEAR_GOAL
        return phase

    def bucket_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        return self.full_context_hash_batch(obs, goal, h)

    def bucket_hash(self, obs: np.ndarray, goal: np.ndarray, h: int) -> int:
        return self.full_context_hash(obs, goal, h)

    def coarse_hash_batch(self, obs: np.ndarray, h: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32)
        dummy_goal = np.zeros((obs_arr.shape[0], 4), dtype=np.float32)
        dummy_goal[:, 2] = 1.0
        features = self.extract_features(obs_arr, dummy_goal)
        tcp_bin = self._bin_array(features["tcp_obj_dist"], self.tcp_obj_bins)
        theta = np.arctan2(features["obj_sin"], features["obj_cos"])
        orientation_bin = np.floor(np.clip((theta + np.pi) / (2.0 * np.pi), 0.0, 0.999999) * 8.0).astype(np.int64)
        speed_bin = self._bin_array(features["obj_speed_xy"], self.obj_speed_bins)
        h_bin = self._bin_array(h.astype(np.float32), tuple(float(v) for v in self.horizon_bins))
        return (tcp_bin * 10_000 + orientation_bin * 1_000 + speed_bin * 100 + h_bin).astype(np.int64)

    def coarse_hash(self, obs: np.ndarray, h: int) -> int:
        return int(self.coarse_hash_batch(obs[None, :], np.asarray([h], dtype=np.int64))[0])

    def full_context_hash_batch(self, obs: np.ndarray, goal: np.ndarray, h: np.ndarray) -> np.ndarray:
        features = self.extract_features(obs, goal)
        phase = self.phase_batch(obs, goal)
        tcp_bin = self._bin_array(features["tcp_obj_dist"], self.tcp_obj_bins)
        goal_bin = self._bin_array(features["pos_dist"], self.obj_goal_dist_bins)
        rot_bin = self._bin_array(features["rot_err"], self.rot_err_bins_rad)
        speed_bin = self._bin_array(features["obj_speed_xy"], self.obj_speed_bins)
        ang_bin = self._bin_array(np.abs(features["obj_ang_vel_z"]), self.ang_vel_bins)
        h_bin = self._bin_array(h.astype(np.float32), tuple(float(v) for v in self.horizon_bins))
        return (
            phase * 10_000_000
            + tcp_bin * 1_000_000
            + goal_bin * 100_000
            + rot_bin * 10_000
            + speed_bin * 1_000
            + ang_bin * 100
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
        out = np.zeros(action.shape[0], dtype=np.int64)
        for dim in range(bins.shape[1]):
            out = (out * 131 + bins[:, dim] + 1) % 2_147_483_647
        return out.astype(np.int64)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        center = 0.5 * (self.action_high + self.action_low)
        scale = np.maximum(1e-6, 0.5 * (self.action_high - self.action_low))
        return ((action - center) / scale).astype(np.float32)

    def diagnostic_metrics(self, obs: np.ndarray, goal: np.ndarray) -> Dict[str, float]:
        features = self.extract_features(obs, goal)
        phase = self.phase_batch(obs, goal)
        pos_dist = features["pos_dist"]
        rot_err = features["rot_err"]
        tcp_obj_dist = features["tcp_obj_dist"]
        obj_speed = features["obj_speed_xy"]
        ang_vel_abs = np.abs(features["obj_ang_vel_z"])
        vel_align = features["vel_align_to_goal"]
        metrics: Dict[str, float] = {
            "pushT/obj_goal_xy_dist_mean": _nanmean(pos_dist),
            "pushT/obj_goal_xy_dist_p50": _nanpercentile(pos_dist, 50.0),
            "pushT/obj_goal_xy_dist_p90": _nanpercentile(pos_dist, 90.0),
            "pushT/obj_goal_rot_err_mean": _nanmean(rot_err),
            "pushT/obj_goal_rot_err_p50": _nanpercentile(rot_err, 50.0),
            "pushT/obj_goal_rot_err_p90": _nanpercentile(rot_err, 90.0),
            "pushT/intersection_ratio_mean": np.nan,
            "pushT/intersection_ratio_p50": np.nan,
            "pushT/intersection_ratio_p90": np.nan,
            "pushT/success_margin_mean": np.nan,
            "pushT/tcp_obj_dist_mean": _nanmean(tcp_obj_dist),
            "pushT/contact_like_rate": float(np.mean(tcp_obj_dist <= self.contact_thresh)) if tcp_obj_dist.size else 0.0,
            "pushT/obj_speed_xy_mean": _nanmean(obj_speed),
            "pushT/obj_ang_vel_z_abs_mean": _nanmean(ang_vel_abs),
            "pushT/vel_align_to_goal_mean": _nanmean(vel_align),
        }
        for phase_id, phase_name in enumerate(self.phase_names):
            mask = phase == phase_id
            metrics[f"pushT/phase_{phase_name}_obj_goal_xy_dist_mean"] = _nanmean(pos_dist[mask])
            metrics[f"pushT/phase_{phase_name}_obj_goal_rot_err_mean"] = _nanmean(rot_err[mask])
            metrics[f"pushT/phase_{phase_name}_contact_like_rate"] = (
                float(np.mean(tcp_obj_dist[mask] <= self.contact_thresh)) if np.any(mask) else 0.0
            )
            metrics[f"pushT/phase_{phase_name}_intersection_ratio_mean"] = np.nan
        return metrics

    def metadata(self) -> Dict[str, object]:
        return {
            "task": "ManiSkillPushT",
            "env_id": "PushT-v1",
            "robot_uids": "panda_stick",
            "coarse_index_key": "tcp-object distance bin, object orientation bin, object speed bin, horizon bin",
            "full_context_key": (
                "phase, tcp-object distance bin, object-goal xy distance bin, "
                "object-goal rotation error bin, object speed bin, angular speed bin, horizon bin"
            ),
            "phase_labels": {idx: name for idx, name in enumerate(self.phase_names)},
            "phase_definition": (
                "near_goal if xy<=pos_near and rot<=rot_near; approach if tcp-object>contact; "
                "push_rotate if rot_err>rot_large; push_translate if xy>pos_near; else contact"
            ),
            "observation_layout": {
                "tcp_pos": list(self.tcp_slice),
                "obj_pos": list(self.obj_slice),
                "obj_cos_theta": [self.obj_cos_index, self.obj_cos_index + 1],
                "obj_sin_theta": [self.obj_sin_index, self.obj_sin_index + 1],
                "tcp_to_obj_pos": list(self.tcp_to_obj_slice),
                "obj_velocity_xy": list(self.obj_velocity_xy_slice),
                "obj_ang_vel_z": [self.obj_ang_vel_z_index, self.obj_ang_vel_z_index + 1],
                "tcp_velocity_xy": list(self.tcp_velocity_xy_slice),
            },
            "goal_layout": {
                "achieved_goal": "[obj_x, obj_y, cos(obj_theta), sin(obj_theta)]",
                "desired_goal": "[goal_x, goal_y, cos(goal_theta), sin(goal_theta)]",
            },
            "goal_dependent_coarse_key": False,
            "official_success": "ManiSkill PushT overlap success from info['success'] when available",
            "train_goal_success": (
                "pose approximation: xy distance<=pos_success_thresh and "
                "angle error<=rot_success_thresh"
            ),
            "pos_success_thresh": self.pos_success_thresh,
            "rot_success_thresh_rad": self.rot_success_thresh,
            "rot_success_thresh_deg": float(np.degrees(self.rot_success_thresh)),
            "contact_thresh": self.contact_thresh,
            "pos_near_thresh": self.pos_near_thresh,
            "rot_near_thresh": self.rot_near_thresh,
            "rot_large_thresh": self.rot_large_thresh,
            "speed_thresh": self.speed_thresh,
            "ang_vel_thresh": self.ang_vel_thresh,
            "tcp_obj_bins": list(self.tcp_obj_bins),
            "obj_goal_dist_bins": list(self.obj_goal_dist_bins),
            "rot_err_bins_rad": list(self.rot_err_bins_rad),
            "obj_speed_bins": list(self.obj_speed_bins),
            "ang_vel_bins": list(self.ang_vel_bins),
            "horizon_bins": list(self.horizon_bins),
            "invariants": {
                "rollout_intervention": False,
                "collection_time_weight_reuse": False,
                "coarse_key_goal_independent": True,
                "b4_candidates": "same-context replay-executed actions",
                "b2_candidates": "unrestricted policy perturbation actions",
            },
        }

    @staticmethod
    def angle_error_from_cossin(
        obj_cos: np.ndarray,
        obj_sin: np.ndarray,
        goal_cos: np.ndarray,
        goal_sin: np.ndarray,
    ) -> np.ndarray:
        cross = obj_sin * goal_cos - obj_cos * goal_sin
        dot = obj_cos * goal_cos + obj_sin * goal_sin
        return np.asarray(np.abs(np.arctan2(cross, dot)), dtype=np.float32)

    @staticmethod
    def _bin_array(values: np.ndarray, boundaries: Tuple[float, ...]) -> np.ndarray:
        out = np.zeros(values.shape[0], dtype=np.int64)
        for boundary in boundaries:
            out += values > float(boundary)
        return out


def _nanmean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _nanpercentile(values: np.ndarray, percentile: float) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanpercentile(arr, percentile))

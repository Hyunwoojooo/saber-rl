from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def resolve_fetch_push_env_id(reward_type: str) -> str:
    rt = reward_type.lower()
    if rt == "sparse":
        return "FetchPush-v4"
    if rt == "dense":
        return "FetchPushDense-v4"
    raise ValueError(f"Unsupported reward_type={reward_type!r}")


def resolve_fetch_pick_and_place_env_id(reward_type: str) -> str:
    rt = reward_type.lower()
    if rt == "sparse":
        return "FetchPickAndPlace-v4"
    if rt == "dense":
        return "FetchPickAndPlaceDense-v4"
    raise ValueError(f"Unsupported reward_type={reward_type!r}")


def resolve_fetch_slide_env_id(reward_type: str) -> str:
    rt = reward_type.lower()
    if rt == "sparse":
        return "FetchSlide-v4"
    if rt == "dense":
        return "FetchSlideDense-v4"
    raise ValueError(f"Unsupported reward_type={reward_type!r}")


def resolve_env_id(task_name: str, reward_type: str) -> str:
    task = task_name.lower()
    if task == "fetch_push":
        return resolve_fetch_push_env_id(reward_type)
    if task == "fetch_pick_and_place":
        return resolve_fetch_pick_and_place_env_id(reward_type)
    if task == "fetch_slide":
        return resolve_fetch_slide_env_id(reward_type)
    if task == "unitree_apple":
        return "UnitreeG1PlaceAppleInBowl-v1"
    if task == "pick_cube":
        return "PickCube-v1"
    if task == "push_t":
        return "PushT-v1"
    if task == "drawer_open":
        return "Meta-World/MT1"
    raise ValueError(f"Unsupported task_name={task_name!r}")


@dataclass
class SeedConfig:
    env_seed: int = 1
    torch_seed: int = 1
    numpy_seed: int = 1
    action_noise_seed: int = 1
    weight_seed: int = 31
    assignment_seed: int = 30_001
    eval_seed: int = 10001

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class LiteConfig:
    run_name: str = "fetch_push_lite_v0"
    output_root: str = "version_v0_lite/results"
    resume: bool = False
    resume_path: Optional[str] = None

    # B0..B4 from the specification.
    task_name: str = "fetch_push"
    method: str = "b4_lite"
    reward_type: str = "sparse"
    env_id: str = "FetchPush-v4"
    metaworld_env_name: str = "drawer-open-v3"
    metaworld_reward_function_version: str = "v2"
    obs_mode: str = "state"
    robot_uids: str = "panda"
    control_mode: Optional[str] = None
    num_envs: int = 1
    backend: Optional[str] = None
    sim_backend: Optional[str] = None
    max_episode_steps: int = 50
    render: bool = False

    total_steps: int = 200_000
    start_steps: int = 10_000
    warmup_steps: int = 5_000
    updates_per_step: int = 1
    batch_size: int = 256
    buffer_capacity: int = 300_000
    her_ratio: float = 0.8

    gamma: float = 0.98
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_dim: int = 256
    noise_std: float = 0.2
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    l2_action_penalty: float = 0.0

    eval_freq: int = 10_000
    eval_episodes: int = 20
    checkpoint_every_steps: int = 50_000
    save_step_checkpoints: bool = True
    log_every_steps: int = 1_000
    smoke: bool = False
    validate_steps: int = 200
    validate_random_policy_steps: int = 200

    # MetaCog-Lite v0 local replay-action support.
    M_B: int = 32
    M_Q_min: int = 8
    N_local_min: int = 8
    local_search_max: int = 8_192
    action_bins_per_dim: int = 5
    policy_support_mode: str = "bin"
    policy_action_radius: float = 0.35
    support_query_mode: str = "current_policy"
    support_assignment_mode: str = "support"
    assignment_experiment: bool = False
    assignment_rule: str = "support"
    assignment_log_enabled: bool = True
    assignment_diagnostic_window_updates: int = 1_000
    support_scan_mode: str = "deterministic_tail_v1"
    support_cache_enabled: bool = True
    support_cache_max_entries: int = 200_000
    local_scan_cache_enabled: bool = True
    local_scan_cache_max_entries: int = 50_000
    support_short_circuit: bool = True
    numba_support_scan: bool = True
    support_predicate_early_exit: bool = True
    support_diagnostic_audit_interval: int = 100
    verify_runtime_optimization: bool = False

    # FetchPush bucket.
    d_contact: float = 0.08
    d_reach: float = 0.10
    d_goal_near: float = 0.05
    ee_obj_bins: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    obj_goal_bins: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    horizon_bins: tuple[int, ...] = (10, 25, 50, 100)

    # UnitreeApple bucket.
    hand_apple_bins: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    apple_goal_bins: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    apple_z_bins: tuple[float, ...] = (0.75, 0.85, 1.00)

    # ManiSkill PickCube bucket.
    pick_cube_z_lift_min: Optional[float] = None
    pick_cube_tcp_cube_bins: tuple[float, ...] = (0.03, 0.05, 0.08, 0.12, 0.20, 0.40)
    pick_cube_goal_bins: tuple[float, ...] = (0.025, 0.05, 0.10, 0.20, 0.40, 0.70)
    pick_cube_z_bins: tuple[float, ...] = (0.04, 0.07, 0.10, 0.15, 0.25, 0.40)

    # ManiSkill PushT bucket.
    push_t_pos_success_thresh: float = 0.05
    push_t_rot_success_thresh: float = 0.2617993877991494
    push_t_rot_weight: float = 0.25
    push_t_contact_thresh: float = 0.06
    push_t_pos_near_thresh: float = 0.05
    push_t_rot_near_thresh: float = 0.2617993877991494
    push_t_rot_large_thresh: float = 0.5235987755982988
    push_t_speed_thresh: float = 0.03
    push_t_ang_vel_thresh: float = 0.10
    push_t_tcp_obj_bins: tuple[float, ...] = (0.03, 0.06, 0.10, 0.20, 0.40)
    push_t_obj_goal_dist_bins: tuple[float, ...] = (0.03, 0.05, 0.10, 0.20, 0.40, 0.70)
    push_t_rot_err_bins_rad: tuple[float, ...] = (0.15, 0.30, 0.52, 0.78, 1.57, 3.14)
    push_t_obj_speed_bins: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10, 0.20)
    push_t_ang_vel_bins: tuple[float, ...] = (0.05, 0.10, 0.20, 0.50, 1.00)

    # FetchPickAndPlace bucket.
    z_lift_min: float = 0.48
    grip_close_thresh: float = 0.04
    pick_hand_obj_bins: tuple[float, ...] = (0.03, 0.05, 0.08, 0.12, 0.20, 0.40)
    pick_obj_goal_bins: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.70)
    pick_object_z_bins: tuple[float, ...] = (0.43, 0.48, 0.55, 0.65, 0.75)
    pick_gripper_gap_bins: tuple[float, ...] = (0.02, 0.05, 0.08)

    # FetchSlide bucket.
    slide_d_contact: float = 0.08
    slide_d_goal_near: float = 0.05
    slide_v_slide_min: float = 0.05
    slide_v_fast: float = 0.15
    slide_ee_obj_bins: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    slide_obj_goal_bins: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.70, 1.00)
    slide_obj_speed_bins: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.40)
    slide_vel_align_bins: tuple[float, ...] = (-0.50, 0.0, 0.50, 0.80)

    # MetaWorld drawer-open-v3 bucket.
    drawer_success_threshold: float = 0.05
    drawer_d_reach: float = 0.08
    drawer_d_goal_near: float = 0.05
    drawer_d_opening_mid: float = 0.12
    drawer_hand_handle_bins: tuple[float, ...] = (0.03, 0.05, 0.08, 0.12, 0.20, 0.40)
    drawer_handle_goal_bins: tuple[float, ...] = (0.03, 0.05, 0.08, 0.12, 0.20, 0.40, 0.70)
    drawer_handle_y_bins: tuple[float, ...] = (-0.05, 0.00, 0.05, 0.10, 0.15, 0.25)
    drawer_gripper_bins: tuple[float, ...] = (-0.50, 0.0, 0.50)
    drawer_horizon_bins: tuple[int, ...] = (10, 25, 50, 100, 200, 500)

    # Gap normalization and weighting.
    N_gap_min: int = 256
    gap_window: int = 50_000
    gap_q_low: float = 0.60
    gap_q_high: float = 0.95
    eps: float = 1e-8
    eps_G: float = 1e-6
    lambda_P: float = 1.0
    lambda_support: float = 1.0
    random_fixed_count_k: int = 0
    U_P_max: float = 1.0

    # SABER document baselines: critic-only TD-error and PER-style loss weighting.
    lambda_TD: float = 1.0
    U_TD_max: float = 1.0
    per_alpha: float = 0.6
    eps_PER: float = 1e-6
    U_PER_max: float = 2.0

    weighting_mode: str = "batch_norm"
    weight_scope: str = "actor_critic"
    unrestricted_candidate_std: float = 0.5

    # Terminal convention is part of the experiment spec.
    success_terminates: bool = False
    timeout_bootstrap: bool = True

    device: str = "auto"
    seeds: SeedConfig = field(default_factory=SeedConfig)

    def normalize(self) -> None:
        if self.assignment_experiment:
            if self.resume or self.resume_path:
                raise ValueError("Assignment experiment permits fresh runs only")
            if self.assignment_rule not in {"support", "random", "td_topk"}:
                raise ValueError("Unknown assignment rule")
            required = {
                "method": "b3_support_runtime_opt", "task_name": "drawer_open",
                "weight_scope": "critic_only", "weighting_mode": "batch_norm",
                "support_assignment_mode": "support", "lambda_support": 1.0,
            }
            for key, expected in required.items():
                if getattr(self, key) != expected:
                    raise ValueError(f"Assignment experiment requires {key}={expected}")
        self.task_name = self.task_name.lower()
        if self.task_name not in {
            "fetch_push",
            "fetch_pick_and_place",
            "fetch_slide",
            "unitree_apple",
            "pick_cube",
            "push_t",
            "drawer_open",
        }:
            raise ValueError(
                "task_name must be fetch_push|fetch_pick_and_place|fetch_slide|unitree_apple|pick_cube|push_t|drawer_open"
            )
        self.method = self.method.lower()
        allowed = {
            "b0_ddpg_her",
            "b1_uniform_scale",
            "b2_unrestricted_gap",
            "b3_support_only",
            "b3_support_random",
            "b3_support_random_runtime_opt",
            "b3_support_shuffled",
            "b3_support_shuffled_runtime_opt",
            "b3_support_runtime_opt",
            "b3_fully_random_fixed_count",
            "b4_lite",
            "td_error",
            "per_style_loss_weighting",
        }
        if self.method not in allowed:
            raise ValueError(f"Unsupported method={self.method!r}, expected one of {sorted(allowed)}")
        self.reward_type = self.reward_type.lower()
        if self.reward_type not in {"sparse", "dense"}:
            raise ValueError("reward_type must be sparse|dense")
        self.env_id = resolve_env_id(self.task_name, self.reward_type)
        self.obs_mode = self.obs_mode.lower()
        self.weighting_mode = self.weighting_mode.lower()
        if self.weighting_mode not in {"batch_norm", "raw"}:
            raise ValueError("weighting_mode must be batch_norm|raw")
        self.weight_scope = self.weight_scope.lower()
        if self.weight_scope not in {"actor_critic", "critic_only", "actor_only", "none"}:
            raise ValueError("weight_scope must be actor_critic|critic_only|actor_only|none")
        self.policy_support_mode = self.policy_support_mode.lower()
        if self.policy_support_mode not in {"bin", "radius", "auto"}:
            raise ValueError("policy_support_mode must be bin|radius|auto")
        self.support_query_mode = self.support_query_mode.lower()
        if self.support_query_mode != "current_policy":
            raise ValueError(
                "DrawerOpen supports support_query_mode=current_policy only"
            )
        self.support_assignment_mode = self.support_assignment_mode.lower()
        if self.support_assignment_mode not in {
            "support",
            "context_eligible_count_matched",
        }:
            raise ValueError(
                "support_assignment_mode must be "
                "support|context_eligible_count_matched"
            )
        self.support_scan_mode = self.support_scan_mode.lower()
        if self.support_scan_mode not in {"deterministic_tail_v1", "deterministic_head_v1", "deterministic_stride_v1"}:
            raise ValueError(
                "support_scan_mode must be deterministic_tail_v1|deterministic_head_v1|deterministic_stride_v1"
            )
        if self.task_name == "unitree_apple":
            if self.max_episode_steps == 50:
                self.max_episode_steps = 100
            if self.total_steps == 200_000:
                self.total_steps = 100_000
            if self.batch_size == 256:
                self.batch_size = 128
            if self.M_B == 32:
                self.M_B = 16
            if self.M_Q_min == 8:
                self.M_Q_min = 4
            if self.N_local_min == 8:
                self.N_local_min = 4
            if self.local_search_max == 8_192:
                self.local_search_max = 1_024
            if self.N_gap_min == 256:
                self.N_gap_min = 128
            if self.policy_support_mode == "bin":
                self.policy_support_mode = "radius"
        if self.task_name == "fetch_pick_and_place":
            if self.M_B == 32:
                self.M_B = 16
            if self.M_Q_min == 8:
                self.M_Q_min = 4
            if self.N_local_min == 8:
                self.N_local_min = 4
            if self.local_search_max == 8_192:
                self.local_search_max = 1_024
            if self.N_gap_min == 256:
                self.N_gap_min = 128
        if self.task_name == "fetch_slide":
            if self.M_B == 32:
                self.M_B = 16
            if self.M_Q_min == 8:
                self.M_Q_min = 4
            if self.N_local_min == 8:
                self.N_local_min = 4
            if self.local_search_max == 8_192:
                self.local_search_max = 1_024
            if self.N_gap_min == 256:
                self.N_gap_min = 128
            if self.slide_d_goal_near == 0.05:
                self.slide_d_goal_near = self.d_goal_near
        if self.task_name == "pick_cube":
            if self.total_steps == 200_000:
                self.total_steps = 100_000
            if self.M_B == 32:
                self.M_B = 16
            if self.M_Q_min == 8:
                self.M_Q_min = 4
            if self.N_local_min == 8:
                self.N_local_min = 4
            if self.local_search_max == 8_192:
                self.local_search_max = 1_024
            if self.N_gap_min == 256:
                self.N_gap_min = 128
            if self.d_goal_near == 0.05:
                self.d_goal_near = 0.025
            if self.policy_support_mode == "bin":
                self.policy_support_mode = "auto"
        if self.task_name == "push_t":
            if self.max_episode_steps == 50:
                self.max_episode_steps = 100
            if self.robot_uids == "panda":
                self.robot_uids = "panda_stick"
            if self.total_steps == 200_000:
                self.total_steps = 100_000
            if self.M_B == 32:
                self.M_B = 16
            if self.M_Q_min == 8:
                self.M_Q_min = 4
            if self.N_local_min == 8:
                self.N_local_min = 4
            if self.local_search_max == 8_192:
                self.local_search_max = 1_024
            if self.N_gap_min == 256:
                self.N_gap_min = 128
        if self.task_name == "drawer_open":
            if self.max_episode_steps == 50:
                self.max_episode_steps = 200
            if self.M_B == 32:
                self.M_B = 16
            if self.M_Q_min == 8:
                self.M_Q_min = 4
            if self.N_local_min == 8:
                self.N_local_min = 4
            if self.local_search_max == 8_192:
                self.local_search_max = 512
            if self.N_gap_min == 256:
                self.N_gap_min = 128
            if self.metaworld_env_name == "":
                self.metaworld_env_name = "drawer-open-v3"
        self.num_envs = max(1, int(self.num_envs))
        self.her_ratio = max(0.0, min(1.0, float(self.her_ratio)))
        self.batch_size = max(1, int(self.batch_size))
        self.M_B = max(1, int(self.M_B))
        self.M_Q_min = max(1, int(self.M_Q_min))
        self.N_local_min = max(1, int(self.N_local_min))
        self.local_search_max = max(0, int(self.local_search_max))
        self.action_bins_per_dim = max(2, int(self.action_bins_per_dim))
        self.policy_delay = max(1, int(self.policy_delay))
        self.target_policy_noise = max(0.0, float(self.target_policy_noise))
        self.target_noise_clip = max(0.0, float(self.target_noise_clip))
        self.support_cache_max_entries = max(0, int(self.support_cache_max_entries))
        self.local_scan_cache_max_entries = max(0, int(self.local_scan_cache_max_entries))
        self.assignment_diagnostic_window_updates = max(
            1, int(self.assignment_diagnostic_window_updates)
        )
        self.numba_support_scan = bool(self.numba_support_scan)
        self.support_predicate_early_exit = bool(
            self.support_predicate_early_exit
        )
        self.support_diagnostic_audit_interval = max(
            0, int(self.support_diagnostic_audit_interval)
        )
        self.N_gap_min = max(1, int(self.N_gap_min))
        self.random_fixed_count_k = max(0, int(self.random_fixed_count_k))
        self.gap_window = max(self.N_gap_min, int(self.gap_window))
        strict_control = (
            self.support_assignment_mode
            == "context_eligible_count_matched"
        )
        if strict_control:
            required = {
                "task_name": (self.task_name, "drawer_open"),
                "method": (self.method, "b3_support_runtime_opt"),
                "weight_scope": (self.weight_scope, "critic_only"),
                "weighting_mode": (self.weighting_mode, "batch_norm"),
                "policy_support_mode": (self.policy_support_mode, "bin"),
                "M_B": (self.M_B, 16),
                "M_Q_min": (self.M_Q_min, 4),
                "N_local_min": (self.N_local_min, 4),
                "local_search_max": (self.local_search_max, 512),
                "action_bins_per_dim": (
                    self.action_bins_per_dim,
                    5,
                ),
                "lambda_support": (
                    float(self.lambda_support),
                    1.0,
                ),
            }
            mismatched = {
                name: {"actual": actual, "expected": expected}
                for name, (actual, expected) in required.items()
                if actual != expected
            }
            if mismatched:
                raise ValueError(
                    "strict context-eligible count-matched control must "
                    "match DrawerOpen SABER Main: "
                    f"{mismatched}"
                )
        if self.smoke:
            if self.task_name in {"unitree_apple", "pick_cube", "push_t", "drawer_open"}:
                self.total_steps = min(self.total_steps, 5_000)
                self.start_steps = min(self.start_steps, 1_000)
                self.warmup_steps = min(self.warmup_steps, 1_000)
                self.batch_size = min(self.batch_size, 128)
                self.buffer_capacity = min(self.buffer_capacity, 20_000)
                self.eval_freq = min(self.eval_freq, 1_000)
                self.eval_episodes = min(self.eval_episodes, 5)
                self.checkpoint_every_steps = min(self.checkpoint_every_steps, 1_000)
                self.M_B = min(self.M_B, 16)
                self.M_Q_min = min(self.M_Q_min, 4)
                self.N_local_min = min(self.N_local_min, 4)
                self.local_search_max = min(self.local_search_max, 512)
                self.N_gap_min = min(self.N_gap_min, 32)
                self.gap_window = min(self.gap_window, 2_000)
            else:
                self.total_steps = 1_000
                self.start_steps = 200
                self.warmup_steps = 200
                self.batch_size = min(self.batch_size, 64)
                self.buffer_capacity = min(self.buffer_capacity, 20_000)
                self.eval_freq = 500
                self.eval_episodes = 5
                self.checkpoint_every_steps = 500
                self.local_search_max = min(self.local_search_max, 1_024)
                self.N_gap_min = min(self.N_gap_min, 32)
                self.gap_window = min(self.gap_window, 2_000)

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["resolved_env_id"] = self.env_id
        return payload

    def output_root_path(self, repo_root: Path) -> Path:
        path = Path(self.output_root)
        return path if path.is_absolute() else repo_root / path

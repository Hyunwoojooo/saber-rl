"""Vendored verbatim from ~/ICML2027/draweropen_ddpg_policy_query/saber_qdiag/simulator.py
(sha256 6e1db8e6fbdffbf4408501c93e24a6efa76eba39fb6b99c651ba129a43968bca); only the relative import of
require/finite/tree_hash is replaced by the copies below (from saber_qdiag/common.py,
sha256 cb1c6f1d62cf3602761235b09051b631ec0663feaf85d741fbccef14881c1010). Do not edit; re-vendor instead.
"""
import hashlib as _hashlib
def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    ok = torch.isfinite(value).all().item() if torch.is_tensor(value) else np.isfinite(value).all()
    require(bool(ok), f"Non-finite {name}; no replacement or failed-sample deletion is allowed")


def tree_hash(value):
    h = _hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            visit(x.detach().cpu().contiguous().numpy())
        elif isinstance(x, np.ndarray):
            require(not x.dtype.hasobject, "Object arrays cannot be hashed as numeric state")
            h.update(str(x.dtype).encode()); h.update(str(x.shape).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode()); visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(str(len(x)).encode())
            for v in x: visit(v)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


# Audited MetaWorld 3.1.1 snapshot adapter, with no rendering or auto-reset.


import copy
from collections import deque
import importlib.metadata
import numpy as np
import torch
import gymnasium as gym
import metaworld
import mujoco



MODEL_MUTABLE = ("body_pos", "body_quat", "site_pos", "site_quat", "eq_data")
SKIP = {"env", "model", "data", "mujoco_renderer", "spec", "_cached_spec"}
WRAPPERS = {"TimeLimit", "OrderEnforcing", "PassiveEnvChecker", "CheckpointWrapper",
            "RandomTaskSelectWrapper", "RecordEpisodeStatistics", "AutoTerminateOnSuccessWrapper"}


def plain(value):
    if value is None or isinstance(value, (str, bytes, bool, int, float, np.generic, np.ndarray, np.random.Generator, gym.Space)):
        return True
    if isinstance(value, (list, tuple, deque)): return all(plain(v) for v in value)
    if isinstance(value, dict): return all(plain(k) and plain(v) for k, v in value.items())
    return False


class DrawerSimulator:
    def __init__(self, config, seed):
        self.config = config
        require(importlib.metadata.version("metaworld") == "3.1.1", "Snapshot adapter audited for MetaWorld 3.1.1 only")
        require(importlib.metadata.version("mujoco") == "3.3.0", "Re-audit state restoration before changing MuJoCo 3.3.0")
        require(importlib.metadata.version("gymnasium") == "1.3.0", "Wrapper adapter audited for Gymnasium 1.3.0 only")
        gym.register_envs(metaworld)
        # MetaWorld task generation can touch legacy global NumPy RNG. Keep it isolated.
        global_rng = np.random.get_state()
        try:
            self.env = gym.make(config.env_id, env_name=config.metaworld_env_name, seed=int(seed),
                                max_episode_steps=config.max_episode_steps, terminate_on_success=False,
                                reward_function_version=config.metaworld_reward_function_version)
        finally:
            np.random.set_state(global_rng)
        self.chain = []
        cur = self.env
        while True:
            self.chain.append(cur)
            if not isinstance(cur, gym.Wrapper): break
            require(type(cur).__name__ in WRAPPERS, f"Unaudited wrapper: {type(cur).__name__}")
            cur = cur.env
        self.base = cur
        require(type(cur).__name__ == "SawyerDrawerOpenEnvV3", "Unexpected unwrapped task")
        require(np.array_equal(self.env.action_space.low, -np.ones(4)) and
                np.array_equal(self.env.action_space.high, np.ones(4)), "Unexpected action bounds")
        self._normal_wrapper_limits = [(i, int(obj._max_episode_steps)) for i, obj in enumerate(self.chain)
                                       if isinstance(obj, gym.wrappers.TimeLimit)]
        self._normal_base_limit = int(self.base.max_path_length)
        self.base.seed(int(seed))
        self.reset(seed)
        self.model_fixed_hash = self._fixed_model_hash()

    def _fixed_model_hash(self):
        arrays = {name: getattr(self.base.model, name) for name in dir(self.base.model)
                  if name not in MODEL_MUTABLE and isinstance(getattr(self.base.model, name, None), np.ndarray)}
        options = {name: getattr(self.base.model.opt, name) for name in dir(self.base.model.opt)
                   if not name.startswith('_') and isinstance(getattr(self.base.model.opt, name, None),
                   (int, float, bool, np.generic, np.ndarray))}
        return tree_hash({"arrays": arrays, "physics_options": options})

    def reset(self, seed):
        # MetaWorld 3.1.1's base reset ignores seed=; seed its own RNG before task selection.
        # A diagnostic continuation must never lengthen a subsequently collected episode.
        for i, limit in self._normal_wrapper_limits: self.chain[i]._max_episode_steps = limit
        self.base.max_path_length = self._normal_base_limit
        self.base.seed(int(seed))
        obs, _ = self.env.reset(seed=int(seed))
        self.obs = np.asarray(obs, dtype=np.float32).copy()
        self.goal = np.asarray(self.base._target_pos, dtype=np.float32).copy()
        self._check()
        return self.state()

    def state(self):
        return np.concatenate((self.obs, self.goal)).astype(np.float32)

    def _check(self):
        require(self.obs.shape == (39,) and self.goal.shape == (3,), "Observation/goal schema changed")
        for name, value in (("observation", self.obs), ("goal", self.goal), ("qpos", self.base.data.qpos), ("qvel", self.base.data.qvel)):
            finite(value, name)
        require(not self.base._did_see_sim_exception, "MetaWorld reported a simulation exception")

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        finite(a, "action"); require(a.shape == (4,) and np.all(np.abs(a) <= 1.000001), "Invalid policy action")
        obs, _, terminated, truncated, info = self.env.step(a)
        self.obs = np.asarray(obs, dtype=np.float32).copy()
        goal = np.asarray(self.base._target_pos, dtype=np.float32)
        require(np.array_equal(goal, self.goal), "Original goal changed during reference rollout")
        self._check()
        goal_success = bool(np.linalg.norm(self.obs[4:7] - self.goal) <= self.config.drawer_success_threshold)
        return {"reward": 0.0 if goal_success else -1.0, "terminated": bool(terminated),
                "truncated": bool(truncated), "goal_success": goal_success,
                "env_success": bool(info.get("success", False))}

    def snapshot(self):
        require(self._fixed_model_hash() == self.model_fixed_hash, "Uncaptured model property changed; extend audited adapter")
        python_state = []
        for obj in self.chain:
            state = {}
            for name, value in vars(obj).items():
                if name in SKIP: continue
                require(plain(value), f"Uncaptured Python state: {type(obj).__name__}.{name} ({type(value).__name__})")
                state[name] = copy.deepcopy(value)
            python_state.append(state)
        python_state[-1]["max_path_length"] = int(self.base.max_path_length)
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        integration = np.empty(mujoco.mj_stateSize(self.base.model, kind), dtype=np.float64)
        mujoco.mj_getState(self.base.model, self.base.data, integration, kind)
        return {"schema": 1, "classes": [type(x).__name__ for x in self.chain], "python": python_state,
                "model": {name: getattr(self.base.model, name).copy() for name in MODEL_MUTABLE},
                "fixed_model_hash": self.model_fixed_hash, "integration": integration,
                "obs": self.obs.copy(), "goal": self.goal.copy()}

    def restore(self, snapshot):
        require(snapshot["schema"] == 1 and snapshot["classes"] == [type(x).__name__ for x in self.chain], "Snapshot wrapper schema differs")
        require(self._fixed_model_hash() == self.model_fixed_hash, "Uncaptured model property changed before restore")
        require(snapshot["fixed_model_hash"] == self.model_fixed_hash, "Different immutable model/asset state")
        for obj, state in zip(self.chain, snapshot["python"]):
            for name in list(vars(obj)):
                if name not in SKIP and name not in state: delattr(obj, name)
            for name, value in state.items(): setattr(obj, name, copy.deepcopy(value))
        for name, value in snapshot["model"].items(): np.copyto(getattr(self.base.model, name), value)
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(self.base.model, self.base.data, snapshot["integration"], kind)
        mujoco.mj_forward(self.base.model, self.base.data)
        # forward recomputes derived positions but also changes solver warmstart; restore integration again.
        mujoco.mj_setState(self.base.model, self.base.data, snapshot["integration"], kind)
        self.obs = snapshot["obs"].copy(); self.goal = snapshot["goal"].copy()
        self._check()

    def extend_time_limit(self, steps):
        # Change only artificial termination counters; no reset, physics or reward changes.
        for obj in self.chain:
            if isinstance(obj, gym.wrappers.TimeLimit):
                obj._max_episode_steps = max(obj._max_episode_steps, obj._elapsed_steps + int(steps) + 1)
        self.base.max_path_length = max(self.base.max_path_length, self.base.curr_path_length + int(steps) + 1)

    def physical(self):
        return np.concatenate((self.base.data.qpos.copy(), self.base.data.qvel.copy()))

    def reseed_future_rng(self, seed):
        """Hold state/goal fixed while giving existing simulator RNGs different futures."""
        visited = set()
        for obj in self.chain:
            for value in vars(obj).values():
                if isinstance(value, np.random.Generator) and id(value) not in visited:
                    visited.add(id(value))
                    value.bit_generator.state = type(value.bit_generator)(int(seed)+len(visited)).state
                elif isinstance(value, gym.Space) and id(value) not in visited:
                    visited.add(id(value)); value.seed(int(seed)+len(visited))

    def close(self): self.env.close()


@torch.no_grad()
def policy_action(actor, state):
    device = next(actor.parameters()).device
    action = actor(torch.as_tensor(state[None], dtype=torch.float32, device=device))[0].cpu().numpy()
    finite(action, "policy action")
    return action.copy()


def compare_trajectory(left, right, actor, steps, protocol, allow_timeout_difference=False):
    obs_error, physics_error = 0.0, 0.0
    count = 0
    for _ in range(steps):
        action = policy_action(actor, left.state())
        a, b = left.step(action), right.step(action)
        obs_error = max(obs_error, float(np.max(np.abs(left.state() - right.state()))))
        physics_error = max(physics_error, float(np.max(np.abs(left.physical() - right.physical()))))
        for name in ("reward", "goal_success", "env_success", "terminated"):
            require(a[name] == b[name], f"Restoration mismatch: {name}")
        if not allow_timeout_difference: require(a["truncated"] == b["truncated"], "Truncation mismatch")
        count += 1
        if a["terminated"] or a["truncated"]: break
    require(obs_error <= protocol.observation_atol and physics_error <= protocol.physics_atol,
            f"Restoration failed: observation={obs_error}, physics={physics_error}")
    return {"steps": count, "max_observation_error": obs_error, "max_physics_error": physics_error}


def rollout(sim, snapshot, actor, length, gamma, future_seed=None, expected_state=None, expected_action=None):
    sim.restore(snapshot)
    if expected_state is not None:
        require(np.array_equal(sim.state(), expected_state), "Reference query state differs from stored evaluation state")
    first_action = policy_action(actor, sim.state())
    if expected_action is not None:
        require(np.array_equal(first_action, expected_action), "Reference first action differs from critic query action")
    if future_seed is not None: sim.reseed_future_rng(future_seed)
    sim.extend_time_limit(length)
    total, discount, actual_steps, ended = 0.0, 1.0, 0, False
    zero_rewards = 0
    for index in range(length):
        info = sim.step(first_action if index == 0 else policy_action(actor, sim.state()))
        require(not info["truncated"], "Unexpected timeout during continuing-value reference")
        require(info["reward"] in {-1.0, 0.0}, "Return tail bound requires the declared sparse reward range")
        zero_rewards += int(info["reward"] == 0.0)
        total += discount * info["reward"]
        discount *= gamma; actual_steps += 1
        if info["terminated"]:
            ended = True; break
    finite(total, "rollout return")
    return {"return": total, "steps": actual_steps, "terminated": ended,
            "tail_bound": 0.0 if ended else discount / (1-gamma), "zero_reward_steps": zero_rewards,
            "negative_reward_steps": actual_steps-zero_rewards}

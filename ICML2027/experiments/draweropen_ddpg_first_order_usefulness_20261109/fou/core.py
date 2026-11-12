"""First-order usefulness of SABER critic weights on frozen checkpoints.

Two families share one code path: "drawer_ddpg" (DrawerOpen, one critic) and "fpp_td3"
(FetchPickAndPlace, twin critics with target smoothing). The critic head(s) are wrapped in
`Heads`; everything that concerns the policy query (risk R, kappa) uses head 0, the critic
the actor reads.

Everything here is read-only with respect to the checkpoint: actor, targets, critic, Adam
state and replay are copied, used, and verified unchanged by hash at the end.

Sign convention used throughout: for an arm X compared with uniform (HER) weights,
    gain_X = R(after uniform update) - R(after X update)
so gain_X > 0 means arm X reduced the policy-query error R more than uniform did.
"""
from __future__ import annotations

import copy
import csv
import importlib
import importlib.util
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

ROOT = Path(__file__).resolve().parents[1]
A1 = ROOT.parent / "draweropen_ddpg_actor_dynamics_20261031"
A1_SRC = A1 / "code" / "src"
if str(A1_SRC) not in sys.path:
    sys.path.insert(0, str(A1_SRC))

from metacog_lite_v0.agent import DDPGAgent  # noqa: E402
from metacog_lite_v0.config import LiteConfig, SeedConfig  # noqa: E402
from metacog_lite_v0.replay import HERReplayBuffer  # noqa: E402
from metacog_lite_v0.trainer import make_task_spec  # noqa: E402
from metacog_lite_v0.weights import GapStats, LiteWeightComputer  # noqa: E402

from .vendor_sim import DrawerSimulator, finite, require, tree_hash  # noqa: E402

CLIP = 10.0                      # production critic clip (agent.py)
ARMS = ("saber", "inverse", "random", "td_topk")
N_RANDOM_DRAWS = 16              # first-order / one-step Random expectation
N_RANDOM_SEQUENCES = 4           # K-step Random branches
N_TARGET_NOISE = 16              # TD3 only: smoothing-noise draws averaged in the policy-query target
TARGET_NOISE_SEED = 940_000


class Heads(torch.nn.Module):
    """Critic head(s) in optimizer order; forward returns [H, B]."""

    def __init__(self, critics):
        super().__init__()
        self.c = torch.nn.ModuleList(critics)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.stack([c(s, a).reshape(-1) for c in self.c])


def load_package(pkg_dir: Path, alias: str):
    """Import a second copy of the training package under its own name (all its imports are
    relative), so the FPP code can live next to the DrawerOpen code in one process."""
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, pkg_dir / "__init__.py",
                                                      submodule_search_locations=[str(pkg_dir)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
    return lambda name: importlib.import_module(f"{alias}.{name}")


# ----------------------------------------------------------------------------- source
@dataclass
class Source:
    path: Path
    step: int
    arm: str
    seed: int
    cfg: LiteConfig
    agent: DDPGAgent
    replay: HERReplayBuffer
    task: object
    weight_computer: LiteWeightComputer
    frozen_hash: str
    kind: str = ""
    last_eval_success: float = float("nan")      # 20-episode evaluation at (or just before) this step
    eval_tail_mean: float = float("nan")         # mean of the last 10 evaluations
    total_updates: int = -1
    family: str = "drawer_ddpg"
    heads: torch.nn.Module | None = None         # critic head(s), parameters in Adam order
    obs_dim: int = 39
    success_threshold: float = 0.05
    her_inclusive: bool = False                  # future-goal HER admits the transition's own next achieved goal
    env: object | None = None                    # fpp_td3: the training package's FetchGoalEnv

    def state_hash(self) -> str:
        a = self.agent
        nets = {k: getattr(a, k).state_dict() for k in ("actor", "actor_target", "critic", "critic_target",
                                                        "critic1", "critic2", "critic1_target", "critic2_target")
                if hasattr(a, k)}
        return tree_hash({**nets, "critic_opt": a.critic_opt.state_dict(), "replay": self.replay.state_dict()})

    def verify_unchanged(self) -> None:
        require(self.state_hash() == self.frozen_hash, f"Source state mutated: {self.path}")


def load_source(path: Path) -> Source:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = payload["config"]
    known = {f.name for f in fields(LiteConfig)}
    cfg = LiteConfig(**{k: copy.deepcopy(v) for k, v in raw.items() if k in known and k != "seeds"},
                     seeds=SeedConfig(**raw["seeds"]))           # effective config; never normalize()
    require(cfg.task_name == "drawer_open" and cfg.max_episode_steps == 200, "DrawerOpen 200-step only")
    require(cfg.success_terminates is False and cfg.timeout_bootstrap is True, "terminal convention changed")
    require(cfg.support_scan_mode == "deterministic_tail_v1" and cfg.policy_support_mode == "bin", "support rule changed")
    require((cfg.N_local_min, cfg.M_Q_min, cfg.M_B, cfg.local_search_max, cfg.action_bins_per_dim)
            == (4, 4, 16, 512, 5), "support hyperparameters differ from the submitted preset")
    kind = str(payload.get("checkpoint_kind", ""))
    require(kind in {"a1_scheduled", "periodic"} and not Path(path).name.startswith("best"),
            f"only scheduled/periodic checkpoints are allowed (got kind={kind!r}); best-checkpoint selection is disallowed")
    if "a1_metadata" in payload:
        arm, seed = str(payload["a1_metadata"]["arm"]), int(payload["a1_metadata"]["seed"])
    else:
        seed = int(cfg.seeds.env_seed)
        if cfg.method == "b0_ddpg_her":
            arm = "her"
        else:
            require(cfg.method == "b3_support_runtime_opt" and cfg.weight_scope == "critic_only"
                    and cfg.weighting_mode == "batch_norm" and float(cfg.lambda_support) == 1.0,
                    "unrecognised training method")
            rule = str(payload.get("assignment_rule", raw.get("assignment_rule", "support")))
            require(rule in {"support", "random", "td_topk"}, f"unknown assignment rule {rule!r}")
            arm = {"support": "saber", "random": "random", "td_topk": "td_topk"}[rule]
    history = payload.get("eval_history_env") or []
    last_eval = float((payload.get("last_eval_metrics") or {}).get("eval_success_mean", float("nan")))
    tail = float(np.mean([v for _, v in history[-10:]])) if history else float("nan")
    task = make_task_spec(cfg, SimpleNamespace(action_low=-np.ones(4, np.float32), action_high=np.ones(4, np.float32),
                                               distance_threshold=float(cfg.drawer_success_threshold)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        agent = DDPGAgent(state_dim=42, action_dim=4, hidden_dim=cfg.hidden_dim, actor_lr=cfg.actor_lr,
                          critic_lr=cfg.critic_lr, gamma=cfg.gamma, tau=cfg.tau,
                          action_low=-np.ones(4, np.float32), action_high=np.ones(4, np.float32),
                          device=torch.device("cpu"), l2_action_penalty=cfg.l2_action_penalty)
    agent.load_state_dict(payload["agent"])
    for m in (agent.actor, agent.actor_target, agent.critic, agent.critic_target):
        m.eval()
        m.requires_grad_(False)
    r = payload["replay"]
    replay = HERReplayBuffer(r["capacity"], r["obs_dim"], r["goal_dim"], r["action_dim"])
    replay.load_state_dict(r)
    require(replay.indexed_slot_count() == replay.size, "unindexed replay slots")
    wcfg = copy.deepcopy(cfg)
    wcfg.method = "b3_support_runtime_opt"          # the SABER predicate, whatever arm trained the checkpoint
    wcfg.support_assignment_mode = "support"
    wcfg.weight_scope, wcfg.weighting_mode, wcfg.lambda_support = "critic_only", "batch_norm", 1.0
    # Without this flag the production code skips the exact eligibility count whenever the policy
    # bin is absent from the coarse bucket (psi is 0 either way); the control arms need exact E.
    wcfg.assignment_experiment, wcfg.assignment_rule = True, "support"
    wcfg.support_diagnostic_audit_interval = 0
    wc = LiteWeightComputer(wcfg, task, GapStats(8, 128, 0.1, 0.9, 1e-6))
    src = Source(Path(path), int(payload["global_step"]), arm, seed, cfg, agent, replay, task, wc, "",
                 kind=kind, last_eval_success=last_eval, eval_tail_mean=tail,
                 total_updates=int(payload.get("total_updates", payload["agent"].get("total_it", -1))),
                 family="drawer_ddpg", heads=Heads([agent.critic]), obs_dim=39,
                 success_threshold=float(cfg.drawer_success_threshold))
    src.frozen_hash = src.state_hash()
    return src


def _eval_history(run_dir: Path, step: int) -> tuple[float, float]:
    """(evaluation success at `step`, mean of the last 10 evaluations up to `step`) from metrics.csv."""
    path = run_dir / "metrics.csv"
    if not path.exists():
        return float("nan"), float("nan")
    hist = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            v = row.get("eval/success_mean", "")
            if v not in ("", None):
                hist[int(float(row["step"]))] = float(v)
    upto = [hist[k] for k in sorted(hist) if k <= step]
    return hist.get(step, float("nan")), (float(np.mean(upto[-10:])) if upto else float("nan"))


def load_source_fpp(path: Path, code_src: Path) -> Source:
    """FetchPickAndPlace TD3 checkpoint, loaded with the package that trained it."""
    alias = "fou_pkg_" + "".join(ch if ch.isalnum() else "_" for ch in str(code_src.resolve()))[-60:]
    pk = load_package(Path(code_src) / "metacog_lite_v0", alias)
    cfgm, agentm, replaym, weightsm, taskm, envsm = (pk(n) for n in (
        "config", "agent", "replay", "weights", "task_fetch_pick_and_place", "envs"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = payload["config"]
    known = {f.name for f in fields(cfgm.LiteConfig)}
    cfg = cfgm.LiteConfig(**{k: copy.deepcopy(v) for k, v in raw.items() if k in known and k != "seeds"},
                          seeds=cfgm.SeedConfig(**raw["seeds"]))          # effective config; never normalize()
    require(cfg.task_name == "fetch_pick_and_place" and cfg.max_episode_steps == 50, "FetchPickAndPlace 50-step only")
    require(cfg.success_terminates is False and cfg.timeout_bootstrap is True, "terminal convention changed")
    require(cfg.support_scan_mode == "deterministic_tail_v1" and cfg.policy_support_mode == "bin", "support rule changed")
    require((cfg.N_local_min, cfg.M_Q_min, cfg.M_B, cfg.local_search_max, cfg.action_bins_per_dim)
            == (4, 4, 16, 512, 5), "support hyperparameters differ from the submitted preset")
    require(cfg.method in {"td3_her", "td3_saber_runtime_opt"}, f"not a TD3 run: {cfg.method}")
    kind = str(payload.get("checkpoint_kind", ""))
    require(kind == "periodic" and not Path(path).name.startswith("best"),
            f"only periodic checkpoints are allowed (got kind={kind!r}); best-checkpoint selection is disallowed")
    arm = "her" if cfg.method == "td3_her" else "saber"
    if arm == "saber":
        require(cfg.weight_scope == "critic_only" and cfg.weighting_mode == "batch_norm"
                and float(cfg.lambda_support) == 1.0, "unrecognised SABER settings")
    env = envsm.FetchGoalEnv(cfg.env_id, cfg.max_episode_steps, seed=0, render=False)
    require((env.obs_dim, env.goal_dim, env.action_dim) == (25, 3, 4), "unexpected Fetch observation/action schema")
    task = taskm.FetchPickAndPlaceTaskSpec(
        action_low=env.action_low, action_high=env.action_high, epsilon_g=env.distance_threshold,
        d_reach=cfg.d_reach, d_goal_near=cfg.d_goal_near, z_lift_min=cfg.z_lift_min,
        grip_close_thresh=cfg.grip_close_thresh, hand_obj_bins=cfg.pick_hand_obj_bins,
        obj_goal_bins=cfg.pick_obj_goal_bins, object_z_bins=cfg.pick_object_z_bins,
        gripper_gap_bins=cfg.pick_gripper_gap_bins, horizon_bins=cfg.horizon_bins)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        agent = agentm.TD3Agent(state_dim=env.state_dim, action_dim=env.action_dim, hidden_dim=cfg.hidden_dim,
                                actor_lr=cfg.actor_lr, critic_lr=cfg.critic_lr, gamma=cfg.gamma, tau=cfg.tau,
                                action_low=env.action_low, action_high=env.action_high, device=torch.device("cpu"),
                                l2_action_penalty=cfg.l2_action_penalty, policy_delay=cfg.td3_policy_delay,
                                target_policy_noise=cfg.td3_target_policy_noise,
                                target_noise_clip=cfg.td3_target_noise_clip)
    agent.load_state_dict(payload["agent"])
    for m in (agent.actor, agent.actor_target, agent.critic1, agent.critic2, agent.critic1_target, agent.critic2_target):
        m.eval()
        m.requires_grad_(False)
    r = payload["replay"]
    replay = replaym.HERReplayBuffer(r["capacity"], r["obs_dim"], r["goal_dim"], r["action_dim"])
    replay.load_state_dict(r)
    n = replay.size
    require(replay.indexed_slot_count() == n, "unindexed replay slots")
    # The task spec above is rebuilt from the config, not taken from the trainer: it must
    # reproduce the index keys and action bins that training stored.
    require(np.array_equal(replay.action_index._key_by_slot[:n], task.coarse_hash_batch(replay.observation[:n], replay.h[:n])),
            "rebuilt task spec does not reproduce the stored coarse index keys")
    require(np.array_equal(replay.action_bin[:n], task.action_bin_batch(replay.action_exec[:n], cfg.action_bins_per_dim)),
            "rebuilt task spec does not reproduce the stored action bins")
    # HER convention of this package: can the final step of an episode be relabelled with itself?
    last = int(np.flatnonzero(replay.episode_step[:n] == cfg.max_episode_steps - 1)[0])
    inclusive = replay._sample_future_goal(last, np.random.default_rng(0)) is not None
    wcfg = copy.deepcopy(cfg)
    wcfg.method = "td3_saber_runtime_opt"          # the SABER predicate, whatever arm trained the checkpoint
    wcfg.weight_scope, wcfg.weighting_mode, wcfg.lambda_support = "critic_only", "batch_norm", 1.0
    wcfg.support_diagnostic_audit_interval = 0
    wc = weightsm.LiteWeightComputer(wcfg, task, weightsm.GapStats(8, 128, 0.1, 0.9, 1e-6))
    last_eval, tail = _eval_history(Path(path).parent, int(payload["global_step"]))
    src = Source(Path(path), int(payload["global_step"]), arm, int(cfg.seeds.env_seed), cfg, agent, replay, task, wc, "",
                 kind=kind, last_eval_success=last_eval, eval_tail_mean=tail,
                 total_updates=int(payload.get("total_updates", payload["agent"].get("total_it", -1))),
                 family="fpp_td3", heads=Heads([agent.critic1, agent.critic2]), obs_dim=env.obs_dim,
                 success_threshold=float(env.distance_threshold), her_inclusive=bool(inclusive), env=env)
    src.frozen_hash = src.state_hash()
    return src


def target_q(src: Source, S2: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
    """Frozen-target value at next states: Q'(s', pi'(s')) for DDPG, min(Q1', Q2') at the
    smoothed target action for TD3 (noise is added and clipped exactly as in TD3Agent.update)."""
    ag = src.agent
    if src.family == "drawer_ddpg":
        return ag.critic_target(S2, ag.actor_target(S2)).reshape(-1)
    a2 = ag.actor_target(S2)
    if noise is not None:
        a2 = torch.max(torch.min(a2 + noise, ag.action_high_t), ag.action_low_t)
    return torch.minimum(ag.critic1_target(S2, a2), ag.critic2_target(S2, a2)).reshape(-1)


def smoothing_noise(src: Source, shape, generator: torch.Generator) -> torch.Tensor | None:
    if src.family == "drawer_ddpg":
        return None
    ag = src.agent
    return torch.clamp(torch.randn(shape, generator=generator) * ag.target_policy_noise,
                       -ag.target_noise_clip, ag.target_noise_clip)


def expected_target_q(src: Source, S2: torch.Tensor) -> torch.Tensor:
    """Policy-query banks regress to the expectation of the training target over smoothing noise."""
    if src.family == "drawer_ddpg":
        return target_q(src, S2)
    g = torch.Generator().manual_seed(TARGET_NOISE_SEED)
    shape = (S2.shape[0], len(src.agent.action_low_np))
    return torch.stack([target_q(src, S2, smoothing_noise(src, shape, g)) for _ in range(N_TARGET_NOISE)]).mean(0)


# ----------------------------------------------------------------------------- parameters
class Params:
    """Flat view of the critic head parameters, in heads.parameters() order (= Adam state order)."""

    def __init__(self, heads: torch.nn.Module):
        self.critic = heads
        self.n_heads = len(heads.c)
        self.names = [n for n, _ in heads.named_parameters()]
        self.shapes = [p.shape for _, p in heads.named_parameters()]
        self.sizes = [p.numel() for _, p in heads.named_parameters()]
        self.flat0 = torch.cat([p.detach().reshape(-1) for p in heads.parameters()]).clone()

    def unflat(self, v: torch.Tensor) -> dict:
        out, i = {}, 0
        for n, s, k in zip(self.names, self.shapes, self.sizes):
            out[n] = v[i:i + k].reshape(s)
            i += k
        return out

    def q_all(self, v: torch.Tensor, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return functional_call(self.critic, self.unflat(v), (s, a))                # [H, B]

    def q(self, v: torch.Tensor, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.q_all(v, s, a)[0]                                               # the head the actor reads

    def per_sample_jacobian(self, v: torch.Tensor, s: torch.Tensor, a: torch.Tensor, head: int = 0) -> torch.Tensor:
        def one(vv, ss, aa):
            return self.q_all(vv, ss[None], aa[None])[head, 0]
        return vmap(grad(one), in_dims=(None, 0, 0))(v, s, a)          # [B, P]


def adam_delta(opt_state: dict, lr: float, g: torch.Tensor, sizes: list[int],
               dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """torch.optim.Adam (no amsgrad / weight decay) update vector for gradient g.

    dtype=float32 follows torch's single-tensor operation order exactly (lerp, addcmul,
    sqrt/bias_correction2_sqrt + eps, addcdiv); float64 is the same formula in double precision."""
    group = opt_state["param_groups"][0]
    require(not group.get("amsgrad", False) and group.get("weight_decay", 0.0) == 0.0
            and not group.get("maximize", False), "unsupported Adam options")
    b1, b2 = group["betas"]
    eps = group["eps"]
    out, i = [], 0
    for idx, k in enumerate(sizes):
        st = opt_state["state"][idx]
        m = st["exp_avg"].reshape(-1).to(dtype)
        v = st["exp_avg_sq"].reshape(-1).to(dtype)
        t = float(st["step"]) + 1.0
        gi = g[i:i + k].to(dtype)
        m1 = m.lerp(gi, 1 - b1)
        v1 = v.mul(b2).addcmul(gi, gi, value=1 - b2)
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        denom = (v1.sqrt() / (bc2 ** 0.5)).add(eps)
        out.append(torch.zeros_like(m1).addcdiv(m1, denom, value=-(lr / bc1)))
        i += k
    return torch.cat(out)


def torch_step(src: "Source", w: torch.Tensor, bt: "Batch", return_params: bool = False) -> torch.Tensor:
    """The real production critic step (weighted loss, clip 10, Adam from saved moments) on a
    copy of the critic; returns the parameter change (or the new parameters)."""
    c = copy.deepcopy(src.heads)
    c.requires_grad_(True)
    opt = torch.optim.Adam(c.parameters(), lr=src.cfg.critic_lr)
    opt.load_state_dict(copy.deepcopy(src.agent.critic_opt.state_dict()))
    before = torch.cat([p.detach().reshape(-1) for p in c.parameters()]).clone()
    res = c(bt.S, bt.A) - bt.y                                           # [H, B]
    loss = sum((w * res[h].pow(2)).mean() for h in range(res.shape[0]))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(c.parameters(), CLIP)
    opt.step()
    after = torch.cat([p.detach().reshape(-1) for p in c.parameters()])
    return after if return_params else after - before


def clip(g: torch.Tensor, max_norm: float = CLIP) -> torch.Tensor:
    """Same coefficient as torch.nn.utils.clip_grad_norm_ for a single flat gradient."""
    coef = torch.clamp(max_norm / (torch.linalg.vector_norm(g) + 1e-6), max=1.0)
    return g * coef


# ----------------------------------------------------------------------------- batches and masks
@dataclass
class Batch:
    seed: int
    S: torch.Tensor
    A: torch.Tensor
    y: torch.Tensor
    a_pi: torch.Tensor
    eligible: np.ndarray
    support: np.ndarray
    local_count: np.ndarray
    self_support: np.ndarray
    phase: np.ndarray
    omega_production: np.ndarray
    slot: np.ndarray | None = None
    age: np.ndarray | None = None            # env steps since the transition was stored (0 = newest)
    S2: torch.Tensor | None = None           # raw pieces of the target, kept for validation only
    reward: torch.Tensor | None = None
    nonterm: torch.Tensor | None = None
    noise: torch.Tensor | None = None


def exact_masks(src: Source, batch, a_pi: np.ndarray):
    """Exact (no short-circuit) eligibility E, support M and local count n for every sample."""
    r, c, task = src.replay, src.cfg, src.task
    coarse = task.coarse_hash_batch(batch.observation, batch.h)
    full = task.full_context_hash_batch(batch.observation, batch.desired_goal_train, batch.h)
    pbins = task.action_bin_batch(a_pi, c.action_bins_per_dim)
    rbins_self = task.action_bin_batch(batch.action_exec, c.action_bins_per_dim)
    B = len(full)
    e, m, n = np.zeros(B, bool), np.zeros(B, bool), np.zeros(B, np.int64)
    cache: dict[int, np.ndarray] = {}
    for i in range(B):
        key = int(coarse[i])
        if key not in cache:
            cache[key], _, _ = r.indexed_reference_indices_deterministic(key, c.local_search_max, c.support_scan_mode)
        cand = cache[key]
        goal = np.repeat(batch.desired_goal_train[i:i + 1], len(cand), axis=0)
        matched = cand[task.full_context_hash_batch(r.observation[cand], goal, r.h[cand]) == full[i]]
        n[i] = len(matched)
        e[i] = n[i] >= c.N_local_min and min(n[i], c.M_B) >= c.M_Q_min
        bins = r.action_bin[matched].copy()
        miss = bins < 0
        if np.any(miss):
            bins[miss] = task.action_bin_batch(r.action_exec[matched[miss]], c.action_bins_per_dim)
        m[i] = e[i] and bool(np.any(bins == pbins[i]))
    self_support = m & (rbins_self == pbins)
    return e, m, n, self_support


def make_batch(src: Source, seed: int, check_production: bool) -> Batch:
    c = src.cfg
    b = src.replay.sample(c.batch_size, c.her_ratio, np.random.default_rng(seed))
    reward = np.where(np.linalg.norm(b.next_achieved_goal - b.desired_goal_train, axis=1)
                      <= src.success_threshold, 0.0, -1.0).astype(np.float32)
    nonterm = (1.0 - b.done.astype(np.float32) * (1.0 - b.timeout.astype(np.float32))).astype(np.float32)
    S = torch.as_tensor(np.concatenate([b.observation, b.desired_goal_train], 1), dtype=torch.float32)
    S2 = torch.as_tensor(np.concatenate([b.next_observation, b.desired_goal_train], 1), dtype=torch.float32)
    A = torch.as_tensor(b.action_exec, dtype=torch.float32)
    ag = src.agent
    with torch.no_grad():
        # TD3: one smoothing-noise draw per batch, fixed by the batch seed and shared by every arm
        noise = smoothing_noise(src, A.shape, torch.Generator().manual_seed(int(seed)))
        y = torch.as_tensor(reward) + c.gamma * torch.as_tensor(nonterm) * target_q(src, S2, noise)
        a_pi = ag.actor(S)
    finite(y, "Bellman target")
    e, m, n, self_support = exact_masks(src, b, a_pi.numpy())
    omega_prod = np.full(c.batch_size, np.nan, np.float32)
    if check_production:
        wb = src.weight_computer.compute(b, src.replay, ag, np.random.default_rng(1), torch.device("cpu"))
        require(np.array_equal(wb.psi_q_rank.astype(bool), m), f"support parity failure (batch seed {seed})")
        if getattr(wb, "context_eligible_mask", None) is not None:
            require(np.array_equal(np.asarray(wb.context_eligible_mask, bool), e), f"eligibility parity failure ({seed})")
        omega_prod = wb.omega_critic.astype(np.float32)
        require(np.allclose(omega_prod, weights(m)), f"omega parity failure ({seed})")
    phase = src.task.phase_batch(b.observation, b.desired_goal_train)
    slot = np.asarray(b.indices, np.int64)
    r = src.replay
    newest = (r.ptr - 1) % r.capacity if r.size == r.capacity else r.size - 1     # ring buffer, one slot per env step
    age = (newest - slot) % r.capacity if r.size == r.capacity else newest - slot
    return Batch(seed, S, A, y, a_pi, e, m, n, self_support, phase, omega_prod, slot=slot, age=age.astype(np.int64),
                 S2=S2, reward=torch.as_tensor(reward), nonterm=torch.as_tensor(nonterm), noise=noise)


def weights(mask: np.ndarray) -> np.ndarray:
    raw = 1.0 + np.asarray(mask, np.float32)
    return (raw / max(1e-8, float(np.mean(raw)))).astype(np.float32)


def arm_mask(arm: str, bt: Batch, residual: np.ndarray | None, rng: np.random.Generator | None) -> np.ndarray:
    m, e = bt.support, bt.eligible
    k = int(m.sum())
    if arm == "uniform":
        return np.zeros_like(m)
    if arm == "saber":
        return m.copy()
    if arm == "inverse":
        return ~m
    idx = np.flatnonzero(e)
    out = np.zeros_like(m)
    if k == 0:
        return out
    if k == len(idx):
        out[idx] = True
        return out
    if arm == "random":
        out[rng.choice(idx, k, replace=False)] = True
        return out
    require(arm == "td_topk", f"unknown arm {arm}")
    v = np.abs(residual)
    cutoff = np.partition(v[idx], len(idx) - k)[len(idx) - k]
    above, ties = idx[v[idx] > cutoff], idx[v[idx] == cutoff]
    out[above] = True
    out[rng.choice(ties, k - len(above), replace=False)] = True
    return out


# ----------------------------------------------------------------------------- query banks
@dataclass
class Bank:
    name: str
    S: torch.Tensor
    A: torch.Tensor
    y: torch.Tensor
    phase: np.ndarray


def _target(src: Source, s2: np.ndarray) -> float:
    with torch.no_grad():
        t = torch.as_tensor(s2[None], dtype=torch.float32)
        return float(src.agent.critic_target(t, src.agent.actor_target(t)))


def _policy(src: Source, s: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return src.agent.actor(torch.as_tensor(s[None], dtype=torch.float32))[0].numpy().copy()


def bank_r1(src: Source, episodes: int, seed0: int = 50_000) -> Bank:
    return _bank_r1_drawer(src, episodes, seed0) if src.family == "drawer_ddpg" else _bank_r1_fetch(src, episodes, seed0)


def bank_r2(src: Source, episodes: int, n_queries: int, seed0: int = 60_000) -> Bank:
    return (_bank_r2_drawer(src, episodes, n_queries, seed0) if src.family == "drawer_ddpg"
            else _bank_r2_fetch(src, episodes, n_queries, seed0))


def _bank_r1_drawer(src: Source, episodes: int, seed0: int = 50_000) -> Bank:
    """Noise-free policy rollouts, original goal; the executed action IS a^pi, so the 1-step
    frozen-target Bellman target is exact for that query."""
    c = src.cfg
    sim = DrawerSimulator(c, seed0)
    S, A, Y, ph = [], [], [], []
    for ep in range(episodes):
        sim.reset(seed0 + ep)
        for _ in range(c.max_episode_steps):
            s = sim.state()
            a = _policy(src, s)
            out = sim.step(a)
            require(not out["terminated"], "unexpected environment termination")
            s2 = np.concatenate([sim.obs, s[39:]]).astype(np.float32)
            S.append(s); A.append(a); Y.append(out["reward"] + c.gamma * _target(src, s2))   # timeout bootstraps
            if out["terminated"] or out["truncated"]:
                break
    S = np.asarray(S, np.float32)
    ph = src.task.phase_batch(S[:, :39], S[:, 39:])
    sim.close()
    return Bank("R1", torch.as_tensor(S), torch.as_tensor(np.asarray(A, np.float32)),
                torch.as_tensor(np.asarray(Y, np.float32)), ph)


def _bank_r2_drawer(src: Source, episodes: int, n_queries: int, seed0: int = 60_000) -> Bank:
    """Noisy behaviour rollouts with snapshots. Each query state gets a future-achieved HER goal
    (p = her_ratio, same rule as replay.sample), the actor's action for that goal, and the
    1-step transition obtained by restoring the snapshot and executing that action."""
    c = src.cfg
    sim = DrawerSimulator(c, seed0)
    rng = np.random.default_rng(seed0)
    T = c.max_episode_steps
    per_ep = int(np.ceil(n_queries / episodes))
    episodes_data = []
    for ep in range(episodes):
        sim.reset(seed0 + ep)
        chosen = set(int(t) for t in rng.choice(T, size=min(per_ep, T), replace=False))
        snaps, states, ach_next = {}, [], []
        for t in range(T):
            s = sim.state()
            states.append(s)
            if t in chosen:
                snaps[t] = sim.snapshot()             # only where a query will be made (memory)
            a = np.clip(_policy(src, s) + rng.normal(0.0, c.noise_std, 4), -1.0, 1.0).astype(np.float32)
            out = sim.step(a)
            require(not out["terminated"], "unexpected environment termination")
            ach_next.append(sim.obs[4:7].copy())
            if out["truncated"]:
                break
        require(len(states) == T, "behaviour episode shorter than the source horizon")
        episodes_data.append((snaps, states, ach_next))
    pool = [(e, t) for e, (sn, _, _) in enumerate(episodes_data) for t in sorted(sn)][:n_queries]
    S, A, Y = [], [], []
    for e, t in pool:
        snaps, states, ach_next = episodes_data[e]
        goal = states[t][39:].copy()
        if rng.random() < c.her_ratio and t + 1 <= T - 1:
            u = int(rng.integers(t + 1, T))                 # future stored step, as replay.sample
            goal = ach_next[u].copy()
        s = np.concatenate([states[t][:39], goal]).astype(np.float32)
        a = _policy(src, s)
        sim.restore(snaps[t])
        require(np.array_equal(sim.state(), states[t]), "snapshot restore changed the query state")
        sim.step(a)
        reward = 0.0 if np.linalg.norm(sim.obs[4:7] - goal) <= c.drawer_success_threshold else -1.0
        s2 = np.concatenate([sim.obs, goal]).astype(np.float32)
        S.append(s); A.append(a); Y.append(reward + c.gamma * _target(src, s2))
    S = np.asarray(S, np.float32)
    ph = src.task.phase_batch(S[:, :39], S[:, 39:])
    sim.close()
    return Bank("R2", torch.as_tensor(S), torch.as_tensor(np.asarray(A, np.float32)),
                torch.as_tensor(np.asarray(Y, np.float32)), ph)


class FetchSim:
    """The training package's FetchGoalEnv plus MuJoCo state save/restore for one-step queries.
    validate.py checks that restore + step reproduces replay-from-reset + step bit-for-bit."""

    def __init__(self, src: Source):
        import mujoco
        self.mj, self.env, self.u = mujoco, src.env, src.env._env.unwrapped
        self._parse = type(src.env)._parse_obs

    def reset(self, seed: int):
        return self.env.reset(seed=int(seed))

    def step(self, action: np.ndarray):
        o, _, terminated, truncated, _ = self.env.step(action)
        require(not terminated, "unexpected environment termination")
        return o, truncated

    def snapshot(self):
        # A full mjData copy, not mj_getState: Fetch builds its observation from derived arrays
        # (site positions, velocities) that are one sub-step stale after mj_step, so recomputing
        # them with mj_forward after a state restore would change the observation.
        return copy.copy(self.u.data)

    def restore(self, snap) -> None:
        # The bindings expose no in-place copy; the env reads self.data on every call, so swapping
        # the object is equivalent (only the unused renderer keeps the old reference).
        self.u.data = copy.copy(snap)

    def observe(self):
        return self._parse(self.u._get_obs(), {})

    def step_raw(self, action: np.ndarray):
        """One step on the unwrapped env (no time-limit bookkeeping); used right after restore."""
        obs, _, terminated, _, info = self.u.step(np.asarray(action, dtype=np.float32))
        require(not terminated, "unexpected environment termination")
        return self._parse(obs, info)


def _fetch_bank(src: Source, name: str, S, A, S2, Rw) -> Bank:
    S = np.asarray(S, np.float32)
    with torch.no_grad():
        y = torch.as_tensor(np.asarray(Rw, np.float32)) + src.cfg.gamma * expected_target_q(
            src, torch.as_tensor(np.asarray(S2, np.float32)))            # timeouts bootstrap; Fetch never terminates
    finite(y, "policy-query target")
    ph = src.task.phase_batch(S[:, :src.obs_dim], S[:, src.obs_dim:])
    return Bank(name, torch.as_tensor(S), torch.as_tensor(np.asarray(A, np.float32)), y, ph)


def _bank_r1_fetch(src: Source, episodes: int, seed0: int = 50_000) -> Bank:
    """Noise-free policy rollouts with the original goal (see _bank_r1_drawer)."""
    c, sim, thr = src.cfg, FetchSim(src), src.success_threshold
    S, A, S2, Rw = [], [], [], []
    for ep in range(episodes):
        o = sim.reset(seed0 + ep)
        goal = o.desired_goal.copy()
        for _ in range(c.max_episode_steps):
            s = np.concatenate([o.observation, goal]).astype(np.float32)
            a = _policy(src, s)
            o, truncated = sim.step(a)
            S.append(s); A.append(a); S2.append(np.concatenate([o.observation, goal]).astype(np.float32))
            Rw.append(0.0 if np.linalg.norm(o.achieved_goal - goal) <= thr else -1.0)
            if truncated:
                break
    return _fetch_bank(src, "R1", S, A, S2, Rw)


def _bank_r2_fetch(src: Source, episodes: int, n_queries: int, seed0: int = 60_000) -> Bank:
    """Noisy behaviour rollouts; each query state gets a future-achieved HER goal chosen with
    this package's own rule (inclusive or exclusive), the actor's action for that goal, and the
    1-step transition obtained by restoring the saved simulator state and executing that action."""
    c, sim, thr = src.cfg, FetchSim(src), src.success_threshold
    rng = np.random.default_rng(seed0)
    T = c.max_episode_steps
    per_ep = int(np.ceil(n_queries / episodes))
    S, A, S2, Rw = [], [], [], []
    for ep in range(episodes):
        o = sim.reset(seed0 + ep)
        goal0 = o.desired_goal.copy()
        chosen = set(int(t) for t in rng.choice(T, size=min(per_ep, T), replace=False))
        snaps, obs, ach_next = {}, [], []
        for t in range(T):
            obs.append(o.observation.copy())
            if t in chosen:
                snaps[t] = sim.snapshot()
            a = np.clip(_policy(src, np.concatenate([o.observation, goal0]).astype(np.float32))
                        + rng.normal(0.0, c.noise_std, 4), -1.0, 1.0).astype(np.float32)
            o, truncated = sim.step(a)
            ach_next.append(o.achieved_goal.copy())
            if truncated:
                break
        require(len(obs) == T, "behaviour episode shorter than the source horizon")
        for t in sorted(snaps):                      # after the episode, so future achieved goals are known
            goal = goal0.copy()
            lo = t if src.her_inclusive else t + 1
            if rng.random() < c.her_ratio and lo <= T - 1:
                goal = ach_next[int(rng.integers(lo, T))].copy()
            s = np.concatenate([obs[t], goal]).astype(np.float32)
            a = _policy(src, s)
            sim.restore(snaps[t])
            require(np.array_equal(sim.observe().observation, obs[t]), "state restore changed the query observation")
            o2 = sim.step_raw(a)
            S.append(s); A.append(a); S2.append(np.concatenate([o2.observation, goal]).astype(np.float32))
            Rw.append(0.0 if np.linalg.norm(o2.achieved_goal - goal) <= thr else -1.0)
    return _fetch_bank(src, "R2", S[:n_queries], A[:n_queries], S2[:n_queries], Rw[:n_queries])


def risk(P: Params, v: torch.Tensor, bank: Bank) -> torch.Tensor:
    return (P.q(v, bank.S, bank.A) - bank.y).pow(2).mean()


def risk_grad(P: Params, v: torch.Tensor, bank: Bank) -> torch.Tensor:
    vv = v.detach().clone().requires_grad_(True)
    return torch.autograd.grad(risk(P, vv, bank), vv)[0].detach()


# ----------------------------------------------------------------------------- per-batch first-order analysis
def analyse_batch(src: Source, P: Params, bt: Batch, banks: list[Bank], gR0: dict, R0: dict,
                  rng: np.random.Generator) -> tuple[dict, dict]:
    v0 = P.flat0
    lr = src.cfg.critic_lr
    opt_state = src.agent.critic_opt.state_dict()
    H = P.n_heads
    with torch.no_grad():
        Q = P.q_all(v0, bt.S, bt.A)
    deltas = [Q[h] - bt.y for h in range(H)]
    delta = deltas[0]                                   # residual of the head the actor reads
    Js = [P.per_sample_jacobian(v0, bt.S, bt.A, h) for h in range(H)]
    Jr = Js[0]
    Jp = P.per_sample_jacobian(v0, bt.S, bt.a_pi)
    # TD ranking as in production: |delta| for one critic, min(|td1|, |td2|) for twin critics
    rank_res = delta.numpy() if H == 1 else torch.stack([d.abs() for d in deltas]).min(0).values.numpy()
    B = len(delta)
    m, e = bt.support, bt.eligible
    p = float(m.mean())
    kappa = torch.nn.functional.cosine_similarity(Jr, Jp, dim=1).numpy()
    dist = torch.linalg.vector_norm(bt.A - bt.a_pi, dim=1).numpy()
    jr_norm = torch.linalg.vector_norm(Jr, dim=1).numpy()
    absd = delta.abs().numpy()

    # arm masks (TD uses the pre-update residual, as the A1 control did)
    masks = {"uniform": arm_mask("uniform", bt, None, None), "saber": arm_mask("saber", bt, None, None),
             "inverse": arm_mask("inverse", bt, None, None),
             "td_topk": arm_mask("td_topk", bt, rank_res, rng)}
    random_masks = [arm_mask("random", bt, None, rng) for _ in range(N_RANDOM_DRAWS)]

    def g_of(mask):                                             # (1/B) sum_h sum_i w_i * 2 delta_hi J_hi
        w = torch.as_tensor(weights(mask))
        return sum((w * 2.0 * deltas[h]) @ Js[h] for h in range(H)) / B

    grads = {k: g_of(v) for k, v in masks.items()}
    grads_rand = [g_of(mm) for mm in random_masks]
    upd = {k: adam_delta(opt_state, lr, clip(g), P.sizes).float() for k, g in grads.items()}
    upd_rand = [adam_delta(opt_state, lr, clip(g), P.sizes).float() for g in grads_rand]
    real = {k: torch_step(src, torch.as_tensor(weights(v)), bt) for k, v in masks.items()}
    real_rand = [torch_step(src, torch.as_tensor(weights(mm)), bt) for mm in random_masks]

    row = dict(batch_seed=bt.seed, p=p, n_support=int(m.sum()), n_eligible=int(e.sum()),
               degenerate=bool(m.sum() == 0 or m.sum() == B),
               gnorm_uniform=float(grads["uniform"].norm()), gnorm_saber=float(grads["saber"].norm()),
               clipped_uniform=bool(float(grads["uniform"].norm()) > CLIP),
               clipped_saber=bool(float(grads["saber"].norm()) > CLIP))
    per_sample = dict(delta=delta.numpy(), kappa=kappa, dist=dist, support=m, eligible=e,
                      self_support=bt.self_support, local_count=bt.local_count, phase=bt.phase,
                      slot=bt.slot, age=bt.age)
    if H > 1:
        per_sample["delta_other_heads"] = torch.stack(deltas[1:], dim=1).numpy()
        # heads own disjoint parameter blocks, so the per-sample loss-gradient norm adds in squares
        grad_norm = np.sqrt(sum((2.0 * deltas[h].abs() * torch.linalg.vector_norm(Js[h], dim=1)).numpy() ** 2
                                for h in range(H)))

    for bank in banks:
        gR = gR0[bank.name]
        u_t = 2.0 * delta * (Jr @ gR)
        for h in range(1, H):
            u_t = u_t + 2.0 * deltas[h] * (Js[h] @ gR)
        u = u_t.numpy()
        per_sample[f"u_{bank.name}"] = u
        if H == 1:
            cos = u / np.maximum(2.0 * absd * jr_norm * float(gR.norm()), 1e-30)
        else:
            cos = u / np.maximum(grad_norm * float(gR.norm()), 1e-30)
        per_sample[f"cos_{bank.name}"] = cos
        groups = {"M": m, "E_not_M": e & ~m, "not_E": ~e, "all": np.ones(B, bool)}
        for gname, gm in groups.items():
            row[f"{bank.name}_u_{gname}"] = float(u[gm].mean()) if gm.any() else np.nan
            row[f"{bank.name}_cos_{gname}"] = float(cos[gm].mean()) if gm.any() else np.nan
        row[f"{bank.name}_u_TD"] = float(u[masks["td_topk"]].mean()) if masks["td_topk"].any() else np.nan
        row[f"{bank.name}_abs_u_mean"] = float(np.abs(u).mean())

        def g1(mask):
            return float(((weights(mask) - 1.0) * u).mean())

        def g2(g):
            return float(gR @ (clip(g) - clip(grads["uniform"])))

        def g3(dphi):
            return float(gR @ (upd["uniform"] - dphi))

        def g3mid(dphi):
            mid = v0 + 0.5 * (upd["uniform"] + dphi)
            return float(risk_grad(P, mid, bank) @ (upd["uniform"] - dphi))

        def actual(dphi):
            with torch.no_grad():
                return float(risk(P, v0 + real["uniform"], bank) - risk(P, v0 + dphi, bank))

        for arm in ("saber", "inverse", "td_topk"):
            row[f"{bank.name}_G1_{arm}"] = g1(masks[arm])
            row[f"{bank.name}_G2_{arm}"] = g2(grads[arm])
            row[f"{bank.name}_G3_{arm}"] = g3(upd[arm])
            row[f"{bank.name}_G3mid_{arm}"] = g3mid(upd[arm])
            row[f"{bank.name}_A1_{arm}"] = actual(real[arm])
        row[f"{bank.name}_G1_random"] = float(np.mean([g1(mm) for mm in random_masks]))
        row[f"{bank.name}_G2_random"] = float(np.mean([g2(g) for g in grads_rand]))
        row[f"{bank.name}_G3_random"] = float(np.mean([g3(d) for d in upd_rand]))
        row[f"{bank.name}_G3mid_random"] = float(np.mean([g3mid(d) for d in upd_rand]))
        row[f"{bank.name}_A1_random"] = float(np.mean([actual(d) for d in real_rand]))
        with torch.no_grad():
            row[f"{bank.name}_A1_common_change"] = float(risk(P, v0 + real["uniform"], bank) - R0[bank.name])
    return row, per_sample


# ----------------------------------------------------------------------------- K-step branches
def run_branches(src: Source, P: Params, batches: list[Batch], banks: list[Bank], record: tuple[int, ...],
                 seed: int) -> list[dict]:
    """Production critic step (weighted loss, clip 10, Adam from saved moments) repeated over a
    shared batch sequence with actor and targets frozen. Masks are fixed per batch because
    actor and replay are fixed; TD Top-k re-ranks on the branch's own current residual."""
    arms = ["uniform", "saber", "inverse", "td_topk"] + [f"random{j}" for j in range(N_RANDOM_SEQUENCES)]
    out = {}
    for arm in arms:
        critic = copy.deepcopy(src.heads)
        critic.requires_grad_(True)
        critic.train()
        opt = torch.optim.Adam(critic.parameters(), lr=src.cfg.critic_lr)
        opt.load_state_dict(copy.deepcopy(src.agent.critic_opt.state_dict()))
        rng = np.random.default_rng([seed, arms.index(arm)])
        hits = {}
        for k, bt in enumerate(batches, start=1):
            res = critic(bt.S, bt.A) - bt.y                              # [H, B]
            n_h = res.shape[0]
            rank = res[0].detach().numpy() if n_h == 1 else res.detach().abs().min(0).values.numpy()
            base = "random" if arm.startswith("random") else arm
            mask = arm_mask(base, bt, rank, rng)
            w = torch.as_tensor(weights(mask))
            loss = sum((w * res[h].pow(2)).mean() for h in range(n_h))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), CLIP)
            opt.step()
            if k in record:
                v = torch.cat([p.detach().reshape(-1) for p in critic.parameters()])
                with torch.no_grad():
                    hits[k] = {bank.name: float(risk(P, v, bank)) for bank in banks}
        out[arm] = hits
    rows = []
    for k in record:
        row = {"K": k}
        for bank in banks:
            ru = out["uniform"][k][bank.name]
            row[f"{bank.name}_R_uniform"] = ru
            for arm in ("saber", "inverse", "td_topk"):
                row[f"{bank.name}_K_{arm}"] = ru - out[arm][k][bank.name]
            row[f"{bank.name}_K_random"] = float(np.mean([ru - out[f"random{j}"][k][bank.name]
                                                          for j in range(N_RANDOM_SEQUENCES)]))
        rows.append(row)
    return rows

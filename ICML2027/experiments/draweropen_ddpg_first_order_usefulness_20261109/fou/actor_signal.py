"""Actor-signal measurement (2026-11-17): does a SABER-weighted critic update give the actor a
better action-gradient signal at a^pi than the uniform (HER) update?

The policy-query value-error measurement (core.py) found no SABER advantage in Q(s, a^pi) itself.
The actor, however, never reads Q's level: TD3/DDPG follow grad_a Q(s, a)|a=pi(s). This module
measures that signal against ground truth.

Ground truth. FetchPickAndPlace (MuJoCo) and the checkpoint's actor are deterministic, so from a
saved simulator state the true action value of the current policy is a single rollout:
    G(s, a) = sum_{t<H} gamma^t r_t,  first action a, then pi for H-1 steps, original goal.
For each query state we evaluate G at antithetic perturbations a^pi +/- sigma z_j and compare the
true difference d*_j = G(a+) - G(a-) with the critic's d_j = Q1(s, a+) - Q1(s, a-) (head 0, the
critic the actor reads). Actions are clipped to [-1, 1] identically for G and Q.

Critics compared: the checkpoint critic (K0) and the critic after K = 100 production critic steps
under each weighting arm (same batches, actor and targets frozen; same loop as core.run_branches,
checked equal by validate_actor_signal.py).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from . import core

# ---- fixed 2026-11-17 before any measurement (PROTOCOL_actor_signal_ko.md)
QUERY_EPISODES = 16              # noise-free policy episodes, reset seeds QUERY_RESET0 + e
QUERY_RESET0 = 70_000
QUERIES_PER_EPISODE = 4          # time steps drawn without replacement from 0..49
QUERY_RNG_SEED = 970_000
N_DIRS = 16                      # random unit directions in action space per query (antithetic pairs)
DIR_RNG_SEED = 971_000
SIGMAS = (0.3, 0.1)              # perturbation radii; 0.3 is primary
ASCENT_EPS = 0.3                 # step along the normalised grad_a Q for the ascent test
HORIZON = 50                     # rollout length of G
K_BRANCH = 100
ARMS = ("uniform", "saber", "inverse", "td_topk") + tuple(f"random{j}" for j in range(core.N_RANDOM_SEQUENCES))
INFORMATIVE_TOL = 1e-9           # |d*| above this counts as an informative pair


@dataclass
class Queries:
    S: torch.Tensor              # [n, 28] observation + original goal
    a_pi: np.ndarray             # [n, 4]
    snaps: list
    goals: np.ndarray            # [n, 3]
    episode: np.ndarray
    t: np.ndarray
    phase: np.ndarray


def build_queries(src: core.Source, sim: core.FetchSim) -> Queries:
    rng = np.random.default_rng(QUERY_RNG_SEED)
    T = src.cfg.max_episode_steps
    S, A, snaps, goals, eps, ts = [], [], [], [], [], []
    for ep in range(QUERY_EPISODES):
        chosen = set(int(t) for t in rng.choice(T, size=QUERIES_PER_EPISODE, replace=False))
        o = sim.reset(QUERY_RESET0 + ep)
        goal = o.desired_goal.copy()
        for t in range(T):
            s = np.concatenate([o.observation, goal]).astype(np.float32)
            a = core._policy(src, s)
            if t in chosen:
                S.append(s); A.append(a); snaps.append(sim.snapshot()); goals.append(goal.copy())
                eps.append(ep); ts.append(t)
            o, truncated = sim.step(a)
            if truncated:
                break
    S = np.asarray(S, np.float32)
    ph = src.task.phase_batch(S[:, :src.obs_dim], S[:, src.obs_dim:])
    return Queries(torch.as_tensor(S), np.asarray(A, np.float32), snaps, np.asarray(goals), np.asarray(eps),
                   np.asarray(ts), ph)


def rollout_return(src: core.Source, sim: core.FetchSim, snap, goal: np.ndarray, a_first: np.ndarray,
                   return_rewards: bool = False):
    """G(s, a): restore, take a_first, then follow the frozen actor; sparse reward of the package."""
    sim.restore(snap)
    gamma, thr = src.cfg.gamma, src.success_threshold
    a, G, disc, rs = np.asarray(a_first, np.float32), 0.0, 1.0, []
    for _ in range(HORIZON):
        o = sim.step_raw(a)
        r = 0.0 if np.linalg.norm(o.achieved_goal - goal) <= thr else -1.0
        rs.append(r)
        G += disc * r
        disc *= gamma
        a = core._policy(src, np.concatenate([o.observation, goal]).astype(np.float32))
    return (G, rs) if return_rewards else G


def directions(n: int) -> np.ndarray:
    z = np.random.default_rng(DIR_RNG_SEED).normal(size=(n, N_DIRS, 4))
    return (z / np.linalg.norm(z, axis=-1, keepdims=True)).astype(np.float32)


def perturbed_actions(q: Queries, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[n, n_sigma, J, 4] clipped a^pi + sigma z and a^pi - sigma z."""
    a = q.a_pi[:, None, None, :]
    sig = np.asarray(SIGMAS, np.float32)[None, :, None, None]
    z = Z[:, None, :, :]
    return np.clip(a + sig * z, -1, 1), np.clip(a - sig * z, -1, 1)


def ground_truth(src: core.Source, sim: core.FetchSim, q: Queries, Ap: np.ndarray, Am: np.ndarray) -> dict:
    n = len(q.a_pi)
    G0 = np.array([rollout_return(src, sim, q.snaps[i], q.goals[i], q.a_pi[i]) for i in range(n)])
    Gp = np.zeros(Ap.shape[:3])
    Gm = np.zeros(Ap.shape[:3])
    for i in range(n):
        for k in range(len(SIGMAS)):
            for j in range(N_DIRS):
                Gp[i, k, j] = rollout_return(src, sim, q.snaps[i], q.goals[i], Ap[i, k, j])
                Gm[i, k, j] = rollout_return(src, sim, q.snaps[i], q.goals[i], Am[i, k, j])
    return dict(G0=G0, Gp=Gp, Gm=Gm, dstar=Gp - Gm)


def branch_critics(src: core.Source, batches: list, seed: int) -> dict[str, torch.nn.Module]:
    """Exactly the loop of core.run_branches (same arms, rng streams and batch order), returning
    the critic heads after K_BRANCH steps instead of policy-query risks."""
    run_arms = ["uniform", "saber", "inverse", "td_topk"] + [f"random{j}" for j in range(core.N_RANDOM_SEQUENCES)]
    core.require(tuple(run_arms) == ARMS and len(batches) == K_BRANCH, "branch arms/batches differ from run_branches")
    out = {}
    for arm in run_arms:
        critic = copy.deepcopy(src.heads)
        critic.requires_grad_(True)
        critic.train()
        opt = torch.optim.Adam(critic.parameters(), lr=src.cfg.critic_lr)
        opt.load_state_dict(copy.deepcopy(src.agent.critic_opt.state_dict()))
        rng = np.random.default_rng([seed, run_arms.index(arm)])
        for bt in batches:
            res = critic(bt.S, bt.A) - bt.y
            n_h = res.shape[0]
            rank = res[0].detach().numpy() if n_h == 1 else res.detach().abs().min(0).values.numpy()
            base = "random" if arm.startswith("random") else arm
            mask = core.arm_mask(base, bt, rank, rng)
            w = torch.as_tensor(core.weights(mask))
            loss = sum((w * res[h].pow(2)).mean() for h in range(n_h))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), core.CLIP)
            opt.step()
        critic.eval()
        critic.requires_grad_(False)
        out[arm] = critic
    return out


def critic_signal(critic: torch.nn.Module, q: Queries, Ap: np.ndarray, Am: np.ndarray) -> dict:
    """Head-0 quantities the actor would read: Q differences on the perturbation pairs and
    grad_a Q at a^pi."""
    n, ns, J, _ = Ap.shape
    S_rep = q.S[:, None, None, :].expand(n, ns, J, q.S.shape[1]).reshape(-1, q.S.shape[1])
    with torch.no_grad():
        Qp = critic(S_rep, torch.as_tensor(Ap.reshape(-1, 4)))[0].reshape(n, ns, J).numpy().astype(np.float64)
        Qm = critic(S_rep, torch.as_tensor(Am.reshape(-1, 4)))[0].reshape(n, ns, J).numpy().astype(np.float64)
        Q0 = critic(q.S, torch.as_tensor(q.a_pi))[0].numpy().astype(np.float64)
    a = torch.as_tensor(q.a_pi).clone().requires_grad_(True)
    with torch.enable_grad():
        g = torch.autograd.grad(critic(q.S, a)[0].sum(), a)[0].numpy().astype(np.float64)
    return dict(dq=Qp - Qm, grad=g, Q0=Q0)


def ascent_actions(q: Queries, grad: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(grad, axis=1, keepdims=True)
    u = np.where(nrm > 0, grad / np.maximum(nrm, 1e-30), 0.0)
    return np.clip(q.a_pi + ASCENT_EPS * u, -1, 1).astype(np.float32)


def summarise(sig: dict, truth: dict, Z: np.ndarray, ascent_gain: np.ndarray) -> dict:
    """Per-critic scalar summaries. Pair accuracy is over informative pairs only."""
    out = {}
    for k, s in enumerate(SIGMAS):
        d, dq = truth["dstar"][:, k, :], sig["dq"][:, k, :]
        inf = np.abs(d) > INFORMATIVE_TOL
        out[f"n_inf_s{s}"] = int(inf.sum())
        out[f"PA_s{s}"] = float(np.mean(np.sign(dq[inf]) == np.sign(d[inf]))) if inf.any() else float("nan")
        # |d*|-weighted agreement and the pairwise correlation (all pairs)
        out[f"PAw_s{s}"] = (float(np.sum(np.abs(d[inf]) * (np.sign(dq[inf]) == np.sign(d[inf]))) / np.sum(np.abs(d[inf])))
                            if inf.any() else float("nan"))
        out[f"corr_s{s}"] = (float(np.corrcoef(d.ravel(), dq.ravel())[0, 1])
                             if np.std(d) > 0 and np.std(dq) > 0 else float("nan"))
    # smoothed true gradient (sigma primary) vs grad_a Q at a^pi
    gstar = np.einsum("nj,njd->nd", truth["dstar"][:, 0, :], Z)
    ok = np.linalg.norm(gstar, axis=1) > 0
    g = sig["grad"]
    cos = np.sum(g * gstar, 1) / np.maximum(np.linalg.norm(g, axis=1) * np.linalg.norm(gstar, axis=1), 1e-30)
    out["n_gstar"] = int(ok.sum())
    out["GC"] = float(np.mean(cos[ok])) if ok.any() else float("nan")
    out["grad_norm_median"] = float(np.median(np.linalg.norm(g, axis=1)))
    out["AI"] = float(np.mean(ascent_gain))
    out["AI_pos"] = float(np.mean(ascent_gain > INFORMATIVE_TOL))
    out["AI_neg"] = float(np.mean(ascent_gain < -INFORMATIVE_TOL))
    return out

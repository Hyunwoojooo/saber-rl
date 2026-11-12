"""Feasibility probe: first-order usefulness u_i = <grad R_pi, grad l_i> on one A1 checkpoint.

Not an experiment result. Checks that (1) the jvp shortcut equals explicit per-sample
gradients, (2) the first-order prediction matches a tiny-step SGD branch, (3) masks match
production, and (4) the cost per checkpoint is small.

usage: python u_probe.py <checkpoint.pt> [n_batches] [n_query_episodes]
"""
from __future__ import annotations

import copy
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, jvp

A1 = Path("/home/mimiclab-gigabyte-gen5/Hyunwoo/saber-rl/ICML2027/experiments/draweropen_ddpg_actor_dynamics_20261031")
sys.path.insert(0, str(A1 / "code" / "src"))
from metacog_lite_v0.agent import DDPGAgent  # noqa: E402
from metacog_lite_v0.config import LiteConfig, SeedConfig  # noqa: E402
from metacog_lite_v0.replay import HERReplayBuffer  # noqa: E402
from metacog_lite_v0.trainer import make_env, make_task_spec  # noqa: E402
from metacog_lite_v0.weights import GapStats, LiteWeightComputer  # noqa: E402

torch.set_num_threads(4)
t0 = time.perf_counter()
ckpt = Path(sys.argv[1])
n_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 8
n_query_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 10

payload = torch.load(ckpt, map_location="cpu", weights_only=False)
raw = payload["config"]
known = {f.name for f in fields(LiteConfig)}
cfg = LiteConfig(**{k: copy.deepcopy(v) for k, v in raw.items() if k in known and k != "seeds"},
                 seeds=SeedConfig(**raw["seeds"]))          # effective config: no normalize()
env = make_env(cfg, seed=987_654, render=False)
task = make_task_spec(cfg, env)
agent = DDPGAgent(state_dim=env.state_dim, action_dim=env.action_dim, hidden_dim=cfg.hidden_dim,
                  actor_lr=cfg.actor_lr, critic_lr=cfg.critic_lr, gamma=cfg.gamma, tau=cfg.tau,
                  action_low=env.action_low, action_high=env.action_high, device=torch.device("cpu"),
                  l2_action_penalty=cfg.l2_action_penalty)
agent.load_state_dict(payload["agent"])
r = payload["replay"]
replay = HERReplayBuffer(r["capacity"], r["obs_dim"], r["goal_dim"], r["action_dim"])
replay.load_state_dict(r)
critic = agent.critic
names = [n for n, _ in critic.named_parameters()]
params = {n: p.detach() for n, p in critic.named_parameters()}
t_load = time.perf_counter() - t0


def q_of(p, s, a):
    return functional_call(critic, p, (s, a)).reshape(-1)


def flat(d):
    return torch.cat([d[n].reshape(-1) for n in names])


def unflat(v):
    out, i = {}, 0
    for n in names:
        k = params[n].numel(); out[n] = v[i:i + k].reshape(params[n].shape); i += k
    return out


# ---- policy-query bank: noise-free rollouts, original goal, fixed (frozen-target) Bellman target ----
qs, qa, qy = [], [], []
with torch.no_grad():
    for ep in range(n_query_eps):
        o = env.reset(seed=50_000 + ep)
        for _ in range(cfg.max_episode_steps):
            x = np.concatenate([o.observation, o.desired_goal]).astype(np.float32)
            a = agent.actor(torch.as_tensor(x)[None]).numpy()[0]
            o2, _, term, trunc, _ = env.step(a)
            rew = float(env.compute_reward(o2.achieved_goal[None], o.desired_goal[None], {})[0])
            x2 = np.concatenate([o2.observation, o.desired_goal]).astype(np.float32)
            nonterm = 0.0 if (term and not trunc) else 1.0             # timeout bootstrap
            t2 = torch.as_tensor(x2)[None]
            y = rew + cfg.gamma * nonterm * float(agent.critic_target(t2, agent.actor_target(t2)))
            qs.append(x); qa.append(a); qy.append(y)
            o = o2
            if term or trunc:
                break
QS, QA, QY = (torch.as_tensor(np.asarray(v), dtype=torch.float32) for v in (qs, qa, qy))
t_bank = time.perf_counter() - t0 - t_load


def R(p):
    return (q_of(p, QS, QA) - QY).pow(2).mean()


p_req = {n: v.clone().requires_grad_(True) for n, v in params.items()}
gR = torch.autograd.grad(R(p_req), [p_req[n] for n in names])
v = torch.cat([g.reshape(-1) for g in gR]).detach()
R0 = float(R(params))

# ---- minibatches with production weights ----
wcfg = copy.deepcopy(cfg)
wcfg.support_diagnostic_audit_interval = 0
wc = LiteWeightComputer(wcfg, task, GapStats(8, 128, 0.1, 0.9, 1e-6))
rows = []
check_done = False
for b in range(n_batches):
    rng = np.random.default_rng(900_000 + b)
    batch = replay.sample(cfg.batch_size, cfg.her_ratio, rng)
    rew = env.compute_reward(batch.next_achieved_goal, batch.desired_goal_train, {}).astype(np.float32)
    nonterm = (1.0 - batch.done.astype(np.float32) * (1.0 - batch.timeout.astype(np.float32))).astype(np.float32)
    S = torch.as_tensor(np.concatenate([batch.observation, batch.desired_goal_train], 1), dtype=torch.float32)
    S2 = torch.as_tensor(np.concatenate([batch.next_observation, batch.desired_goal_train], 1), dtype=torch.float32)
    A = torch.as_tensor(batch.action_exec, dtype=torch.float32)
    with torch.no_grad():
        y = torch.as_tensor(rew) + cfg.gamma * torch.as_tensor(nonterm) * agent.critic_target(S2, agent.actor_target(S2)).reshape(-1)
        a_pi = agent.actor(S)
    wb = wc.compute(batch, replay, agent, np.random.default_rng(1), torch.device("cpu"))
    psi = wb.psi_q_rank.astype(bool)
    omega = torch.as_tensor(wb.omega_critic)
    p = psi.mean()
    with torch.no_grad():
        delta = q_of(params, S, A) - y
    _, jv = jvp(lambda pv: q_of(unflat(pv), S, A), (flat(params),), (v,))
    u = (2.0 * delta * jv).numpy()                                   # loss = delta^2 (production)

    if not check_done:                                               # explicit per-sample gradient check
        for i in range(4):
            pr = {n: t.clone().requires_grad_(True) for n, t in params.items()}
            li = (q_of(pr, S[i:i + 1], A[i:i + 1]) - y[i]).pow(2).sum()
            gi = torch.cat([g.reshape(-1) for g in torch.autograd.grad(li, [pr[n] for n in names])])
            assert abs(float(gi @ v) - u[i]) <= 1e-4 * max(1.0, abs(u[i])), (float(gi @ v), u[i])
        check_done = True

    # first-order prediction vs tiny SGD branch (SABER weights vs uniform)
    pred = float(((omega - 1.0).numpy() * u).mean())                 # <grad R, g_S - g_U>
    p64 = {n: t.double() for n, t in params.items()}
    S64, A64, y64 = S.double(), A.double(), y.double()
    QS64, QA64, QY64 = QS.double(), QA.double(), QY.double()

    def sgd(w, eta):
        pr = {n: t.clone().requires_grad_(True) for n, t in p64.items()}
        loss = (w.double() * (q_of(pr, S64, A64) - y64).pow(2)).mean()
        gs = torch.autograd.grad(loss, [pr[n] for n in names])
        pn = {n: p64[n] - eta * g for n, g in zip(names, gs)}
        return float((q_of(pn, QS64, QA64) - QY64).pow(2).mean())
    fds = {eta: (sgd(omega, eta) - sgd(torch.ones_like(omega), eta)) / eta for eta in (1e-6, 1e-7, 1e-8)}
    fd = fds[1e-7]

    # coupling between own replay action and own policy action
    jr = []
    jp = []
    for i in range(cfg.batch_size):
        pr = {n: t.clone().requires_grad_(True) for n, t in params.items()}
        g1 = torch.autograd.grad(q_of(pr, S[i:i + 1], A[i:i + 1]).sum(), [pr[n] for n in names])
        g2 = torch.autograd.grad(q_of(pr, S[i:i + 1], a_pi[i:i + 1]).sum(), [pr[n] for n in names])
        jr.append(torch.cat([g.reshape(-1) for g in g1])); jp.append(torch.cat([g.reshape(-1) for g in g2]))
    jr = torch.stack(jr); jp = torch.stack(jp)
    cos = torch.nn.functional.cosine_similarity(jr, jp, dim=1).numpy()
    dist = torch.linalg.vector_norm(A - a_pi, dim=1).numpy()
    elig = wb.context_count_eligible_mask if hasattr(wb, "context_count_eligible_mask") else None
    rows.append(dict(b=b, p=p, u_pos=u[psi].mean() if psi.any() else np.nan,
                     u_neg=u[~psi].mean() if (~psi).any() else np.nan, pred=pred, fd=fd, fds=fds,
                     cos_pos=cos[psi].mean() if psi.any() else np.nan, cos_neg=cos[~psi].mean(),
                     dist_pos=dist[psi].mean() if psi.any() else np.nan, dist_neg=dist[~psi].mean(),
                     abs_u=np.abs(u).mean()))

t_all = time.perf_counter() - t0
print(f"checkpoint {ckpt.parent.parent.parent.name}/{ckpt.parent.parent.name}/{ckpt.name}  step={payload['global_step']}")
print(f"query bank: {len(QY)} on-policy queries, R0={R0:.4f}, |grad R|={float(v.norm()):.4f}")
print(f"time: load {t_load:.1f}s  bank {t_bank:.1f}s  total {t_all:.1f}s  ({n_batches} batches)")
print(" b     p    u+        u-        pred(<gR,gS-gU>)  -fd/eta     cos+   cos-   |a^r-a^pi|+ -")
for x in rows:
    print(f"{x['b']:2d} {x['p']:.3f} {x['u_pos']:+.2e} {x['u_neg']:+.2e}  {x['pred']:+.3e}   {-x['fd']:+.3e}  "
          f"{x['cos_pos']:.3f}  {x['cos_neg']:.3f}  {x['dist_pos']:.3f} {x['dist_neg']:.3f}   "
          + "  ".join(f"eta={e:g}:{-d:+.3e}" for e, d in x['fds'].items()))


# ---- actual production step: Adam(lr) from saved moments + clip 10, SABER weights vs uniform ----
def adam_branch(w, batch_seed):
    rng = np.random.default_rng(900_000 + batch_seed)
    batch = replay.sample(cfg.batch_size, cfg.her_ratio, rng)
    rew = env.compute_reward(batch.next_achieved_goal, batch.desired_goal_train, {}).astype(np.float32)
    nonterm = (1.0 - batch.done.astype(np.float32) * (1.0 - batch.timeout.astype(np.float32))).astype(np.float32)
    S = torch.as_tensor(np.concatenate([batch.observation, batch.desired_goal_train], 1), dtype=torch.float32)
    S2 = torch.as_tensor(np.concatenate([batch.next_observation, batch.desired_goal_train], 1), dtype=torch.float32)
    A = torch.as_tensor(batch.action_exec, dtype=torch.float32)
    with torch.no_grad():
        y = torch.as_tensor(rew) + cfg.gamma * torch.as_tensor(nonterm) * agent.critic_target(S2, agent.actor_target(S2)).reshape(-1)
    out = {}
    for name in ("uniform", "saber"):
        c = copy.deepcopy(agent.critic); c.requires_grad_(True); c.train()
        opt = torch.optim.Adam(c.parameters(), lr=cfg.critic_lr)
        opt.load_state_dict(copy.deepcopy(agent.critic_opt.state_dict()))
        ww = torch.ones(cfg.batch_size) if name == "uniform" else w
        loss = (ww * (c(S, A).reshape(-1) - y).pow(2)).mean()
        opt.zero_grad(); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(c.parameters(), 10.0))
        opt.step()
        with torch.no_grad():
            out[name] = (float((c(QS, QA).reshape(-1) - QY).pow(2).mean()), gn)
    return out


print("\nactual Adam+clip step (R after; grad-norm before clip):")
for b in range(n_batches):
    rng = np.random.default_rng(900_000 + b)
    batch = replay.sample(cfg.batch_size, cfg.her_ratio, rng)
    wb = wc.compute(batch, replay, agent, np.random.default_rng(1), torch.device("cpu"))
    res = adam_branch(torch.as_tensor(wb.omega_critic), b)
    dU, dS = res["uniform"][0] - R0, res["saber"][0] - R0
    print(f"b={b}  R0={R0:.5f}  dR uniform={dU:+.3e} (|g|={res['uniform'][1]:.1f})  dR saber={dS:+.3e} (|g|={res['saber'][1]:.1f})  "
          f"saber-uniform={dS - dU:+.3e}")

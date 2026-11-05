"""Read-only gradient probes and native-update observations for A1."""
import contextlib
import random
from pathlib import Path
import numpy as np
import torch
from common import JsonLines, sha, write_json
from protocol import PROBE_SIZE, INDIVIDUAL_GRADIENT_SAMPLES, checkpoints, probe_due


@contextlib.contextmanager
def preserve_rng():
    """The experiment runs on CPU; private generators never use training RNGs."""
    py, np_state, th = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(th)


def tensor_norm(tensors):
    # Accumulate in double so tiny float32 gradients do not become zero merely
    # because their squares underflow during diagnostic reduction.
    return float(torch.sqrt(sum(t.detach().double().square().sum() for t in tensors)))


def policy_measure(agent, states, individual_count=INDIVIDUAL_GRADIENT_SAMPLES):
    """autograd.grad never populates .grad or steps an optimizer."""
    x = torch.as_tensor(states, dtype=torch.float32, device=agent.device)
    if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
        raise ValueError("Probe states must be a finite, nonempty matrix")
    params = tuple(agent.actor.parameters())
    with torch.enable_grad():
        logits = agent.actor.net(x)
        actions = torch.tanh(logits)
        q = agent.critic(x, actions)
        if not all(torch.isfinite(v).all() for v in (logits, actions, q)):
            raise FloatingPointError("Nonfinite policy probe output")
        # Sum gives each independent row's dQ_i/da_i without 1/B scaling.
        dqda = torch.autograd.grad(q.sum(), actions, retain_graph=True)[0]
        count = min(individual_count, len(x))
        grads = torch.autograd.grad(-q.mean(), params, retain_graph=count > 0)
        individual = []
        for i in range(count):
            gi = torch.autograd.grad(-q[i], params, retain_graph=i+1 < count)
            individual.append(tensor_norm(gi))
    if not torch.isfinite(dqda).all() or not all(torch.isfinite(g).all() for g in grads):
        raise FloatingPointError("Nonfinite policy probe gradient")
    a, z, dq = actions.detach(), logits.detach(), dqda.detach()
    derivative = 1-a.square()
    row_norm = torch.linalg.vector_norm(dq.double(), dim=1)
    row = dict(query_count=len(x), mean_abs_pre_tanh=float(z.abs().mean()),
        max_abs_pre_tanh=float(z.abs().max()), action_abs_ge_0999_fraction=float((a.abs() >= .999).float().mean()),
        exact_boundary_fraction=float((a.abs() == 1).float().mean()),
        tanh_derivative_exact_zero_fraction=float((derivative == 0).float().mean()),
        tanh_derivative_lt_1e6_fraction=float((derivative < 1e-6).float().mean()),
        all_action_components_saturated_query_fraction=float((derivative == 0).all(dim=1).float().mean()),
        critic_action_gradient_l2_mean=float(row_norm.mean()),
        critic_action_gradient_zero_query_fraction=float((row_norm == 0).double().mean()),
        actor_batch_gradient_l2=tensor_norm(grads),
        individual_gradient_query_count=count,
        individual_actor_gradient_l2_mean=float(np.mean(individual)) if count else None,
        individual_actor_gradient_zero_fraction=float(np.mean(np.array(individual) == 0)) if count else None,
        q_mean=float(q.detach().mean()), q_min=float(q.detach().min()), q_max=float(q.detach().max()))
    return row, dq.cpu().numpy().copy()


def make_probe_bank(config):
    """32 fixed original/future-goal queries from four separate random rollouts.

    Captures the actions and reset seeds for reproducibility. These observations
    are not simulator snapshots and are not advertised as restorable states.
    """
    from metacog_lite_v0.trainer import make_env
    rows, original, her, actions_all, reset_seeds, ep_ids, times = [], [], [], [], [], [], []
    positions = (0, 20, 40, 60, 80, 100, 140, 180)
    with preserve_rng():
        rng = np.random.default_rng(700_000 + config.seeds.env_seed)
        env = make_env(config, 600_000 + config.seeds.env_seed, False)
        try:
            for episode in range(4):
                reset_seed = 600_000 + config.seeds.env_seed*10 + episode
                reset_seeds.append(reset_seed)
                obs = env.reset(seed=reset_seed)
                before, following, goals = [], [], []
                for t in range(200):
                    before.append(obs.observation.copy())
                    goals.append(obs.desired_goal.copy())
                    action = rng.uniform(env.action_low, env.action_high).astype(np.float32)
                    actions_all.append(action.copy())
                    obs, _, terminated, truncated, _ = env.step(action)
                    following.append(obs.achieved_goal.copy())
                    if terminated or truncated:
                        if t != 199:
                            raise RuntimeError("Probe environment ended before its declared 200-step horizon")
                for t in positions:
                    future_stored_index = int(rng.integers(t+1, 200))
                    rows.append(before[t]); original.append(goals[t]); her.append(following[future_stored_index])
                    ep_ids.append(episode); times.append(t)
        finally:
            env.close()
    bank = dict(observation=np.asarray(rows,dtype=np.float32), original_goal=np.asarray(original,dtype=np.float32),
                her_goal=np.asarray(her,dtype=np.float32), actions=np.asarray(actions_all,dtype=np.float32),
                reset_seeds=np.asarray(reset_seeds), episode=np.asarray(ep_ids), time=np.asarray(times))
    assert len(bank["observation"]) == PROBE_SIZE
    return bank


class Observer:
    def __init__(self, arm, seed, profile, attempt_dir):
        self.arm, self.seed, self.profile = arm, seed, profile
        self.root = Path(attempt_dir)/"diagnostics"
        self.root.mkdir(exist_ok=False)
        self.streams = {name:JsonLines(self.root/(name+".jsonl")) for name in
                        ("probes", "updates", "update_windows", "training_windows", "evaluation", "events", "checkpoints")}
        self.current_step = 0
        self.update_window, self.transition_window = [], []
        self.ep_goal = self.ep_env = 0.0
        self.first_goal_step = self.first_env_step = None
        self.active = False
        self.saved_steps = []
        self.sampled_updates = 0
        self.closed = False

    def emit(self, stream, **row):
        self.streams[stream].write(dict(arm=self.arm, seed=self.seed, step=self.current_step, **row))

    def start(self, agent, task, replay, run_dir, config):
        self.cfg, self.replay, self.task = config, replay, task
        self.bank = make_probe_bank(config)
        np.savez_compressed(self.root/"fixed_probe_bank.npz", **self.bank)
        obs, goal, hg = self.bank["observation"], self.bank["original_goal"], self.bank["her_goal"]
        self.fixed = {"fixed_original":np.concatenate([obs,goal],axis=1),
                      "fixed_her":np.concatenate([obs,hg],axis=1)}
        write_json(self.root/"probe_bank.json", dict(query_count=len(obs), probe_environment_steps=800,
            bank_sha256=sha(self.root/"fixed_probe_bank.npz"), observation_dim=obs.shape[1],
            native_tail_original_goal_max_abs_difference=float(np.max(np.abs(obs[:,-3:]-goal))),
            native_tail_her_goal_max_abs_difference=float(np.max(np.abs(obs[:,-3:]-hg))),
            note="Tail agreement is a representation diagnostic, not evidence of causal harm; no input is changed.",
            simulator_state_serialized=False, training_streams_used=False))

    def probe(self, agent, states, source, phase):
        metrics, dq = policy_measure(agent, states)
        self.emit("probes", query_source=source, phase=phase, **metrics)
        return dq

    def before_update(self, step, agent, batch, wb):
        self.current_step = step
        self.active = probe_due(step, self.cfg.total_steps)
        if not self.active:
            return
        self.actor_before = [p.detach().clone() for p in agent.actor.parameters()]
        self.critic_before = [p.detach().clone() for p in agent.critic.parameters()]
        n = min(PROBE_SIZE,len(batch.indices))
        obs = batch.observation[:n]
        original = self.replay.desired_goal[batch.indices[:n]].copy()
        self.live = {"training_her":np.concatenate([obs,batch.desired_goal_train[:n]],axis=1),
                     "training_original":np.concatenate([obs,original],axis=1)}
        self.pre_direction = {}
        for key,states in self.live.items():
            self.pre_direction[key] = self.probe(agent,states,key,"before_critic")

    def after_critic(self, agent, state):
        if not self.active:
            return
        for key,states in self.live.items():
            dq = self.probe(agent,states,key,"after_critic_before_actor")
            a, b = self.pre_direction[key].ravel().astype(np.float64), dq.ravel().astype(np.float64)
            norm_product = np.linalg.norm(a)*np.linalg.norm(b)
            self.emit("updates", measurement="critic_direction_change", query_source=key,
                critic_action_gradient_cosine=float(a@b/norm_product) if norm_product > 0 else None,
                critic_action_gradient_delta_l2=float(np.linalg.norm(b-a)))

    def after_update(self, step, agent, metrics, selector, wb):
        omega = np.asarray(wb.omega_critic)
        if not np.isfinite(omega).all() or np.any(omega <= 0):
            raise FloatingPointError("Invalid critic coefficients")
        if not np.isclose(omega.mean(), 1, atol=2e-6):
            raise RuntimeError("Critic coefficient mean changed")
        if not np.array_equal(wb.omega_actor, np.ones_like(wb.omega_actor)):
            raise RuntimeError("Actor coefficients changed")
        if self.arm == "her" and not np.array_equal(omega,np.ones_like(omega)):
            raise RuntimeError("HER arm received nonuniform coefficients")
        s = selector.stats
        row = dict(step=step, update=int(agent.total_it),
            native_actor_preclip_gradient_norm=metrics.actor_grad_norm,
            native_critic_preclip_gradient_norm=metrics.critic_grad_norm,
            actor_gradient_zero=float(metrics.actor_grad_norm == 0),
            actor_gradient_clipped=float(metrics.actor_grad_norm > 10),
            eligible_rate=s["eligible_rate"], support_rate=s["support_rate"],
            assignment_informative=s["informative_update"],
            assignment_saturated_nonempty=s["saturated_nonempty_update"],
            critic_coefficient_variance=float(np.var(omega)),
            actual_high_weight_fraction=float(np.mean(wb.w_p_raw > 1)),
            critic_loss=metrics.critic_loss, actor_loss=metrics.actor_loss)
        if not all(np.isfinite(v) for v in row.values()):
            raise FloatingPointError("Nonfinite native-update observation")
        self.update_window.append(row)
        if self.active:
            actor_move = [p.detach().double()-b.double() for p,b in zip(agent.actor.parameters(),self.actor_before)]
            critic_move = [p.detach().double()-b.double() for p,b in zip(agent.critic.parameters(),self.critic_before)]
            data = dict(row)
            data.pop("step")
            self.emit("updates", measurement="native_optimizer_update", actor_parameter_delta_l2=tensor_norm(actor_move),
                critic_parameter_delta_l2=tensor_norm(critic_move), **data)
            self.sampled_updates += 1
            self.actor_before = self.critic_before = None
        limit = 100 if step <= 10000 else 1000
        if len(self.update_window) >= limit:
            self.flush_updates()

    def flush_updates(self):
        if not self.update_window:
            return
        rows = self.update_window
        means = {k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("step","update")}
        self.emit("update_windows", start_step=rows[0]["step"], end_step=rows[-1]["step"],
            start_update=rows[0]["update"], end_update=rows[-1]["update"], n_updates=len(rows), means=means)
        self.update_window = []

    def on_transition(self, step, obs, action, info, done, episode):
        self.current_step = step
        gs, es = float(info["goal_success"]), float(info["env_success"])
        self.ep_goal, self.ep_env = max(self.ep_goal,gs), max(self.ep_env,es)
        for label,value in (("goal",gs),("env",es)):
            attr = "first_"+label+"_step"
            if value > 0 and getattr(self,attr) is None:
                setattr(self,attr,step)
                self.emit("events", event="first_training_"+label+"_success", episode=episode)
        ee, handle = obs.observation[:3], obs.observation[4:7]
        self.transition_window.append(dict(step=step,action=np.asarray(action).copy(),
            hand_handle_distance=float(np.linalg.norm(ee-handle)), goal_success=gs, env_success=es))
        if done:
            self.emit("events", event="training_episode_end", episode=episode,
                goal_success_ever=self.ep_goal, env_success_ever=self.ep_env)
            self.ep_goal = self.ep_env = 0.0
        if len(self.transition_window) >= 1000:
            self.flush_transitions()

    def flush_transitions(self):
        if not self.transition_window:
            return
        rows = self.transition_window
        a = np.asarray([r["action"] for r in rows],dtype=np.float64)
        bins = np.clip(np.floor((a+1)*2.5).astype(int),0,4)
        self.emit("training_windows", start_step=rows[0]["step"], end_step=rows[-1]["step"],
            n_transitions=len(rows), action_mean=a.mean(axis=0).tolist(), action_std=a.std(axis=0).tolist(),
            unique_executed_action_bins=len(set(map(tuple,bins))),
            hand_handle_distance_mean=float(np.mean([r["hand_handle_distance"] for r in rows])),
            hand_handle_distance_min=float(min(r["hand_handle_distance"] for r in rows)),
            original_goal_success_transition_fraction=float(np.mean([r["goal_success"] for r in rows])))
        self.transition_window = []

    def on_evaluation(self, agent, observations, goals, metrics):
        clean = {k:float(v) if np.isfinite(v) else None for k,v in metrics.items()}
        self.emit("evaluation", geometry_source="evaluation_observations_post_transition", metrics=clean)
        indices = np.linspace(0,len(observations)-1,PROBE_SIZE,dtype=int)
        states = np.concatenate([np.asarray(observations)[indices],np.asarray(goals)[indices]],axis=1)
        self.probe(agent,states,"evaluation_visited_original","after_training_update")

    def on_step_end(self, step, agent, replay, payload_fn):
        self.current_step = step
        if probe_due(step,self.cfg.total_steps):
            for key,states in self.fixed.items():
                self.probe(agent,states,key,"step_end")
        if step in checkpoints(self.cfg.total_steps):
            payload = payload_fn(step,"a1_scheduled")
            if not payload["episode_state"]["at_episode_boundary"]:
                raise RuntimeError("Scheduled checkpoint is not at an episode boundary")
            payload["a1_metadata"] = dict(arm=self.arm,seed=self.seed,profile=self.profile,
                simulator_state_serialized=False,online_continuation_supported=False,
                fixed_probe_bank_sha256=sha(self.root/"fixed_probe_bank.npz"))
            dest = self.root.parent/"scheduled_checkpoints"
            dest.mkdir(exist_ok=True)
            path = dest/f"step_{step:07d}.pt"
            if path.exists():
                raise FileExistsError(path)
            temp = path.with_suffix(".tmp")
            torch.save(payload,temp)
            temp.replace(path)
            self.saved_steps.append(step)
            self.emit("checkpoints", file=str(path.relative_to(self.root.parent)),sha256=sha(path),
                total_updates=int(agent.total_it), replay_size=int(replay.size),bytes=path.stat().st_size)

    def close(self):
        if self.closed:
            return
        self.flush_updates()
        self.flush_transitions()
        for writer in self.streams.values():
            writer.close()
        self.closed = True
        write_json(self.root/"summary.json",dict(saved_steps=self.saved_steps,sampled_updates=self.sampled_updates,
            first_training_goal_success_step=self.first_goal_step,first_training_env_success_step=self.first_env_step,
            probe_rng_independent=True,policy_query_true_return_measured=False))

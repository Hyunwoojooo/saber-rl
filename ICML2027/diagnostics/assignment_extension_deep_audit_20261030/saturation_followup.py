"""Additional independent batches and optimizer-moment inspection for failed SABER."""
import gc
import numpy as np
import torch
from scipy.stats import norm
from common import *
from checkpoint_analysis import load


def analyze():
 torch.set_num_threads(1);results=[];moments=[]
 for seed in (13,14,15,16,18,20):
  cfg,task,replay,agent,step=load('support',seed,'final')
  state=agent.actor_opt.state_dict()['state']
  moments.append(dict(seed=seed,actor_adam_exp_avg_abs_max=max(float(s['exp_avg'].abs().max()) for s in state.values()),
       actor_adam_exp_avg_sq_abs_max=max(float(s['exp_avg_sq'].abs().max()) for s in state.values())))
  for repeat in range(4):
   batch=replay.sample(256,cfg.her_ratio,np.random.default_rng(2026103100+seed*10+repeat))
   for kind in ('her','original'):
    goals=batch.desired_goal_train if kind=='her' else replay.desired_goal[batch.indices]
    x=torch.from_numpy(np.concatenate([batch.observation,goals],axis=1).astype(np.float32))
    a=agent.actor(x);loss=-agent.critic(x,a).mean();grads=torch.autograd.grad(loss,tuple(agent.actor.parameters()))
    results.append(dict(seed=seed,repeat=repeat,goal_kind=kind,queries=256,
       actor_gradient_l2=float(torch.sqrt(sum(g.square().sum() for g in grads))),
       exact_boundary_fraction=float((a.detach().abs()==1).float().mean()),
       positive_z_fraction=float((a.detach()[:,2]==1).float().mean())))
  print('SATURATION_FOLLOWUP',seed,'zero gradient batches',sum(r['actor_gradient_l2']==0 for r in results if r['seed']==seed),'/8',flush=True)
  del cfg,task,replay,agent;gc.collect()
 table(OUT/'saturation_followup.csv',results);table(OUT/'actor_optimizer_moments.csv',moments)
 write(OUT/'exploration_calculation.json',dict(conditional_assumption='deterministic z action exactly +1; additive independent Normal(0,0.2^2), clipped to [-1,1]',
       probability_noisy_z_negative=float(norm.cdf(-5)),mean_steps_per_negative_z=1/float(norm.cdf(-5)),
       probability_behavior_same_4d_corner_bin_under_K5=float(norm.cdf(2)**4),
       interpretation='Conditional calculation only; does not assert the policy was saturated for the entire historical training trajectory.'))

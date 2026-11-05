"""Post-hoc saturation follow-up; gradients measured, no optimizer steps."""
import dataclasses,gc
import numpy as np
import torch
from common import *
from checkpoint_analysis import load


def analyze():
 torch.set_num_threads(1);results=[]
 for arm,seed,kind in [(a,s,'final') for a in ARMS for s in SEEDS]+[('support',s,'best') for s in (13,14,15,16,18,20)]:
  cfg,task,replay,agent,step=load(arm,seed,kind)
  batch=replay.sample(256,cfg.her_ratio,np.random.default_rng(2026103000+seed))
  before={k:v.clone() for k,v in agent.actor.state_dict().items()};updates=agent.total_it
  for goal_kind in ('her','original'):
   goals=batch.desired_goal_train if goal_kind=='her' else replay.desired_goal[batch.indices]
   states=torch.from_numpy(np.concatenate([batch.observation,goals],axis=1).astype(np.float32))
   logits=agent.actor.net(states);actions=torch.tanh(logits)
   values=agent.critic(states,actions);loss=-values.mean()
   action_grad=torch.autograd.grad(loss,actions,retain_graph=True)[0]
   parameters=tuple(agent.actor.parameters())
   grads=torch.autograd.grad(loss,parameters)
   total_norm=torch.sqrt(sum(g.square().sum() for g in grads))
   derivative=1-actions.detach().square()
   results.append(dict(arm=arm,seed=seed,checkpoint=kind,checkpoint_step=step,goal_kind=goal_kind,
        actor_gradient_l2=float(total_norm.item()),action_gradient_l2=float(torch.linalg.vector_norm(action_grad).item()),
        mean_abs_pre_tanh=float(logits.detach().abs().mean().item()),max_abs_pre_tanh=float(logits.detach().abs().max().item()),
        tanh_derivative_zero_fraction=float((derivative==0).float().mean().item()),
        tanh_derivative_lt1e6_fraction=float((derivative<1e-6).float().mean().item()),
        exact_boundary_action_fraction=float((actions.detach().abs()==1).float().mean().item()),
        action_saturated_component_fraction=float((actions.detach().abs()>.99).float().mean().item()),
        query_count=256,network_unchanged=True,optimizer_steps=0))
  require(agent.total_it==updates and all(torch.equal(v,agent.actor.state_dict()[k]) for k,v in before.items()),'Gradient audit changed network')
  print('GRADIENT',arm,seed,kind,'actor norms',[round(r['actor_gradient_l2'],8) for r in results[-2:]],
        'zero tanh derivative',[round(r['tanh_derivative_zero_fraction'],3) for r in results[-2:]],flush=True)
  del cfg,task,replay,agent;gc.collect()
 table(OUT/'gradient_summary.csv',results)
 write(OUT/'gradient_audit.json',dict(checkpoints=36,goal_kinds_per_checkpoint=2,total_queries=18432,
       optimizer_updates=0,post_hoc_added_after_rollout_saturation_observed=True))

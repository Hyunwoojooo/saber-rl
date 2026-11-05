"""Read-only checkpoint support parity, replay integrity and policy replay."""
import dataclasses,gc,hashlib,sys,time
import numpy as np
import torch
from common import *

sys.path.insert(0,str(STUDY/'study/code/src'))
from metacog_lite_v0.config import LiteConfig,SeedConfig
from metacog_lite_v0.agent import DDPGAgent
from metacog_lite_v0.replay import HERReplayBuffer
from metacog_lite_v0.task_drawer_open import MetaWorldDrawerOpenTaskSpec
from metacog_lite_v0.weights import LiteWeightComputer,GapStats
from metacog_lite_v0.trainer import make_env


def restore_config(saved):
 fields={f.name for f in dataclasses.fields(LiteConfig)}
 require(set(saved)-fields <= {'thread_hint','terminal_convention','resolved_env_id'},'Unknown saved config field')
 require(fields<=set(saved),'Missing saved config field')
 args={k:v for k,v in saved.items() if k in fields};args['seeds']=SeedConfig(**args['seeds'])
 cfg=LiteConfig(**args);cfg.device='cpu';return cfg


def load(arm,seed,kind):
 d=attempt(arm,seed);path=d/'run'/('checkpoint.pt' if kind=='final' else 'best_checkpoint.pt')
 record(path);saved=torch.load(path,map_location='cpu',weights_only=False)
 config=read(d/'run/config.json');require(saved['config']==config,'Checkpoint config differs')
 require(saved['assignment_rule']==arm and saved['resume_audit']['resumed_in_this_process'] is False,'Checkpoint identity differs')
 require(saved['total_updates']==saved['global_step']-4999,'Checkpoint budget differs')
 cfg=restore_config(config);rp=saved['replay'];n=rp['size']
 replay=HERReplayBuffer(capacity=rp['capacity'],obs_dim=rp['obs_dim'],goal_dim=rp['goal_dim'],action_dim=rp['action_dim'])
 replay.load_state_dict(rp)
 env=read(d/'run/env_snapshot.json')['train_env']
 task=MetaWorldDrawerOpenTaskSpec(action_low=np.array(env['action_low'],np.float32),action_high=np.array(env['action_high'],np.float32),
      epsilon_g=env['distance_threshold'],d_reach=cfg.drawer_d_reach,d_goal_near=cfg.drawer_d_goal_near,d_opening_mid=cfg.drawer_d_opening_mid,
      hand_handle_bins=tuple(cfg.drawer_hand_handle_bins),handle_goal_bins=tuple(cfg.drawer_handle_goal_bins),handle_y_bins=tuple(cfg.drawer_handle_y_bins),
      gripper_bins=tuple(cfg.drawer_gripper_bins),horizon_bins=tuple(cfg.drawer_horizon_bins))
 keys=task.coarse_hash_batch(replay.observation[:n],replay.h[:n]);bins=task.action_bin_batch(replay.action_exec[:n],cfg.action_bins_per_dim)
 require(np.array_equal(keys,replay.action_index._key_by_slot[:n]),'Coarse index slots differ')
 require(np.array_equal(bins,replay.action_bin[:n]),'Replay action bins differ')
 for key in np.unique(keys):
  slots=replay.action_index.candidates(int(key),n);expected=np.flatnonzero(keys==key)
  require(np.array_equal(np.sort(slots),expected),'Coarse index membership differs')
  b,c=np.unique(bins[expected],return_counts=True)
  actual=replay.action_index._action_bin_counts_by_key.get(int(key),{})
  require({int(k):int(v) for k,v in actual.items()}=={int(k):int(v) for k,v in zip(b,c)},'Action-bin count index differs')
 agent=DDPGAgent(state_dim=env['state_dim'],action_dim=env['action_dim'],hidden_dim=cfg.hidden_dim,actor_lr=cfg.actor_lr,critic_lr=cfg.critic_lr,
       gamma=cfg.gamma,tau=cfg.tau,action_low=task.action_low,action_high=task.action_high,device=torch.device('cpu'),l2_action_penalty=cfg.l2_action_penalty)
 agent.load_state_dict(saved['agent'])
 for model in (agent.actor,agent.actor_target,agent.critic,agent.critic_target):model.eval()
 return cfg,task,replay,agent,saved['global_step']


def literal_support(task,replay,selected,goal,full_key,policy_bin,cfg):
 selected=np.asarray(selected,dtype=np.int64)
 if len(selected)==0:return False,False,0
 keys=task.full_context_hash_batch(replay.observation[selected],np.broadcast_to(goal,(len(selected),len(goal))),replay.h[selected])
 local=selected[keys==full_key];n=len(local)
 eligible=n>=cfg.N_local_min and min(n,cfg.M_B)>=cfg.M_Q_min
 return bool(eligible),bool(eligible and np.any(replay.action_bin[local]==policy_bin)),n


def probe(cfg,task,replay,agent,arm,seed,kind):
 batch=replay.sample(256,cfg.her_ratio,np.random.default_rng(2026103000+seed))
 result=[];summary=[]
 for goal_kind in ('her','original'):
  goals=batch.desired_goal_train if goal_kind=='her' else replay.desired_goal[batch.indices].copy()
  b=dataclasses.replace(batch,desired_goal_train=goals)
  states=torch.from_numpy(np.concatenate([b.observation,goals],axis=1).astype(np.float32))
  ns=torch.from_numpy(np.concatenate([b.next_observation,goals],axis=1).astype(np.float32))
  with torch.no_grad():
   actions=agent.policy_actions(states).numpy();q=agent.q_values(states,torch.from_numpy(actions)).numpy().reshape(-1)
   replay_q=agent.q_values(states,torch.from_numpy(b.action_exec)).numpy().reshape(-1)
   target_q=agent.target_q_values(ns).numpy().reshape(-1)
  reward=np.where(np.linalg.norm(b.next_achieved_goal-goals,axis=1)<=cfg.drawer_success_threshold,0.,-1.)
  target=reward+cfg.gamma*(1-b.done*(1-b.timeout))*target_q;td=np.abs(replay_q-target)
  computer=LiteWeightComputer(cfg,task,GapStats(cfg.gap_window,cfg.N_gap_min,cfg.gap_q_low,cfg.gap_q_high,cfg.eps_G))
  wb=computer.compute(b,replay,agent,np.random.default_rng(700+seed),torch.device('cpu'),assignment_rng=np.random.default_rng(seed))
  wb_repeat=computer.compute(b,replay,agent,np.random.default_rng(700+seed),torch.device('cpu'),assignment_rng=np.random.default_rng(seed))
  require(np.array_equal(wb.support_mask,wb_repeat.support_mask) and np.array_equal(wb.context_eligible_mask,wb_repeat.context_eligible_mask),'Cache repeat changes support')
  require(np.array_equal(wb.w_p_raw,1+wb.support_mask.astype(np.float32)),'Native raw weights differ')
  require(np.allclose(wb.omega_critic,wb.w_p_raw/wb.w_p_raw.mean(),atol=2e-6) and np.all(wb.omega_actor==1),'Wrong normalized critic-only weights')
  fk=task.full_context_hash_batch(b.observation,goals,b.h);ck=task.coarse_hash_batch(b.observation,b.h);pb=task.action_bin_batch(actions,cfg.action_bins_per_dim)
  for i in range(256):
   candidates=replay.action_index.candidates(int(ck[i]),replay.size);tail=candidates[-cfg.local_search_max:]
   eligible,support,count=literal_support(task,replay,tail,goals[i],fk[i],pb[i],cfg)
   require(eligible==bool(wb.context_eligible_mask[i]) and support==bool(wb.support_mask[i]),f'Native/reference parity failed: {arm}/{seed}/{kind}/{goal_kind}/{i}')
   full=loo=None
   if i<8:
    full=literal_support(task,replay,candidates,goals[i],fk[i],pb[i],cfg)[1]
    loo=literal_support(task,replay,tail[tail!=b.indices[i]],goals[i],fk[i],pb[i],cfg)[1]
   result.append(dict(arm=arm,seed=seed,checkpoint=kind,goal_kind=goal_kind,query=i,replay_slot=int(b.indices[i]),
         native_support=int(support),eligible=int(eligible),local_tail_count=count,full_support=full,leave_one_out_support=loo,
         q_policy=float(q[i]),q_replay=float(replay_q[i]),abs_td=float(td[i]),positive_reward=int(reward[i]==0),
         phase=int(task.phase_batch(b.observation[i:i+1],goals[i:i+1])[0]),
         action_abs_max=float(np.max(np.abs(actions[i]))),action_distance=float(np.linalg.norm(actions[i]-b.action_exec[i])),
         goal_changed=int(np.linalg.norm(goals[i]-replay.desired_goal[b.indices[i]])>1e-8)))
  mask=wb.support_mask
  original_goal=replay.desired_goal[b.indices]
  summary.append(dict(arm=arm,seed=seed,checkpoint=kind,goal_kind=goal_kind,queries=256,
       support_rate=float(mask.mean()),eligible_rate=float(wb.context_eligible_mask.mean()),mean_weight=float(wb.omega_critic.mean()),
       positive_reward_fraction=float(np.mean(reward==0)),positive_reward_fraction_high=float(np.mean(reward[mask]==0)) if mask.any() else None,
       positive_reward_fraction_low=float(np.mean(reward[~mask]==0)) if (~mask).any() else None,
       td_high=float(td[mask].mean()) if mask.any() else None,td_low=float(td[~mask].mean()) if (~mask).any() else None,
       q_policy_mean=float(q.mean()),q_policy_positive_fraction=float(np.mean(q>1e-5)),q_policy_below_minus50_fraction=float(np.mean(q<-50.00001)),
       action_saturated_component_fraction=float(np.mean(np.abs(actions)>.99)),
       observation_tail_matches_original_goal=bool(np.allclose(b.observation[:,-3:],original_goal,atol=1e-6)),
       her_goal_change_fraction=float(np.mean(np.linalg.norm(goals-original_goal,axis=1)>1e-8))))
 return result,summary


def rollouts(cfg,task,agent,arm,seed):
 env=make_env(cfg,cfg.seeds.eval_seed,False);result=[];trajectory=[]
 try:
  for episode in range(4):
   reset_seed=cfg.seeds.eval_seed+10*2_000_000+episode;obs=env.reset(seed=reset_seed)
   initial_hash=hashlib.sha256(obs.flat_state().tobytes()).hexdigest();succ=0.;ret=0.;phases=[];dist=[];acts=[]
   for t in range(200):
    state=obs.flat_state();action=agent.act(state,noise_std=0.,noise_rng=np.random.default_rng(0))
    current,rew,terminated,truncated,info=env.step(action);ret+=rew;succ=max(succ,float(info['env_success']))
    phase=int(task.phase_batch(current.observation[None],current.desired_goal[None])[0]);phases.append(phase)
    hand,handle,_,_=task.parse_features(current.observation[None]);distance=float(np.linalg.norm(hand-handle));dist.append(distance);acts.append(action)
    trajectory.append(dict(arm=arm,seed=seed,episode=episode,t=t+1,phase=phase,reward=rew,
         env_success=float(info['env_success']),hand_handle_distance=distance,
         handle_goal_distance=float(np.linalg.norm(current.achieved_goal-current.desired_goal)),
         action_x=float(action[0]),action_y=float(action[1]),action_z=float(action[2]),action_gripper=float(action[3])))
    obs=current
    if terminated or truncated:break
   result.append(dict(arm=arm,seed=seed,episode=episode,reset_seed=reset_seed,initial_state_sha256=initial_hash,
       steps=len(phases),success=succ,return_=ret,min_hand_handle_distance=min(dist),
       contact_reached=bool(np.any(np.array(phases)>=1)),opening_reached=bool(np.any(np.array(phases)>=2)),
       open_reached=bool(np.any(np.array(phases)==3)),approach_fraction=float(np.mean(np.array(phases)==0)),
       action_saturated_component_fraction=float(np.mean(np.abs(np.array(acts))>.99)),
       final_handle_goal_distance=float(np.linalg.norm(obs.achieved_goal-obs.desired_goal))))
 finally:env.close()
 return result,trajectory


def analyze():
 torch.set_num_threads(1)
 cp_results=[];query_rows=[];summaries=[];rollout_rows=[];trajectory_rows=[]
 catalog=[(a,s,'final') for a in ARMS for s in SEEDS]+[('support',s,'best') for s in (13,14,15,16,18,20)]
 folder=OUT/'checkpoint_records';folder.mkdir(exist_ok=False)
 for arm,seed,kind in catalog:
  started=time.monotonic();cfg,task,replay,agent,step=load(arm,seed,kind)
  before={k:v.clone() for k,v in agent.actor.state_dict().items()};updates=agent.total_it
  queries,stats=probe(cfg,task,replay,agent,arm,seed,kind);query_rows+=queries;summaries+=stats
  if kind=='final':
   rolls,traj=rollouts(cfg,task,agent,arm,seed);rollout_rows+=rolls;trajectory_rows+=traj
   expected=read(attempt(arm,seed)/'run/summary.json')['final_eval_env_success']
   require(all(x['success']==expected for x in rolls),'Original binary final evaluations did not reproduce')
  require(agent.total_it==updates and all(torch.equal(v,agent.actor.state_dict()[k]) for k,v in before.items()),'Offline audit modified actor/update count')
  value=dict(arm=arm,seed=seed,checkpoint=kind,step=step,replay_size=replay.size,index_integrity=True,
       support_parity_queries=len(queries),parity_mismatches=0,cache_repeat_mismatches=0,
       updates_before=updates,updates_after=agent.total_it,seconds=time.monotonic()-started)
  cp_results.append(value);write(folder/f'{arm}_{seed:02d}_{kind}.json',value)
  print('CHECKPOINT',arm,seed,kind,'step',step,'support',[round(x['support_rate'],3) for x in stats],
        'seconds',round(value['seconds'],1),flush=True)
  del cfg,task,replay,agent;gc.collect()
 for seed in SEEDS:
  for ep in range(4):
   hashes={r['initial_state_sha256'] for r in rollout_rows if r['seed']==seed and r['episode']==ep}
   require(len(hashes)==1,'Paired-arm rollout reset states differ')
 table(OUT/'checkpoint_summary.csv',cp_results);table(OUT/'query_summary.csv',summaries);table(OUT/'queries.csv',query_rows)
 table(OUT/'rollout_episodes.csv',rollout_rows);table(OUT/'rollout_trajectories.csv',trajectory_rows)
 write(OUT/'checkpoint_audit.json',dict(checkpoints=len(cp_results),queries=len(query_rows),
      parity_mismatches=0,index_failures=0,cache_mismatches=0,replayed_episodes=len(rollout_rows),
      rollout_environment_steps=len(trajectory_rows),paired_initial_states_identical=True,
      original_binary_final_success_reproduced=True,new_training_updates=0))

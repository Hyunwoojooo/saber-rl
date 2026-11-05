import csv,json,statistics,subprocess
import numpy as np
from scipy.stats import binomtest
from common import *


def mixture_decomposition(old,new):
 def terms(entries):
  yes=[r['auc'] for r in entries if r['hit90'] is not None]
  no=[r['auc'] for r in entries if r['hit90'] is None]
  require(yes and no,'Decomposition requires both descriptive groups')
  return len(yes)/len(entries),float(np.mean(yes)),float(np.mean(no))
 p0,s0,f0=terms(old);p1,s1,f1=terms(new)
 rate=(p1-p0)*((s1+s0)-(f1+f0))/2
 conditional=(p1+p0)*(s1-s0)/2+(2-p1-p0)*(f1-f0)/2
 delta=float(np.mean([r['auc'] for r in new])-np.mean([r['auc'] for r in old]))
 require(abs(rate+conditional-delta)<1e-12,'Decomposition fails to sum')
 return dict(old_hit90_fraction=p0,new_hit90_fraction=p1,old_hit_group_auc=s0,new_hit_group_auc=s1,
             old_nohit_group_auc=f0,new_nohit_group_auc=f1,observed_mean_change=delta,
             descriptive_hit_fraction_component=rate,descriptive_conditional_auc_component=conditional,
             causal_interpretation=False)


def weighted(windows,field):
 values=[r for r in windows if r.get(field+'_mean') not in (None,'') and int(r[field+'_n'])>0]
 den=sum(int(r[field+'_n']) for r in values)
 return sum(float(r[field+'_mean'])*int(r[field+'_n']) for r in values)/den if den else None


def analyze():
 OUT.mkdir(exist_ok=False)
 original=read(STUDY/'historical/report.json'); published=read(STUDY/'outputs/analysis/report.json')
 require(not published['failures_or_missing'],'Incomplete native report')
 records=[];curves=[];windows_out=[];bands=[];prefix={};configs=[];envs=[]
 band_edges=(0,100000,250000,500000,1000000,1500000,2000000)
 for arm in ARMS:
  for seed in SEEDS:
   d=attempt(arm,seed);state=read(d/'status.json');cfg=read(d/'run/config.json');env=read(d/'environment.json')
   configs.append(cfg);envs.append(env)
   require(state['status']=='complete' and state['checkpoint_continuation'] is False,'Run not fresh/complete')
   require(read(d/'checkpoint_audit.json')['passed'] is True,'Missing checkpoint audit')
   summary=read(d/'run/summary.json');require(summary['total_updates']==1995001 and summary['total_steps']==2000000,'Budget mismatch')
   data=rows(d/'run/metrics.csv');ev=[r for r in data if r['eval/env_success_mean'] not in ('',None)]
   steps=np.array([int(r['step']) for r in ev]);y=np.array([float(r['eval/env_success_mean']) for r in ev])
   require(np.array_equal(steps,np.arange(10000,2000001,10000)),'Bad evaluation grid')
   require(np.isfinite(y).all() and ((y>=0)&(y<=1)).all(),'Bad evaluation values')
   auc=float(np.trapezoid(y,steps)/1990000);fq=float(np.trapezoid(y[steps>=1500000],steps[steps>=1500000])/500000)
   native=next(r['metrics'] for r in published['runs'] if r['arm']==arm and r['seed']==seed)
   for key,v in [('auc',auc),('fq_auc',fq),('final_success',float(y[-1]))]:require(abs(v-native[key])<1e-12,'Reported metric mismatch')
   for row in ev:curves.append(dict(arm=arm,seed=seed,step=int(row['step']),env_success=float(row['eval/env_success_mean']),goal_success=float(row['eval/goal_success_mean'])))
   episodes={int(r['episode']):r for r in data if r.get('train/episode_len')=='200'}
   require(len(episodes)==10000 and set(episodes)==set(range(1,10001)),'Episode log gaps/duplicates')
   eps=list(episodes.values());pos=[r for r in eps if float(r['train/episode_success'])>0]
   current=[(int(r['step']),float(r['train/episode_return']),float(r['train/episode_success'])) for r in eps if int(r['step'])<=10000]
   if arm=='support':prefix[seed]=current
   else:require(current==prefix[seed],'Initial random-behavior episode outcomes differ')
   for row in data:
    if int(row['step'])>=5000:
     for key in ('loss/critic','loss/actor','loss/td_error_abs_mean','value/q_mean','value/target_mean'):
      if row.get(key) not in ('',None):require(np.isfinite(float(row[key])),f'Nonfinite {key}')
   wins=[r for r in rows(d/'run/assignment_windows.csv') if r['subset']=='all']
   require(sum(int(r['n_updates']) for r in wins)==1995001,'Assignment update grid mismatch')
   for row in wins:
    windows_out.append(dict(arm=arm,seed=seed,step=int(row['step']),n_updates=int(row['n_updates']),
                            support=float(row['support_rate_mean']),eligible=float(row['eligible_rate_mean']),
                            saturation=float(row['saturated_nonempty_update_mean'])))
   sat=[r for r in wins if float(r['saturated_nonempty_update_mean'])>0]
   hit=np.flatnonzero(y>=.9)
   rec=dict(arm=arm,seed=seed,auc=auc,fq_auc=fq,final_success=float(y[-1]),hit90=int(steps[hit[0]]) if hit.size else None,
            zero_all_eval=bool(np.all(y==0)),positive_training_episodes=len(pos),
            first_positive_train_step=min(int(r['step']) for r in pos) if pos else None,
            first_positive_eval_step=int(steps[np.flatnonzero(y>0)[0]]) if np.any(y>0) else None,
            first_saturation_window_end=int(sat[0]['step']) if sat else None,
            support_rate=weighted(wins,'support_rate'),saturation_rate=weighted(wins,'saturated_nonempty_update'),
            goal_auc=float(np.trapezoid([float(r['eval/goal_success_mean']) for r in ev],steps)/1990000),
            best_checkpoint_step=summary['best_eval_step'],cpu_device=cfg['device'])
   records.append(rec)
   for lo,hi in zip(band_edges[:-1],band_edges[1:]):
    w=[r for r in wins if lo<int(r['step'])<=hi];e=[r for r in eps if lo<int(r['step'])<=hi]
    bands.append(dict(arm=arm,seed=seed,band_start=lo,band_end=hi,support=weighted(w,'support_rate'),
                      saturation=weighted(w,'saturated_nonempty_update'),
                      positive_episode_fraction=sum(float(r['train/episode_success'])>0 for r in e)/len(e),
                      positive_episodes=sum(float(r['train/episode_success'])>0 for r in e),episodes=len(e)))
   print('LOGS',arm,seed,auc,'positive training episodes',len(pos),flush=True)
 excluded={'assignment_rule','seeds','output_root','run_name'}
 baseline={k:v for k,v in configs[0].items() if k not in excluded}
 require(all({k:v for k,v in c.items() if k not in excluded}==baseline for c in configs),'Unexpected config differences')
 require(all(e==envs[0] for e in envs),'New cohort environment records differ')
 provenance=read(STUDY/'historical/provenance.json');diff=[]
 for relative,expected in provenance['source_files_sha256'].items():
  require(sha(STUDY/'original'/relative)==expected,'Original source integrity failure')
  if relative.startswith('code/') and sha(STUDY/'study'/relative)!=expected:diff.append(relative)
 require(not diff,'Native code modified')
 release=read(STUDY/'release_manifest.json')
 for relative,expected in release['files'].items():require(sha(STUDY/relative)==expected,'Frozen extension release changed')
 decompositions={};extra_tests={}
 for arm in ARMS:
  prev=[dict(auc=r['metrics']['auc'],hit90=r['metrics']['hit90']) for r in original['runs'] if r['arm']==arm]
  curr=[r for r in records if r['arm']==arm]
  decompositions[arm]=mixture_decomposition(prev,curr)
 for arm in ARMS[1:]:
  a={r['seed']:r['hit90'] is not None for r in records if r['arm']=='support'}
  b={r['seed']:r['hit90'] is not None for r in records if r['arm']==arm}
  a_only=sum(a[s] and not b[s] for s in SEEDS);b_only=sum(b[s] and not a[s] for s in SEEDS)
  extra_tests[arm]=dict(support_only=a_only,control_only=b_only,
      exact_two_sided_p=float(binomtest(a_only,a_only+b_only,.5).pvalue) if a_only+b_only else 1,
      nominal_exploratory_not_multiplicity_adjusted=True)
 table(OUT/'runs.csv',records);table(OUT/'evaluation_curves.csv',curves);table(OUT/'support_windows.csv',windows_out);table(OUT/'time_bands.csv',bands)
 result=dict(passed=True,config_parity=True,environment_parity=True,code_parity=True,
             historical_environment_verified=False,new_environment=envs[0],decomposition=decompositions,
             exploratory_paired_hit90_tests=extra_tests,primary_inference_unchanged=published['inference'])
 write(OUT/'log_audit.json',result)
 return result


if __name__=='__main__':analyze()

#!/usr/bin/env python3
"""Completion gate for DrawerOpen DDPG replay-action-query seeds 1--5."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import analyze_results as analysis
import freeze_source

TOTAL_STEPS=2_000_000
TOTAL_UPDATES=1_995_001
FINAL_REPLAY_SIZE=300_000
EVAL_EPISODES=20
QUERY_MODE="replay_action"
ASSIGNMENT_MODE="support"
WEIGHT_SCOPE="critic_only"
QUERY_FIELDS=(
 "query/query_bin_matches_current_policy_rate",
 "query/query_bin_matches_replay_action_rate",
 "query/context_count_eligible_rate", "query/support_positive_rate",
 "query/high_weight_count", "query/raw_weight_mean",
 "query/normalized_weight_mean",
)
REQUIRED_UPDATE_FIELDS=(
 "train/replay_size", "loss/critic", "loss/actor",
 "loss/td_error_abs_mean", "loss/td_error_abs_p90",
 "value/q_mean", "value/target_mean", *QUERY_FIELDS,
)
REQUIRED_CONFIG={
 "task_name":"drawer_open", "reward_type":"sparse",
 "metaworld_env_name":"drawer-open-v3",
 "metaworld_reward_function_version":"v2", "max_episode_steps":200,
 "total_steps":TOTAL_STEPS, "start_steps":10_000, "warmup_steps":5_000,
 "updates_per_step":1, "batch_size":256, "buffer_capacity":FINAL_REPLAY_SIZE,
 "her_ratio":0.8, "gamma":0.98, "tau":0.005, "actor_lr":3e-4,
 "critic_lr":3e-4, "hidden_dim":256, "noise_std":0.2,
 "eval_freq":10_000, "eval_episodes":EVAL_EPISODES,
 "save_step_checkpoints":False, "M_B":16, "M_Q_min":4,
 "N_local_min":4, "N_gap_min":128, "local_search_max":512,
 "action_bins_per_dim":5, "lambda_support":1.0,
 "weight_scope":WEIGHT_SCOPE, "weighting_mode":"batch_norm",
 "policy_support_mode":"bin", "support_scan_mode":"deterministic_tail_v1",
 "support_cache_enabled":True, "support_cache_max_entries":200_000,
 "local_scan_cache_enabled":True, "local_scan_cache_max_entries":50_000,
 "support_short_circuit":True, "support_query_mode":QUERY_MODE,
 "support_assignment_mode":ASSIGNMENT_MODE, "numba_support_scan":True,
 "support_predicate_early_exit":True,
 "support_diagnostic_audit_interval":100, "success_terminates":False,
 "timeout_bootstrap":True, "method":"b3_support_runtime_opt",
 "drawer_success_threshold":0.05, "drawer_d_reach":0.08,
 "drawer_d_goal_near":0.05, "drawer_d_opening_mid":0.12,
}

@dataclass(frozen=True)
class Expectations:
 total_steps:int=TOTAL_STEPS
 total_updates:int=TOTAL_UPDATES
 final_replay_size:int=FINAL_REPLAY_SIZE
 eval_steps:tuple[int,...]=analysis.EVAL_STEPS
 eval_episodes:int=EVAL_EPISODES

def sha256(path:Path)->str:
 d=hashlib.sha256()
 with path.open("rb") as h:
  for chunk in iter(lambda:h.read(1024*1024),b""): d.update(chunk)
 return d.hexdigest()

def finite(name:str,value:object)->float:
 parsed=analysis.finite_float(value)
 if parsed is None: raise ValueError(f"{name}: expected finite numeric value, got {value!r}")
 return parsed

def close(left:float,right:float,tolerance:float=1e-9)->bool:
 return math.isclose(float(left),float(right),rel_tol=0.0,abs_tol=tolerance)

def require_config(config:Mapping[str,object],seed:int,expected:Expectations)->None:
 contract=dict(REQUIRED_CONFIG)
 contract.update(total_steps=expected.total_steps,buffer_capacity=expected.final_replay_size,eval_episodes=expected.eval_episodes)
 for key,wanted in contract.items():
  if key not in config: raise ValueError(f"config: missing canonical field {key}")
  actual=config[key]
  if isinstance(wanted,float):
   if not close(finite(f"config.{key}",actual),wanted,1e-12): raise ValueError(f"config.{key}: {actual!r} != {wanted!r}")
  elif actual!=wanted: raise ValueError(f"config.{key}: {actual!r} != {wanted!r}")
 seeds=config.get("seeds")
 if not isinstance(seeds,dict): raise ValueError("config.seeds must be an object")
 mappings={"env_seed":seed,"torch_seed":seed,"numpy_seed":seed,"action_noise_seed":seed,"weight_seed":20_000+seed,"eval_seed":10_000+seed}
 for key,wanted in mappings.items():
  if int(seeds.get(key,-1))!=wanted: raise ValueError(f"config.seeds.{key}: {seeds.get(key)!r} != {wanted}")

def _seed_record(payload:Mapping[str,object],seed:int)->Mapping[str,object]:
 records=payload.get("seed_records",payload.get("seeds"))
 if isinstance(records,list):
  for row in records:
   if isinstance(row,dict) and int(row.get("seed",-1))==seed:return row
 if isinstance(records,dict):
  for key in (str(seed),f"seed_{seed}"):
   row=records.get(key)
   if isinstance(row,dict):return row
 return payload

def find_audit(run_dir:Path,result_root:Path,name:str)->Path:
 experiment_root=Path(__file__).resolve().parent
 for candidate in (run_dir/name,result_root/name,experiment_root/name):
  if candidate.is_file():return candidate
 raise FileNotFoundError(f"missing {name} in run, result root, or experiment root")

def validate_config_parity(path:Path,seed:int)->dict[str,object]:
 payload=analysis.read_json(path)
 if payload.get("passed") is not True:raise ValueError(f"{path}: config parity failed")
 checks=payload.get("canonical_source_checks")
 if isinstance(checks,dict) and (not checks or not all(checks.values())):raise ValueError(f"{path}: canonical Main source checks failed")
 row=_seed_record(payload,seed)
 if row.get("passed",row.get("status") in {"pass","passed"}) is not True:raise ValueError(f"{path}: seed {seed} parity failed")
 unexpected=row.get("nonallowed_differences",row.get("unexpected_differences",[]))
 if not isinstance(unexpected,list) or unexpected:raise ValueError(f"{path}: nonallowlisted differences: {unexpected!r}")
 differences=row.get("differences",row.get("allowed_differences",[]))
 if not isinstance(differences,list):raise ValueError(f"{path}: differences must be a list")
 paths=[]
 for item in differences:
  key=str(item.get("path","")) if isinstance(item,dict) else str(item)
  if key not in {"support_query_mode","run_name","output_root"}:raise ValueError(f"{path}: disallowed config difference {key!r}")
  if isinstance(item,dict) and item.get("allowed",True) is not True:raise ValueError(f"{path}: difference not allowed: {item!r}")
  paths.append(key)
 return {"passed":True,"path":path.resolve().as_posix(),"allowed_difference_paths":paths}

def validate_resume_audit(path:Path,seed:int)->dict[str,object]:
 payload=analysis.read_json(path)
 if payload.get("passed") is not True or payload.get("seed_coverage")!=list(analysis.SEEDS):raise ValueError(f"{path}: checkpoint/resume audit failed")
 row=_seed_record(payload,seed)
 required=("query_mode_preserved","learning_result_reproduced","rng_state_restored","replay_query_is_deterministic")
 if row.get("passed",row.get("status") in {"pass","passed"}) is not True:raise ValueError(f"{path}: seed {seed} resume audit failed")
 failures=[key for key in required if row.get(key) is not True]
 if failures:raise ValueError(f"{path}: failed resume fields {failures}")
 return {"passed":True,"path":path.resolve().as_posix(),**{key:True for key in required}}

def validate_source_parity(path:Path)->dict[str,object]:
 payload=analysis.read_json(path)
 if payload.get("passed") is not True or payload.get("failures",[])!=[]:raise ValueError(f"{path}: source parity failed")
 return {"passed":True,"path":path.resolve().as_posix(),"sha256":sha256(path)}

def validate_source_freeze(path:Path,root:Path)->dict[str,object]:
 payload=analysis.read_json(path)
 if payload.get("schema_version")!=2 or payload.get("head_binding_verified") is not True or payload.get("recorded_commit_was_current_head_at_write") is not True:raise ValueError(f"{path}: source freeze lacks schema-v2 HEAD binding")
 # Production completion is repository-aware: verify every worktree byte, Git
 # blob OID, recorded HEAD hash, committed source set, and root launcher.
 freeze_source.verify(path,root=root,repo_root=root.parents[1])
 rows=payload.get("files");entrypoints=payload.get("entrypoints")
 if not isinstance(rows,list) or not rows:raise ValueError(f"{path}: empty source freeze")
 if not isinstance(entrypoints,list) or len(entrypoints)!=1 or entrypoints[0].get("role")!="root_launcher":raise ValueError(f"{path}: root launcher binding missing")
 commit=str(payload.get("source_commit",""))
 if len(commit)!=40 or any(c not in "0123456789abcdef" for c in commit.lower()):raise ValueError(f"{path}: invalid source commit")
 return {"passed":True,"path":path.resolve().as_posix(),"source_commit":commit,"schema_version":2,"head_binding_verified":True,"recorded_commit_was_current_head_at_write":True,"file_count":len(rows),"entrypoint_count":1,"root_launcher_repo_path":entrypoints[0].get("repo_path")}

def validate_env_snapshot(snapshot:Mapping[str,object],seed:int)->None:
 for key,env_seed in (("train_env",seed),("eval_env",10_000+seed)):
  env=snapshot.get(key)
  if not isinstance(env,dict):raise ValueError(f"env_snapshot.{key} missing")
  for field,wanted in {"env_name":"drawer-open-v3","max_episode_steps":200,"reward_type":"sparse","desired_goal_source":"_target_pos","action_dim":4}.items():
   if env.get(field)!=wanted:raise ValueError(f"env_snapshot.{key}.{field} mismatch")
  kwargs=env.get("spec_kwargs")
  if not isinstance(kwargs,dict) or int(kwargs.get("seed",-1))!=env_seed:raise ValueError(f"env_snapshot.{key} seed mismatch")
 task=snapshot.get("task_spec")
 if not isinstance(task,dict) or "DrawerOpen" not in str(task.get("task","")):raise ValueError("env_snapshot task mismatch")

def validate_metrics(path:Path,warmup_steps:int)->dict[str,object]:
 checked=0;match_min=1.0;match_error=0.0;weight_error=0.0
 with path.open(newline="",encoding="utf-8") as handle:
  reader=csv.DictReader(handle);fields=set(reader.fieldnames or [])
  required=set(REQUIRED_UPDATE_FIELDS)|{"config/support_query_mode","config/weight_scope"}
  missing=sorted(required-fields)
  if missing:raise ValueError(f"{path}: missing replay-query columns {missing}")
  for row in reader:
   step=analysis.finite_float(row.get("step"))
   if step is None:raise ValueError(f"{path}: non-finite step in metrics row")
   if step<warmup_steps:continue
   checked+=1
   if row.get("config/support_query_mode")!=QUERY_MODE:raise ValueError(f"{path}: wrong query mode at {int(step)}")
   if row.get("config/weight_scope")!=WEIGHT_SCOPE:raise ValueError(f"{path}: wrong weight scope at {int(step)}")
   values={field:finite(f"metrics[{int(step)}].{field}",row.get(field)) for field in REQUIRED_UPDATE_FIELDS}
   match=values["query/query_bin_matches_replay_action_rate"];normalized=values["query/normalized_weight_mean"]
   if not close(match,1.0,1e-12):raise ValueError(f"{path}: replay-bin match {match} != 1 at {int(step)}")
   if not close(normalized,1.0,1e-6):raise ValueError(f"{path}: normalized mean {normalized} != 1 at {int(step)}")
   if values["query/raw_weight_mean"]<=0 or not 0<=values["query/high_weight_count"]<=256:raise ValueError(f"{path}: invalid weights at {int(step)}")
   for field in ("query/query_bin_matches_current_policy_rate","query/context_count_eligible_rate","query/support_positive_rate"):
    if not 0<=values[field]<=1:raise ValueError(f"{path}: {field} outside [0,1]")
   match_min=min(match_min,match);match_error=max(match_error,abs(match-1));weight_error=max(weight_error,abs(normalized-1))
 if checked==0:raise ValueError(f"{path}: no finite post-warmup update rows")
 return {"post_warmup_update_rows_checked":checked,"replay_action_bin_match_min":match_min,"replay_action_bin_match_max_abs_error":match_error,"normalized_weight_mean_max_abs_error":weight_error,"all_required_fields_finite":True}

def load_checkpoint(path:Path)->Mapping[str,object]:
 try:import torch
 except ImportError as exc:raise RuntimeError("PyTorch is required to audit checkpoint.pt") from exc
 payload=torch.load(path,map_location="cpu",weights_only=False)
 if not isinstance(payload,dict):raise ValueError(f"{path}: checkpoint not a mapping")
 return payload

def canonical_checkpoint_config(config:Mapping[str,object])->dict[str,object]:
 result=dict(config)
 for field in ("run_name","output_root","resume","resume_path"):result.pop(field,None)
 return result

def validate_checkpoint(path:Path,config:Mapping[str,object],expected:Expectations)->dict[str,object]:
 payload=load_checkpoint(path)
 if int(payload.get("global_step",-1))!=expected.total_steps or int(payload.get("total_updates",-1))!=expected.total_updates:raise ValueError(f"{path}: step/update mismatch")
 replay=payload.get("replay")
 if not isinstance(replay,dict) or int(replay.get("size",-1))!=expected.final_replay_size:raise ValueError(f"{path}: replay size mismatch")
 rng=("noise_rng_state","sample_rng_state","weight_rng_state","python_random_state","numpy_global_rng_state","torch_cpu_rng_state","torch_cuda_rng_state_all")
 missing=[key for key in rng if key not in payload]
 if missing:raise ValueError(f"{path}: missing RNG states {missing}")
 history=payload.get("eval_history_env")
 if not isinstance(history,list) or len(history)!=len(expected.eval_steps):raise ValueError(f"{path}: incomplete eval history")
 steps=[int(row[0]) for row in history];values=[finite(f"checkpoint.eval[{step}]",row[1]) for step,row in zip(steps,history)]
 if steps!=list(expected.eval_steps) or any(not 0<=value<=1 for value in values):raise ValueError(f"{path}: invalid exact eval history")
 saved=payload.get("config")
 if not isinstance(saved,dict) or canonical_checkpoint_config(saved)!=canonical_checkpoint_config(config):raise ValueError(f"{path}: checkpoint config mismatch")
 if saved.get("support_query_mode")!=QUERY_MODE:raise ValueError(f"{path}: query mode not preserved")
 agent=payload.get("agent")
 if not isinstance(agent,dict) or not {"critic","critic_target"}.issubset(agent):raise ValueError(f"{path}: not single-critic DDPG")
 if any(key in agent for key in ("critic2","critic2_target","critic_2","critic_target_2")):raise ValueError(f"{path}: unexpected second critic")
 meta=payload.get("checkpoint_metadata");resume=payload.get("resume_audit")
 if not isinstance(meta,dict) or meta.get("config_fingerprint")!=payload.get("config_fingerprint"):raise ValueError(f"{path}: checkpoint metadata mismatch")
 if not isinstance(resume,dict) or resume.get("config_fingerprint")!=payload.get("config_fingerprint"):raise ValueError(f"{path}: resume metadata mismatch")
 return {"global_step":expected.total_steps,"total_updates":expected.total_updates,"replay_size":expected.final_replay_size,"eval_history_points":len(history),"query_mode_preserved":True,"single_critic_ddpg":True,"rng_state_present":True}

def validate_summary(summary:Mapping[str,object],steps:Sequence[int],values:Sequence[float],expected:Expectations)->dict[str,object]:
 for field,wanted in (("total_steps",expected.total_steps),("total_updates",expected.total_updates),("final_replay_size",expected.final_replay_size),("support_query_mode",QUERY_MODE),("support_assignment_mode",ASSIGNMENT_MODE)):
  if summary.get(field)!=wanted:raise ValueError(f"summary.{field}: {summary.get(field)!r} != {wanted!r}")
 auc=analysis.normalized_auc(steps,values);final=values[-1];best=max(values);hit=analysis.first_hit90(steps,values)
 for field,wanted in (("eval_AUC_success",auc),("final_eval_env_success",final),("best_eval_env_success",best)):
  if not close(finite(f"summary.{field}",summary.get(field)),wanted,1e-10):raise ValueError(f"summary.{field} disagrees with exact grid")
 native=summary.get("first_hit_0.90")
 if (None if native in (None,"") else int(native))!=hit:raise ValueError("summary Hit90 disagrees with grid")
 losses=summary.get("final_losses")
 if not isinstance(losses,dict) or not losses:raise ValueError("summary.final_losses missing")
 for key,value in losses.items():finite(f"summary.final_losses.{key}",value)
 metrics=summary.get("final_lite_metrics")
 if not isinstance(metrics,dict):raise ValueError("summary.final_lite_metrics missing")
 for field in QUERY_FIELDS:finite(f"summary.final_lite_metrics.{field}",metrics.get(field))
 if not close(finite("summary replay match",metrics.get("query/query_bin_matches_replay_action_rate")),1.0,1e-12):raise ValueError("summary replay match differs from one")
 if not close(finite("summary normalized mean",metrics.get("query/normalized_weight_mean")),1.0,1e-6):raise ValueError("summary normalized mean differs from one")
 return {"auc":auc,"final_success":final,"best_success":best,"hit90":hit}

def git_head(root:Path)->str|None:
 proc=subprocess.run(["git","rev-parse","HEAD"],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
 return proc.stdout.strip() if proc.returncode==0 else None

def validate_run(run_dir:Path,result_root:Path,seed:int,expected:Expectations=Expectations())->dict[str,object]:
 required=("config.json","env_snapshot.json","summary.json","metrics.csv","checkpoint.pt")
 missing=[name for name in required if not (run_dir/name).is_file()]
 if missing:raise FileNotFoundError(f"{run_dir}: missing completion files {missing}")
 steps_saved=sorted(run_dir.glob("checkpoint_step_*.pt"))
 if steps_saved:raise ValueError(f"{run_dir}: step checkpoints must be disabled")
 config=analysis.read_json(run_dir/"config.json");summary=analysis.read_json(run_dir/"summary.json");snapshot=analysis.read_json(run_dir/"env_snapshot.json")
 require_config(config,seed,expected);validate_env_snapshot(snapshot,seed)
 metrics_path=run_dir/"metrics.csv";steps,values=analysis.load_eval_curve(metrics_path,expected.eval_steps)
 metric_gate=validate_metrics(metrics_path,int(config["warmup_steps"]));summary_gate=validate_summary(summary,steps,values,expected);checkpoint_gate=validate_checkpoint(run_dir/"checkpoint.pt",config,expected)
 parity=validate_config_parity(find_audit(run_dir,result_root,"config_parity.json"),seed);resume=validate_resume_audit(find_audit(run_dir,result_root,"checkpoint_resume_audit.json"),seed)
 inputs=[]
 for path in (run_dir/"config.json",run_dir/"env_snapshot.json",run_dir/"summary.json",metrics_path,run_dir/"checkpoint.pt"):
  inputs.append({"path":path.resolve().as_posix(),"size_bytes":path.stat().st_size,"sha256":sha256(path)})
 return {"seed":seed,"run_dir":run_dir.resolve().as_posix(),"passed":True,"total_steps":expected.total_steps,"total_updates":expected.total_updates,"final_replay_size":expected.final_replay_size,"evaluation_points":len(steps),"evaluation_episodes_per_point":expected.eval_episodes,"single_critic_ddpg":True,"support_query_mode":QUERY_MODE,"support_assignment_mode":ASSIGNMENT_MODE,"weight_scope":WEIGHT_SCOPE,"summary":summary_gate,"metrics":metric_gate,"checkpoint":checkpoint_gate,"config_parity":parity,"checkpoint_resume_audit":resume,"environment_snapshot_validated":True,"inputs":inputs}

def validate_all(result_root:Path,repo_root:Path,expected:Expectations=Expectations())->dict[str,object]:
 runs=analysis.discover_runs(result_root);records=[validate_run(run_dir,result_root,seed,expected) for seed,run_dir in runs.items()]
 experiment=Path(__file__).resolve().parent;first=next(iter(runs.values()))
 source_parity=validate_source_parity(find_audit(first,result_root,"source_parity.json"));source_freeze=validate_source_freeze(find_audit(first,result_root,"freeze_manifest.json"),experiment);head=git_head(repo_root)
 return {"schema_version":1,"experiment":"DDPG-HER-B3-Replay-Action-Query-2M","passed":True,"source_commit":source_freeze["source_commit"],"current_git_head":head,"source_commit_relation":"recorded_source_commit_matches_current_head" if head==source_freeze["source_commit"] else "recorded_source_commit_differs_from_current_head","result_root":result_root.resolve().as_posix(),"seed_count":len(records),"seeds":list(analysis.SEEDS),"evaluation_grid":list(expected.eval_steps),"evaluation_field":analysis.PRIMARY_SUCCESS_FIELD,"query_contract":{"support_query_mode":QUERY_MODE,"support_assignment_mode":ASSIGNMENT_MODE,"weight_scope":WEIGHT_SCOPE,"replay_action_bin_match":1.0,"sampled_transition_removed_from_local_pool":False},"source_parity":source_parity,"source_freeze":source_freeze,"records":records}

def parse_args(argv:Sequence[str]|None=None)->argparse.Namespace:
 repo=Path(__file__).resolve().parents[2];parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--repo-root",type=Path,default=repo);parser.add_argument("--replay-root",type=Path,default=repo/"metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Replay-Action-Query-2M");parser.add_argument("--output",type=Path);parser.add_argument("--force",action="store_true");return parser.parse_args(argv)

def main(argv:Sequence[str]|None=None)->int:
 args=parse_args(argv);root=args.replay_root.resolve();output=args.output.resolve() if args.output else root/"completion_validation.json"
 if output.exists() and not args.force:raise FileExistsError(f"refusing to overwrite existing completion validation: {output}")
 result=validate_all(root,args.repo_root.resolve());output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8");print(json.dumps({"status":"passed","output":output.as_posix(),"seed_count":len(result["records"])},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())

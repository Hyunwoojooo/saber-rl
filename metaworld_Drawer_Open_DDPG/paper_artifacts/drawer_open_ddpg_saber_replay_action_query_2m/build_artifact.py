#!/usr/bin/env python3
"""Build the non-overwriting compact DrawerOpen replay-action-query artifact."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,shutil,tempfile
from datetime import datetime,timezone
from pathlib import Path
from typing import Any,Iterable,Mapping,Sequence
import analyze_results as analysis
import validate_completion as completion
ARTIFACT_ID="drawer_open_ddpg_saber_replay_action_query_2m"
EXPERIMENT="DDPG-HER-B3-Replay-Action-Query-2M"
COMPARATOR_ARTIFACT_COMMIT="93ae5c03633726f8e7783a157cac8516348f088f"
HERE=Path(__file__).resolve().parent;REPO_ROOT=HERE.parents[1]
DEFAULT_REPLAY_ROOT=REPO_ROOT/f"metaworld_Drawer_Open_DDPG/results_2m/{EXPERIMENT}"
DEFAULT_COMPARATOR=REPO_ROOT/"metaworld_Drawer_Open_DDPG/paper_artifacts/drawer_open_ddpg_saber_context_eligible_count_matched_2m"
DEFAULT_OUTPUT=REPO_ROOT/f"metaworld_Drawer_Open_DDPG/paper_artifacts/{ARTIFACT_ID}"
def sha256(path:Path)->str:
 d=hashlib.sha256()
 with path.open("rb") as h:
  for chunk in iter(lambda:h.read(1048576),b""):d.update(chunk)
 return d.hexdigest()
def write_json(path:Path,payload:object)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def clean_json(value:Any,replacements:Sequence[tuple[str,str]])->Any:
 if isinstance(value,float):return value if math.isfinite(value) else None
 if isinstance(value,dict):return {str(k):clean_json(v,replacements) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [clean_json(v,replacements) for v in value]
 if isinstance(value,str):
  for old,new in replacements:value=value.replace(old,new)
 return value
def record_input(repo:Path,path:Path,role:str,**metadata:object)->dict[str,object]:
 resolved=path.resolve()
 try:source=resolved.relative_to(repo.resolve()).as_posix()
 except ValueError:source=f"<external>/{resolved.name}"
 return {"role":role,"source_path":source,"size_bytes":resolved.stat().st_size,"sha256":sha256(resolved),**metadata}
def copy_selected(source:Path,destination:Path)->None:
 expected=set(analysis.EVAL_STEPS);rows=[];seen=set()
 with source.open(newline="",encoding="utf-8") as h:
  reader=csv.DictReader(h);fields=list(reader.fieldnames or [])
  if analysis.PRIMARY_SUCCESS_FIELD not in fields:raise ValueError(f"{source}: missing primary success")
  for row in reader:
   if row.get(analysis.PRIMARY_SUCCESS_FIELD) in ("",None):continue
   step_value=analysis.finite_float(row.get("step"));success=analysis.finite_float(row.get(analysis.PRIMARY_SUCCESS_FIELD))
   if step_value is None or success is None:raise ValueError(f"{source}: non-finite eval row")
   step=int(step_value)
   if step not in expected or step in seen:raise ValueError(f"{source}: unexpected/duplicate step {step}")
   seen.add(step);rows.append(row)
 if seen!=expected or [int(float(row["step"])) for row in rows]!=list(analysis.EVAL_STEPS):raise ValueError(f"{source}: exact grid mismatch")
 destination.parent.mkdir(parents=True,exist_ok=True)
 with destination.open("w",newline="",encoding="utf-8") as h:
  writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def packaged_files(root:Path)->list[dict[str,object]]:return [{"path":p.relative_to(root).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(root.rglob("*")) if p.is_file() and p.name!="artifact_manifest.json"]
def content_digest(rows:Iterable[Mapping[str,object]])->str:
 d=hashlib.sha256()
 for row in rows:d.update(str(row["path"]).encode());d.update(b"\0");d.update(str(row["sha256"]).encode());d.update(b"\n")
 return d.hexdigest()
def direct_analysis(methods:Sequence[analysis.MethodSpec],runs:Mapping[str,Mapping[int,Path]])->dict[str,object]:
 per_seed=[];curves={}
 for method in methods:
  for seed in analysis.SEEDS:
   row,values=analysis.analyze_run(method,seed,runs[method.key][seed]);per_seed.append(row);curves[(method.key,seed)]=values
 aggregate=analysis.aggregate_rows(per_seed);paired=analysis.paired_comparison_rows(per_seed);labels={m.key:m.label for m in methods}
 return {"schema_version":1,"seeds":list(analysis.SEEDS),"evaluation_steps":list(analysis.EVAL_STEPS),"evaluation_field":analysis.PRIMARY_SUCCESS_FIELD,"auc_convention":"observed-only trapezoid normalized by (2,000,000 - 10,000)","smoothing":False,"interpolation":False,"forward_fill":False,"extrapolation":False,"bootstrap":"ordered exhaustive paired bootstrap, 5^5=3125 index sequences","per_seed":per_seed,"aggregate":aggregate,"paired_comparisons":paired,"mean_curves":analysis.mean_curve_rows(curves,labels)}
def query_summary(metrics:Path,warmup:int)->dict[str,object]:
 values={field:[] for field in completion.QUERY_FIELDS};rows=0
 with metrics.open(newline="",encoding="utf-8") as h:
  for row in csv.DictReader(h):
   step=analysis.finite_float(row.get("step"))
   if step is None:raise ValueError(f"{metrics}: non-finite step in metrics row")
   if step<warmup:continue
   rows+=1
   required={field:completion.finite(field,row.get(field)) for field in completion.REQUIRED_UPDATE_FIELDS}
   for field in completion.QUERY_FIELDS:values[field].append(required[field])
 if not rows:raise ValueError(f"{metrics}: no post-warmup query rows")
 return {"schema_version":1,"post_warmup_update_rows":rows,"support_query_mode":"replay_action","support_assignment_mode":"support","sampled_transition_removed_from_local_pool":False,"self_support_possible":True,"fields":{field:{"count":len(items),"mean":math.fsum(items)/len(items),"min":min(items),"max":max(items)} for field,items in values.items()},"invariants":{"all_finite":True,"replay_action_bin_match_all_one":all(math.isclose(v,1.0,rel_tol=0,abs_tol=1e-12) for v in values["query/query_bin_matches_replay_action_rate"]),"normalized_weight_mean_all_one":all(math.isclose(v,1.0,rel_tol=0,abs_tol=1e-6) for v in values["query/normalized_weight_mean"])}}
def reconstructed_config(method:str,seed:int,proxy:Mapping[str,object],proxy_sha:str)->dict[str,object]:
 keys=("task_name","reward_type","metaworld_env_name","metaworld_reward_function_version","max_episode_steps","total_steps","start_steps","warmup_steps","updates_per_step","batch_size","buffer_capacity","her_ratio","gamma","tau","actor_lr","critic_lr","hidden_dim","noise_std","eval_freq","eval_episodes","success_terminates","timeout_bootstrap","seeds")
 return {"schema_version":1,"record_type":"reconstructed_metadata_not_original_run_config","not_original_run_config":True,"method_key":method,"seed":seed,"reason":"Verified comparator compact data contains exact metrics/summaries but not original HER/Main configs.","proxy_source":"same-seed Strict config from verified comparator artifact","proxy_source_sha256":proxy_sha,"reconstructed_shared_fields":{key:proxy[key] for key in keys if key in proxy},"method_specific_training_fields_not_reconstructed":True}
def proxy_environment(method:str,seed:int,snapshot:Mapping[str,object],proxy_sha:str)->dict[str,object]:return {"schema_version":1,"record_type":"same_seed_environment_proxy_not_original_snapshot","not_original_run_env_snapshot":True,"method_key":method,"seed":seed,"reason":"Comparator compact data omits original HER/Main env snapshots; this is an explicitly labelled same-seed Strict environment proxy.","proxy_source_sha256":proxy_sha,"proxy_snapshot":snapshot}
def readme_text(result:Mapping[str,object],source_commit:str,comparator_source:str)->str:
 aggregate={str(r["method_key"]):r for r in result["aggregate"]};primary=next(r for r in result["paired_comparisons"] if r["contrast_key"]=="main_minus_replay")
 lines=[f"# {EXPERIMENT}","","Compact, independently verifiable artifact for DrawerOpen seeds 1–5. Exact success AUC uses only observed 10k–2M points, with no smoothing, interpolation, forward-fill, or extrapolation.","","Replay-action query does not remove the sampled transition from the local replay pool. Consequently the replay action can self-support when that transition is present in the bounded local scan. This is the same intended replay-query definition used by the FPP query ablation.","","Only the SABER membership query changes. Actor objective/update and the Bellman target are unchanged. Bellman regression remains at the replay action, and weights remain critic-only.","","| Method | AUC mean ± sample SD | Final mean ± sample SD |","|---|---:|---:|"]
 for key in ("her","main","replay"):
  row=aggregate[key];lines.append(f"| {row['method']} | {float(row['auc_mean']):.6f} ± {float(row['auc_sample_sd']):.6f} | {float(row['final_success_mean']):.6f} ± {float(row['final_success_sample_sd']):.6f} |")
 lines+=["",f"Primary Main − Replay AUC difference: {float(primary['mean_auc_difference']):.6f}; exhaustive paired-bootstrap 95% CI [{float(primary['bootstrap_ci95_lower']):.6f}, {float(primary['bootstrap_ci95_upper']):.6f}]; exact two-sided sign-flip p={float(primary['exact_two_sided_sign_flip_pvalue']):.6f}.","","All five seeds and paired statistics are reported regardless of direction. If the CI includes zero, this artifact does not claim that an independent current-policy-query effect is established.","",f"HER/Main were reused from verified artifact commit {COMPARATOR_ARTIFACT_COMMIT} (packaging artifact source commit {comparator_source}; original training source commit unknown/not recorded) and were not retrained. Original comparator config/env files were absent, so HER/Main config records are explicitly reconstructed metadata and env records are explicitly same-seed Strict proxies, never original snapshots.","",f"Frozen replay source commit: {source_commit}.","","Verify from this directory with:","","    python verify_artifact.py --artifact-dir .","","Checkpoints, full raw metrics, JSONL, logs, smoke/correctness/pilot outputs, caches, and temporary files are excluded."]
 return "\n".join(lines)+"\n"
def build(repo_root:Path,replay_root:Path,comparator:Path,output:Path)->None:
 repo_root=repo_root.resolve();replay_root=replay_root.resolve();comparator=comparator.resolve();output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():raise FileExistsError(f"refusing to overwrite {output}")
 comparator_manifest=analysis.read_json(comparator/"artifact_manifest.json");comparator_source=str(comparator_manifest.get("source_commit",""));comparator_input_path=comparator/"input_manifest.json";comparator_upstream=analysis.read_json(comparator_input_path)
 if len(comparator_source)!=40 or comparator_upstream.get("source_commit")!=comparator_source:raise ValueError("invalid comparator packaging source provenance")
 upstream_rows=comparator_upstream.get("inputs")
 if not isinstance(upstream_rows,list):raise ValueError("comparator input manifest is malformed")
 upstream_by_role={str(row.get("role","")):row for row in upstream_rows if isinstance(row,dict)}
 gate=completion.validate_all(replay_root,repo_root);replay_runs=analysis.discover_runs(replay_root)
 runs={key:{seed:comparator/"data"/key/f"seed_{seed}" for seed in analysis.SEEDS} for key in ("her","main")};runs["replay"]=replay_runs
 methods=(analysis.MethodSpec("her","DrawerOpen DDPG HER",comparator/"data/her"),analysis.MethodSpec("main","DrawerOpen DDPG HER + SABER Main",comparator/"data/main"),analysis.MethodSpec("replay","DrawerOpen DDPG Replay-Action Query",replay_root))
 result=direct_analysis(methods,runs);replacements=((repo_root.as_posix(),"."),);stage=Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}.",dir=output.parent));inputs=[]
 try:
  packaged=clean_json(result,replacements)
  for row in packaged["per_seed"]:
   key=str(row["method_key"]);seed=int(row["seed"]);row["run_dir"]=f"data/{key}/seed_{seed}";row["metrics_path"]=f"data/{key}/seed_{seed}/metrics_selected_10k.csv"
  analysis.write_csv(stage/"per_seed_results.csv",packaged["per_seed"]);analysis.write_csv(stage/"aggregate_results.csv",packaged["aggregate"]);analysis.write_csv(stage/"paired_comparisons.csv",packaged["paired_comparisons"]);analysis.write_csv(stage/"exact_grid_mean_curves.csv",packaged["mean_curves"]);write_json(stage/"analysis_results.json",packaged)
  gate_records={int(r["seed"]):r for r in gate["records"]}
  for method in methods:
   for seed,run_dir in runs[method.key].items():
    target=stage/"data"/method.key/f"seed_{seed}";metrics=analysis.metrics_path_for(run_dir);summary=run_dir/"summary.json";copy_selected(metrics,target/"metrics_selected_10k.csv");write_json(target/"summary.json",clean_json(analysis.read_json(summary),replacements));commit=gate["source_commit"]
    if method.key=="replay":inputs.extend((record_input(repo_root,metrics,f"replay.seed_{seed}.metrics",source_commit=commit),record_input(repo_root,summary,f"replay.seed_{seed}.summary",source_commit=commit)))
    else:
     chained=[]
     for suffix,source in (("metrics",metrics),("summary",summary)):
      role=f"{method.key}.seed_{seed}.{suffix}";upstream=upstream_by_role.get(role)
      if not isinstance(upstream,dict):raise ValueError(f"comparator input manifest missing {role}")
      chained.append(record_input(repo_root,source,role,packaging_artifact_source_commit=comparator_source,original_training_source_commit=None,original_training_source_commit_status="unknown_not_recorded",upstream_input_provenance=dict(upstream)))
     inputs.extend(chained)
    if method.key=="replay":
     config_path=run_dir/"config.json";env_path=run_dir/"env_snapshot.json";config=analysis.read_json(config_path);write_json(target/"config.json",clean_json(config,replacements));write_json(target/"env_snapshot.json",clean_json(analysis.read_json(env_path),replacements));write_json(target/"query_diagnostic_summary.json",query_summary(run_dir/"metrics.csv",int(config["warmup_steps"])));write_json(target/"completion_validation.json",clean_json(gate_records[seed],replacements));inputs.extend((record_input(repo_root,config_path,f"replay.seed_{seed}.config",source_commit=commit),record_input(repo_root,env_path,f"replay.seed_{seed}.env_snapshot",source_commit=commit),record_input(repo_root,run_dir/"checkpoint.pt",f"replay.seed_{seed}.completion_checkpoint_excluded",source_commit=commit)))
    else:
     proxy_config_path=comparator/"data/strict"/f"seed_{seed}"/"config.json";proxy_env_path=comparator/"data/strict"/f"seed_{seed}"/"env_snapshot.json";proxy_config=analysis.read_json(proxy_config_path);proxy_env=clean_json(analysis.read_json(proxy_env_path),replacements);write_json(target/"config.json",reconstructed_config(method.key,seed,proxy_config,sha256(proxy_config_path)));write_json(target/"env_snapshot.json",proxy_environment(method.key,seed,proxy_env,sha256(proxy_env_path)));inputs.extend((record_input(repo_root,proxy_config_path,f"{method.key}.seed_{seed}.reconstructed_config_proxy",packaging_artifact_source_commit=comparator_source,original_training_source_commit=None,original_training_source_commit_status="unknown_not_recorded",not_original_comparator_config=True),record_input(repo_root,proxy_env_path,f"{method.key}.seed_{seed}.environment_proxy",packaging_artifact_source_commit=comparator_source,original_training_source_commit=None,original_training_source_commit_status="unknown_not_recorded",not_original_comparator_env_snapshot=True)))
  for name in ("config_parity.json","checkpoint_resume_audit.json","source_parity.json","freeze_manifest.json"):
   source=HERE/name
   if not source.is_file():raise FileNotFoundError(f"required provenance missing: {source}")
   write_json(stage/"provenance"/name,clean_json(analysis.read_json(source),replacements));inputs.append(record_input(repo_root,source,f"provenance.{name}",source_commit=gate["source_commit"]))
  shutil.copyfile(comparator_input_path,stage/"provenance"/"comparator_input_manifest.json");inputs.append(record_input(repo_root,comparator_input_path,"provenance.comparator_input_manifest",packaging_artifact_source_commit=comparator_source))
  freeze=analysis.read_json(HERE/"freeze_manifest.json");frozen=freeze.get("files")
  if not isinstance(frozen,list) or not frozen:raise ValueError("empty source freeze")
  lines=[]
  for row in frozen:
   rel=str(row["path"]);source=HERE/rel;dest=stage/"source_snapshot"/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
   if dest.stat().st_size!=int(row["size"]) or sha256(dest)!=row["sha256"]:raise ValueError(f"frozen copy mismatch: {rel}")
   lines.append(f"{row['sha256']}  source_snapshot/{rel}")
  entrypoints=freeze.get("entrypoints")
  if freeze.get("schema_version")!=2 or freeze.get("head_binding_verified") is not True or freeze.get("recorded_commit_was_current_head_at_write") is not True or not isinstance(entrypoints,list) or len(entrypoints)!=1:raise ValueError("source freeze lacks schema-v2 root-launcher binding")
  launcher_row=entrypoints[0]
  if not isinstance(launcher_row,dict) or launcher_row.get("role")!="root_launcher" or launcher_row.get("head_bound") is not True:raise ValueError("invalid frozen root launcher metadata")
  launcher_source=repo_root/str(launcher_row.get("repo_path",""));launcher_dest=stage/"source_snapshot"/"root_launcher"/Path(str(launcher_row.get("path",""))).name;launcher_dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(launcher_source,launcher_dest)
  if launcher_dest.stat().st_size!=int(launcher_row.get("size",-1)) or sha256(launcher_dest)!=launcher_row.get("sha256") or launcher_row.get("sha256")!=launcher_row.get("head_sha256"):raise ValueError("frozen root launcher copy mismatch")
  lines.append(f"{launcher_row['sha256']}  source_snapshot/root_launcher/{launcher_dest.name}");inputs.append(record_input(repo_root,launcher_source,"frozen.root_launcher",source_commit=gate["source_commit"],git_blob_oid=launcher_row.get("git_blob_oid"),head_bound=True))
  (stage/"frozen_source.sha256").write_text("\n".join(lines)+"\n",encoding="utf-8")
  for name in ("analyze_results.py","build_artifact.py","verify_artifact.py"):shutil.copyfile(HERE/name,stage/name)
  write_json(stage/"completion_validation.json",clean_json(gate,replacements))
  comparator_input={"artifact_path":comparator.relative_to(repo_root).as_posix(),"artifact_git_commit":COMPARATOR_ARTIFACT_COMMIT,"artifact_manifest_sha256":sha256(comparator/"artifact_manifest.json"),"packaging_artifact_source_commit":comparator_source,"original_training_source_commit":None,"original_training_source_commit_status":"unknown_not_recorded","upstream_input_manifest_path":"provenance/comparator_input_manifest.json","upstream_input_manifest_sha256":sha256(comparator_input_path),"arms_reused":["her","main"],"retrained":False,"original_comparator_config_env_available":False,"metadata_policy":"reconstructed config metadata and explicitly labelled same-seed Strict environment proxy"}
  write_json(stage/"input_manifest.json",{"schema_version":1,"artifact_id":ARTIFACT_ID,"source_commit":gate["source_commit"],"comparator":comparator_input,"input_file_count":len(inputs),"inputs":inputs,"transform":{"metrics":"exact observed eval rows 10k..2M","json":"local repo path replaced by dot; non-finite replaced by null","checkpoints_copied":False,"raw_metrics_copied":False,"jsonl_logs_smoke_copied":False}})
  (stage/"README.md").write_text(readme_text(packaged,str(gate["source_commit"]),comparator_source),encoding="utf-8")
  records=packaged_files(stage);manifest={"schema_version":1,"artifact_id":ARTIFACT_ID,"experiment":EXPERIMENT,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"source_commit":gate["source_commit"],"comparator_artifact_commit":COMPARATOR_ARTIFACT_COMMIT,"comparator_packaging_source_commit":comparator_source,"method_seed_coverage":{key:list(analysis.SEEDS) for key in ("her","main","replay")},"evaluation":{"field":analysis.PRIMARY_SUCCESS_FIELD,"steps":list(analysis.EVAL_STEPS),"points_per_seed":200,"episodes_per_point":20,"auc":"observed-only normalized trapezoid","smoothing":False,"interpolation":False,"forward_fill":False,"extrapolation":False},"bootstrap":{"method":"ordered exhaustive paired bootstrap","resamples":3125,"confidence_level":0.95},"exact_sign_flip":{"two_sided":True,"assignments":32},"completion_passed":True,"config_parity_passed":True,"source_parity_passed":True,"packaged_file_count_excluding_manifest":len(records),"package_content_sha256_excluding_manifest":content_digest(records),"packaged_files":records,"self_hash_note":"artifact_manifest.json excluded to avoid recursive self-hash"};write_json(stage/"artifact_manifest.json",manifest);stage.rename(output)
 except BaseException:shutil.rmtree(stage,ignore_errors=True);raise
def parse_args(argv:Sequence[str]|None=None)->argparse.Namespace:
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--repo-root",type=Path,default=REPO_ROOT);parser.add_argument("--replay-root",type=Path,default=DEFAULT_REPLAY_ROOT);parser.add_argument("--comparator-artifact",type=Path,default=DEFAULT_COMPARATOR);parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT);return parser.parse_args(argv)
def main(argv:Sequence[str]|None=None)->int:
 args=parse_args(argv);build(args.repo_root,args.replay_root,args.comparator_artifact,args.output_dir);print(json.dumps({"status":"built","artifact":args.output_dir.resolve().as_posix()}));return 0
if __name__=="__main__":raise SystemExit(main())

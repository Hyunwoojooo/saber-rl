"""Three-way source, manifest, import-origin and actual-config audit."""
import csv,difflib,hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
STUDY=Path('/home/mimiclab-gigabyte-gen5/ICML2027/draweropen_ddpg_assignment_extension_20261020')
PREFIX='ICML_27/experiments/draweropen_ddpg_saber_vs_random_vs_td_topk/source/'

def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')

def main():
 commit=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()
 files=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,PREFIX],text=True).splitlines()
 rows=[];diff=[]
 for full in files:
  relative=full.removeprefix(PREFIX)
  git=subprocess.check_output(['git','show',commit+':'+full]);preserved=(STUDY/'original'/relative).read_bytes();new=(STUDY/'study'/relative).read_bytes()
  assert git==preserved,('Github/preserved original differs',relative)
  rows.append(dict(file=relative,github_sha256=digest(git),preserved_sha256=digest(preserved),extension_sha256=digest(new),
                   identical=git==new,group='native' if relative.startswith('code/') else 'upstream' if relative.startswith('upstream/') else 'wrapper_docs_metadata'))
  if git!=new:diff.extend(difflib.unified_diff(git.decode().splitlines(True),new.decode().splitlines(True),fromfile='seed1_10/'+relative,tofile='seed11_20/'+relative))
 relative_names={r['file'] for r in rows}
 added=[str(p.relative_to(STUDY/'study')) for p in (STUDY/'study').rglob('*') if p.is_file() and
        not {'validation','__pycache__'}&set(p.relative_to(STUDY/'study').parts) and str(p.relative_to(STUDY/'study')) not in relative_names]
 with (ROOT/'file_hashes.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 (ROOT/'source_diff.patch').write_text(''.join(diff))
 old=read(ROOT/'old_probe.json');new=read(ROOT/'new_probe.json')
 assert old['same_seed_configs']==new['same_seed_configs'],'Same-seed CPU/CUDA constructor differs'
 assert old['extension_configs']==new['extension_configs'],'Extension constructor differs'
 assert all(old['origins'][k]['sha256']==new['origins'][k]['sha256'] for k in old['origins']),'Imported module implementation differs'
 old_manifest=(STUDY/'original/source_manifest.json').read_bytes();new_manifest=(STUDY/'study/source_manifest.json').read_bytes()
 old_sha=digest(old_manifest);new_sha=digest(new_manifest)
 for folder in ('original','study'):
  manifest=read(STUDY/folder/'source_manifest.json')
  for name,expected in manifest['sha256'].items():assert digest((STUDY/folder/name).read_bytes())==expected,(folder,name)
 historical=read(STUDY/'historical/report.json')
 assert len(historical['runs'])==30
 for r in historical['runs']:
  assert len(r['attempts'])==1 and r['attempts'][0]['source_manifest_sha256']==old_sha
 old_states=[a for r in historical['runs'] for a in r['attempts']]
 new_states=[];runs=[]
 templates={(r['arm'],r['seed']):r['config'] for r in old['extension_configs']}
 metadata=set()
 for arm in ('support','random','td_topk'):
  for seed in range(11,21):
   d=STUDY/f'outputs/production/{arm}/seed_{seed:02d}/attempt_01'
   status=read(d/'status.json');saved=read(d/'run/config.json');requested=read(d/'requested_config.json')
   new_states.append(status)
   assert status['source_manifest_sha256']==new_sha and status['status']=='complete' and status['checkpoint_continuation'] is False
   template=templates[arm,seed].copy();template['output_root']=saved['output_root']
   different={k:dict(expected=v,actual=saved.get(k)) for k,v in template.items() if saved.get(k)!=v}
   assert not different,(arm,seed,different)
   assert requested==template,(arm,seed,'requested config differs')
   extras=set(saved)-set(template);assert extras<={'thread_hint','terminal_convention','resolved_env_id'},extras
   metadata.update(extras)
   log=(d/'run/run.log').read_text();assert 'device=cpu' in log
   runs.append(dict(arm=arm,seed=seed,source_manifest_sha256=status['source_manifest_sha256'],device=saved['device'],
                    old_source_reconstructed_learning_config_matches=True,extra_reporting_fields=sorted(extras)))
 outer=read(STUDY/'release_manifest.json')
 for name,expected in outer['files'].items():assert digest((STUDY/name).read_bytes())==expected,('Release changed',name)
 same_native=[r for r in rows if r['group']=='native' and r['identical']];same_upstream=[r for r in rows if r['group']=='upstream' and r['identical']]
 assert len(same_native)==26 and len(same_upstream)==25
 def peak(states):
  events=sorted([(datetime.fromisoformat(x['started_utc']),1) for x in states]+[(datetime.fromisoformat(x['finished_utc']),-1) for x in states])
  count=maximum=0
  for _,delta in events:count+=delta;maximum=max(maximum,count)
  return maximum
 result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),github_commit=commit,github_source_path=PREFIX,
     original_files=len(files),github_matches_preserved_original=True,identical_native_files=len(same_native),
     identical_upstream_files=len(same_upstream),unchanged_original_files=sum(r['identical'] for r in rows),
     changed_files=[r['file'] for r in rows if not r['identical']],added_files=added,
     same_seed_configuration_checks=len(old['same_seed_configs']),extension_actual_configuration_checks=len(runs),
     imported_module_origin_checks=len(old['origins']),original_source_manifest_sha256=old_sha,
     extension_source_manifest_sha256=new_sha,old_run_manifest_matches=30,new_run_manifest_matches=30,
     reconstructed_source_default_device=old['source_default_device'],new_actual_device='cpu',
     historical_actual_device_verified=False,historical_actual_packages_verified=False,
     historical_recorded_peak_active_attempts=peak(old_states),new_recorded_peak_active_attempts=peak(new_states),
     historical_method_schedule='interleaved methods',new_method_schedule='support then random then td_topk',
     generated_config_reporting_fields=sorted(metadata),
     qualification='Core learning code is byte-identical. Entire source folder and execution wrappers are not identical. Historical runtime parity remains unverified.',
     no_training_executed=True)
 write(ROOT/'report.json',result);write(ROOT/'actual_run_checks.json',runs)
 lines=['# Seed 1–10 / 11–20 source 재감사','',
 f"검사시각: {result['checked_utc']}. GitHub main commit `{commit}`.",'',
 '**전체 source 폴더가 동일한 것은 아니다. 핵심 학습 코드 `code/` 26개와 `upstream/` 25개는 바이트까지 동일하다.**','',
 '## 대조 방식','',
 '1. 최신 GitHub Git blob → 로컬 보존 original → 실제 확장 study의 파일별 SHA256과 원문 비교.',
 '2. 과거 30개 status의 manifest hash → 원본 manifest → 파일별 hash 대조. 확장 30개도 같은 방식으로 대조.',
 '3. 별도 Python 프로세스 두 개에서 각 source를 import하고 실제 module 경로·hash 확인. Config, actor/critic, trainer, replay/HER, weighting, selector, task, env, scanner 등 10개 module이 각 전용 code/src에서 로드되는지 확인.',
 '4. 같은 seed 1–10, 세 arm, CPU/CUDA 각각의 config 60개 비교. 원본 함수를 현재 환경에서 호출한 재구성 검증이며 과거 실행 환경의 재현이 아니다.',
 '5. 원본 함수의 seed 허용목록만 audit process 메모리에서 11–20으로 넓혀, 원본 함수 본문으로 만든 30개 config와 실제 저장된 requested_config/config 비교. 디스크의 원본을 수정하지 않고 trainer를 호출하지 않았다.','',
 '## 결과','',
 '| 항목 | 결과 |','|---|---|',
 f'| GitHub ↔ 보존 original | {len(files)}개 모두 동일 |',
 f'| GitHub ↔ 확장 study의 code/ | {len(same_native)}개 모두 동일 |',
 f'| GitHub ↔ 확장 study의 upstream/ | {len(same_upstream)}개 모두 동일 |',
 '| 실제 imported 학습 module | 10개 모두 전용 code/src를 사용하며 구현 hash 동일 |',
 '| 같은 seed·device로 생성한 config | 60/60 동일 |',
 '| 확장 30회의 실제 저장 학습 config | 원본 함수에서 재구성한 11–20 설정과 30/30 일치 |',
 '| 실행 source hash | 과거 30개는 원본 manifest, 새 30개는 확장 manifest와 일치 |','',
 '다른 폴더의 metacog_lite_v0가 import되는 문제, support/TD/Random label 교체, actor/critic LR·loss·target·HER·replay·support 식 변경은 이 비교에서 발견하지 못했다.','',
 '## 달라진 파일과 영향','',
 '| 파일 | 변경 | 학습 수치에 대한 판단 |','|---|---|---|',
 '| protocol.py | 허용 seed 범위, bootstrap seed, 분석 version | 학습 설정 생성 본문과 RNG seed 공식 동일. Bootstrap은 사후 통계 전용 |',
 '| train.py | CLI seed 범위, PID 기록, 학습 종료 후 checkpoint 감사 | run_training 호출 전후 학습 경로·함수는 동일. 추가 감사는 학습 종료 이후 |',
 '| launch.py | 최대3 worker, 자원 검사, 실패시 대기열 중단, polling, NUMBA_NUM_THREADS=1, pyc 쓰기 방지 | 실행 조건 변경은 존재. 새 조건이 원래 실행 조건과 같다고 주장할 수 없음 |',
 '| validate.py | smoke 검증 seed를 1→11로 이동 | 검증 대상만 변경 |',
 '| tests/test_assignment.py | 통합 분석 fixture를 새 seed 범위로 이동 | 테스트만 변경 |',
 '| freeze.py / source_manifest.json | 새 snapshot ID·origin 및 파일 hash | 추적/무결성 metadata 변경 |',
 '| README.md / PROTOCOL.md | 확장 범위 설명 추가 | 문서 변경 |',
 '| completion.py (추가) | final checkpoint/replay/optimizer/RNG 감사 | 학습 종료 후 검사 |','',
 '원본 대비 9개 파일이 변경되고 1개가 추가됐다. 별도 최상위 queue·분석·검증 wrapper도 추가됐다. 전체 diff는 [source_diff.patch](source_diff.patch), 파일 hash는 [file_hashes.csv](file_hashes.csv)에 있다.','',
 '## 실행 조건에서 중요한 미확인점','',
 '**추가 30회는 실제 CPU 실행임이 config와 run.log 양쪽에서 확인된다. 원본 launcher/config의 기본값은 CUDA지만, 과거 30회가 실제 어떤 --device 값으로 실행됐는지는 공개 run별 config/log가 없어 확인하지 못한다.**',
 '원본도 --device cpu를 지원하므로 기본값만 보고 GPU→CPU 변경이 실제로 일어났다고 단정할 수 없다. 반대로 두 cohort가 같은 device·package 환경이었다고 단정해서도 안 된다.',
 f"실행별 시작·종료 timestamp로 계산한 lifecycle 중첩은 과거 최대 {peak(old_states)}개, 새 cohort 최대 {peak(new_states)}개다. 과거에는 Support/Random/TD를 섞어 실행했고 새 실험은 사용자 요청대로 SABER→Random→TD 순서였다. 이는 보존된 attempt 기록의 동시 활성 수이며 개별 CPU/GPU kernel의 실제 동시 실행량을 측정한 것은 아니다.",
 'NUMBA_NUM_THREADS=1을 새 launcher가 명시했다. 보존된 support kernel은 parallel=True/prange를 쓰지 않는 직렬 njit 구현이어서 알고리즘의 선택식 변경으로 보이지 않지만 실행 환경 차이로 공개한다. OMP/MKL/OPENBLAS=1과 torch.set_num_threads(1)은 원본에도 있었다.','',
 '## 정확한 결론','',
 '**“핵심 학습 소스와 설정 생성식은 동일하다”는 확인된다. “전체 실행 코드가 동일하고 seed만 바뀌었다”는 표현은 정확하지 않다.** 실행·검증 wrapper는 수정됐고, 과거의 실제 device/package 동일성은 아직 검증되지 않았다.',
 '이 검사는 고정된 snapshot과 보존된 run 기록을 대조한 것이다. 과거 서버에 접속해 원 실행 환경이나 누락된 원시 config를 감사한 것은 아니다. 기존 학습 code와 결과 파일은 변경하지 않았다.']
 (ROOT/'REPORT_ko.md').write_text('\n'.join(lines)+'\n')
 write(ROOT/'manifest.json',dict(created_utc=result['checked_utc'],github_commit=commit,
      outputs_sha256={p.name:digest(p.read_bytes()) for p in ROOT.iterdir() if p.is_file() and p.name!='manifest.json'}))
 print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()

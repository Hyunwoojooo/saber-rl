# SABER 작업공간 안내

2026-10-13 정리 기준. 논문 자료, ICML 보완 설계, 과거 실행 소스와 원시 결과의 위치를 정리한다.

## 자료 위치

| 자료 | 위치 | 용도 |
| --- | --- | --- |
| AAAI 논문과 기술 부록 | [paper/](paper/) | 제출 원고, 그림, 표, 분석 코드와 부록 자료 |
| 제출용 코드·결과 패키지 | [code_supplementary/](code_supplementary/) | 익명 제출 패키지와 재현 자료 |
| 리뷰 이후 실험 현황 | [ICML_27/2026-10-06_post_review_experiment_status_ko.md](ICML_27/2026-10-06_post_review_experiment_status_ko.md) | 10월 6일 시점의 정적 보고서 |
| ICML 보완 실험 설계 | [ICML2027/](ICML2027/) | Policy-query 진단, 설정 민감도, 공통 context 전이 설계 |
| FPP 실험 | [experiment_FPP/](experiment_FPP/) | Backbone별 코드와 분석·논문용 자료 |
| DrawerOpen–DDPG 실험 | [metaworld_Drawer_Open_DDPG/](metaworld_Drawer_Open_DDPG/) | 공유 코드, 독립 실험 패키지, 결과 |
| DrawerClose–DDPG 실험 | [metaworld_Drawer_Close_DDPG/](metaworld_Drawer_Close_DDPG/) | 공유 코드, 독립 실험 패키지, 결과 |
| 기존 strict count-matched 실행 소스 | [strict_context_eligible_count_matched_2m/](metaworld_Drawer_Open_DDPG/strict_context_eligible_count_matched_2m/) | 이 PC에서 사용한 동결 소스와 검증 기록 |

`ICML_27/`는 기존 보고서 위치이고, `ICML2027/`는 후속 설계 위치다. 기존 링크와 실행 기록을 유지하기 위해 이름을 바꾸거나 합치지 않았다. 설계 폴더 이름 `experment_design`도 기존 경로를 유지한다.

## 이 PC에 별도로 있는 자료

- Policy-query 실험 실행 코드는 `~/ICML2027/draweropen_ddpg_policy_query/`에서 관리한다. 이 Git 저장소 밖에 있으므로 이 저장소를 clone하거나 pull하는 것만으로 받아지지 않는다.
- `results_2m/`, `runs/` 아래의 원시 로그와 checkpoint는 기존 위치에 보존한다. Git에서 추적하는 요약 자료와 실제 학습 상태 파일은 다르다.
- 10월 6일 보고서가 참조하는 `experiments_post_review/`는 이번 동기화 기준 원격 저장소에 포함되어 있지 않다.
- `machine_B_pip_freeze.txt`, `machine_B_torch_collect_env.txt`는 이 PC의 환경 기록으로 유지한다. `.git/info/exclude`에 명시해 일반적인 커밋 대상에서 제외했다.

공유 `code/`는 원격 최신 버전을 기준으로 한다. 기존 strict count-matched 실행을 확인할 때는 위의 독립 실행 소스와 해당 `freeze_manifest.json`을 사용한다. 과거 실행의 manifest를 현재 공유 코드에 맞춰 새로 생성하면 실행 당시의 근거가 달라진다.

### 확인된 기존 manifest 차이

`metaworld_Drawer_Open_DDPG/freeze_context_count_only_source.py --verify`는 최신 원격에서도 README 해시 때문에 실패한다. 나머지 32개 파일은 일치하며, manifest가 등록된 `bfc6e0c`에서는 README를 포함한 33개가 모두 일치한다. 이후 README에 다른 실험 설명이 추가된 데 따른 차이로 확인했다. 실행 당시의 manifest는 수정하지 않았다.

정리 후 context-only의 support 판정, critic/actor loss 범위, 설정 일치 테스트 3개와 독립 strict 실행 소스 38개 파일의 해시 검증은 통과했다. 원시 학습 데이터의 보존 확인은 별도의 로컬 정리 기록에 남겼다.

## 브랜치와 복구 자료

- `main`: 정리 시점의 `origin/main`과 같은 `84407b3`를 가리킨다.
- `work/icml2027`: 최신 원격 위에 이 PC의 미등록 실험 설계와 독립 실행 소스를 보존하는 작업 브랜치다.
- `backup/pre-sync-20261013t103306z`: 정리 전 로컬 `main`을 보존한다.
- `backup/local-files-20261013t103306z`: 정리 전 수정·미추적 파일을 보관한 stash commit을 가리킨다. 실제 복구 자료이며 개발 기준 브랜치로 사용하지 않는다.

파일 백업과 검증 기록은 이 PC의 `.local_backups/repo_sync_20261013T103306Z/`에 있다. `files/`에는 원본 174개, `local_changes.patch`에는 기존 수정 내역, `snapshot.json`에는 파일 해시와 보존 대상 목록이 있다. 이 폴더는 로컬 복구용이며 Git 추적 대상에서 제외했다.

백업에서 복구할 때는 필요한 파일을 골라 현재 버전과 비교한다. 저장된 stash 전체를 최신 트리에 적용하면 이미 원격에 반영된 파일과 다시 충돌할 수 있다.

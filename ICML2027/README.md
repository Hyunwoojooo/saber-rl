# ICML 2027 보완 실험

이 폴더는 AAAI 리뷰 이후 보완 실험의 설계, 실행 코드와 완료 자료를 관리한다. 전체 저장소의 자료 위치와 브랜치 안내는 [WORKSPACE.md](../WORKSPACE.md)에 있다.

## 미팅 자료

[2026-11-05 리뷰 이후 첫 미팅 자료](meetings/2026-11-05_post_review_first_meeting_ko.md)는 GitHub main의 공유 실험과 이 PC의 결과를 합쳐, 리뷰 내용·실험 목적·결과·남은 질문·다음 우선순위를 쉽게 설명한다. 문서 앞부분에 핵심 요약이 있으며, 뒤에는 출처와 회의 결정 항목을 두었다. 진행 상태는 확인 시점의 기록이다.

## 완료 자료 — 2026-11-05 공개

| 작업 | 완료 범위 | 자료 |
| --- | --- | --- |
| A1 초기 학습·actor 포화 추적 | DrawerOpen–DDPG, HER/SABER/Random/TD, seeds 101–103, 각 500k, 총 12회 | [완료 결과·검증 안내](paper_artifacts/a1_actor_dynamics_20261031/README.md), [실행 코드](experiments/draweropen_ddpg_actor_dynamics_20261031/README.md) |
| Assignment seed 확장 사후 진단 | 기존 seeds 11–20의 완료된 30회에 대한 로그·checkpoint·gradient·geometry 분석; 추가 학습 없음 | [한국어 보고서](diagnostics/assignment_extension_deep_audit_20261030/outputs/REPORT_ko.md), [분석 코드와 재현 범위](diagnostics/assignment_extension_deep_audit_20261030/README.md) |
| Seed 1–10 / 11–20 소스 재감사 | 학습 핵심 코드·설정 생성식과 실행 wrapper 차이 대조 | [감사 보고서](diagnostics/assignment_source_reaudit_20261030/REPORT_ko.md) |

A1은 2026-11-01 13:35 KST에 학습·자동 분석까지 완료됐다. 공개 묶음에는 모든 seed의 결과와 압축한 평가·진단 기록을 포함한다. 대용량 checkpoint는 로컬에 보존한다. 동결된 실험 코드와 source manifest를 변경하지 않고 별도 공개 artifact로 구성했다.

```bash
python3 ICML2027/paper_artifacts/a1_actor_dynamics_20261031/verify.py
```

위 명령은 공개 파일과 동결 소스의 해시, 평가 지표 및 포화 관측 시점을 검증한다. 새 학습을 시작하지 않는다. 500k 탐색 결과를 기존 2M 결과와 합산하지 않는다.

## 실험 설계

현재 완료 결과를 반영한 [2026-10-30 추가 실험 검토](experment_design/2026-10-30_additional_experiments/README_ko.md)를 먼저 참고한다. 기존 Q 진단·가중치 강도·SAC·seed 확장 결과와 남은 질문을 구분하고, 초기 정책 포화 진단·goal 입력 점검·context 전이의 우선순위를 정리했다. 새 학습 실행이나 예약을 뜻하지 않는다.

| 순서 | 문서 | 확인할 질문 |
| --- | --- | --- |
| 1 | [Policy-query 가치 진단](experment_design/01_policy_query_value_diagnostic.md) | 같은 정책과 데이터에서 critic 학습 배분을 바꾸면 정책 행동의 가치 예측이 개선되는가? |
| 2 | [Support 설정 민감도](experment_design/02_support_hyperparameter_sensitivity.md) | 주변 데이터 최소 개수와 추가 가중치 강도를 바꿔도 효과가 유지되는가? |
| 3 | [공통 context 전이](experment_design/03_common_context_transfer.md) | 환경별 수작업을 줄인 context 규칙을 다른 과제로 옮겨도 효과가 유지되는가? |

설계 문서의 경로와 내용은 이번 저장소 정리에서 그대로 보존했다. 문서에 적힌 실험 계획과 실제 완료 결과는 구분한다.

## 실행 코드와 결과 확인

최신 추가 실험 A1의 [초기 학습·actor 포화 추적 코드](experiments/draweropen_ddpg_actor_dynamics_20261031/README.md)는 이 저장소 안에서 관리한다. HER/SABER/Random/TD의 초기 상태와 학습 신호를 같은 방식으로 기록하는 별도 pilot이며, 기존 policy-query 진단과 구분한다. 실행 조건과 검증 기록은 해당 폴더를 참고한다.

1번 실험의 DrawerOpen–DDPG 실행 코드는 이 PC의 `~/ICML2027/draweropen_ddpg_policy_query/`에 별도로 있다. 해당 폴더의 `PROTOCOL.md`, 설정 파일과 검증 기록이 구현 상태를 설명한다. 본실험에 사용할 checkpoint의 출처·학습 시점·내용을 확인한 뒤 입력 목록을 구성해야 한다.

기존 [10월 6일 실험 현황 보고서](../ICML_27/2026-10-06_post_review_experiment_status_ko.md)는 작성 당시의 정적 기록이다. 원격 저장소에 없는 checkpoint나 `experiments_post_review/`의 원자료까지 포함한 패키지는 아니다.

기존 실험의 checkpoint·원시 로그·동결 소스 경로는 유지했다. 새 실행의 출력은 기존 결과와 구분되는 경로에 보관한다.

# ICML 2027 보완 실험

이 폴더는 AAAI 리뷰 이후 보완할 실험의 설계 문서를 관리한다. 전체 저장소의 자료 위치와 브랜치 안내는 [WORKSPACE.md](../WORKSPACE.md)에 있다.

## 실험 설계

| 순서 | 문서 | 확인할 질문 |
| --- | --- | --- |
| 1 | [Policy-query 가치 진단](experment_design/01_policy_query_value_diagnostic.md) | 같은 정책과 데이터에서 critic 학습 배분을 바꾸면 정책 행동의 가치 예측이 개선되는가? |
| 2 | [Support 설정 민감도](experment_design/02_support_hyperparameter_sensitivity.md) | 주변 데이터 최소 개수와 추가 가중치 강도를 바꿔도 효과가 유지되는가? |
| 3 | [공통 context 전이](experment_design/03_common_context_transfer.md) | 환경별 수작업을 줄인 context 규칙을 다른 과제로 옮겨도 효과가 유지되는가? |

설계 문서의 경로와 내용은 이번 저장소 정리에서 그대로 보존했다. 문서에 적힌 실험 계획과 실제 완료 결과는 구분한다.

## 실행 코드와 결과 확인

1번 실험의 DrawerOpen–DDPG 실행 코드는 이 PC의 `~/ICML2027/draweropen_ddpg_policy_query/`에 별도로 있다. 해당 폴더의 `PROTOCOL.md`, 설정 파일과 검증 기록이 구현 상태를 설명한다. 본실험에 사용할 checkpoint의 출처·학습 시점·내용을 확인한 뒤 입력 목록을 구성해야 한다.

기존 [10월 6일 실험 현황 보고서](../ICML_27/2026-10-06_post_review_experiment_status_ko.md)는 작성 당시의 정적 기록이다. 원격 저장소에 없는 checkpoint나 `experiments_post_review/`의 원자료까지 포함한 패키지는 아니다.

기존 실험의 checkpoint·원시 로그·동결 소스 경로는 유지했다. 새 실행의 출력은 기존 결과와 구분되는 경로에 보관한다.

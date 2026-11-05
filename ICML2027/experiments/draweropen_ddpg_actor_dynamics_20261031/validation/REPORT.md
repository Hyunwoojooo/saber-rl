# A1 구현 검증 결과

완료: 2026-10-31T03:33:40.706084+00:00

**단위 테스트 14개와 실제 smoke 5회가 모두 통과했다. 500k pilot은 시작하지 않았다.**

| 방법 | 계측 | 완료 steps | Actor / critic update 수 | 고정 checkpoint 수 |
|---|---|---:|---:|---:|
| her | on | 10,200 | 각각 5,201 | 7 |
| saber | on | 10,200 | 각각 5,201 | 7 |
| random | on | 10,200 | 각각 5,201 | 7 |
| td_topk | on | 10,200 | 각각 5,201 | 7 |
| saber | off | 10,200 | 각각 5,201 | 0 |

Seed 101의 기술 검증이며 성능 비교 결과로 집계하지 않는다. 총 training steps는 51,000이다. 별도 evaluation/probe rollouts는 이 수에 포함하지 않는다.

확인한 사항:

- 각 실행의 설정, 평가 grid, 모델·target·Adam·replay/index·RNG·checkpoint hash 검사 통과.
- SABER 계측 on/off: 모델과 optimizer, replay, 학습 RNG, 평가 이력, assignment window CSV가 동일.
- 네 방법의 4,800-step 학습 전 checkpoint 및 고정 probe bank가 동일.
- HER의 모든 관측 update에서 actual high-weight fraction과 coefficient variance가 0.
- 기존 실행 코드 26개와 원본 설정의 hash 유지 확인. Working copy에서 바꾼 native 파일은 4개.

Source manifest SHA256: `4fe8ef8fdde863802548ed992d918926ffb593f81ceb69529d2ecec1573eccab`

상세 기록: [results.json](results.json), [검증 gate](gate.json), [source 감사](source_audit.json), [원본 대비 코드 차이](native_changes.diff).

원시 실행과 상세 보고서는 `validation/runs/check_01/`에 보존하며 용량이 큰 checkpoint는 Git에서 제외한다.

검증은 초기 10,200 steps 범위다. 500k에서의 수치 안정성, 실패율 또는 SABER의 우위를 입증한 결과가 아니다. Simulator 상태 복원과 online continuation은 지원하지 않는다.

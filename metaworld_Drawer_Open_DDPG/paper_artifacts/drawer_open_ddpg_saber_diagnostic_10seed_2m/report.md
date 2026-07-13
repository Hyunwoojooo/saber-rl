# DrawerOpen DDPG SABER 10-seed diagnostic 보고서

Artifact ID: `DRAWEROPEN-DDPG-SABER-DIAGNOSTIC-10SEED-2M`

## 1. 데이터와 처리

Diagnostic cohort의 seed ID는 정확히 1–10이다. 모든 seed는 10,000부터 2,000,000까지 10k 간격의 평가점 200개와 고유 diagnostic snapshot 9,976개를 갖는다. 동일 step 중복은 분석 대상 값의 동일성을 검증한 뒤 마지막 raw 행을 선택했다. Snapshot은 자기상관이 있으므로 통계적 독립 표본으로 취급하지 않고 seed 내부 평균을 만든 뒤 seed를 독립 단위로 사용했다.

Seeds 1–5는 diagnostic logging 재실행이고 seeds 6–10은 별도의 extension이므로 하나의 matched launch가 아니다. 기존 seeds 1–5 비-diagnostic 결과는 aggregate 재현 reference로만 사용했다.

## 2. 성능

- Final success 평균: `0.890`
- Success AUC 평균: `0.6526`
- Hit90: `9/10` seeds
- Hit90 도달 seed만의 평균: `554444` steps

## 3. 성공 9-seed TD/residual

Support-positive의 업데이트 전 절대 TD error는 `0.3204`, support-negative는 `0.2739`로 비율은 `1.170x`다. 방향은 `9/9` seed에서 양수다.

같은 weighted minibatch에서 residual reduction은 support-positive `0.00839`, support-negative `0.00401`로 비율은 `2.09x`다. 이 값은 same-minibatch local optimization response이며 독립 batch 또는 unseen transition의 개선을 뜻하지 않는다.

## 4. Training-goal phase allocation

| Phase | Batch fraction | Support-positive within phase |
| --- | ---: | ---: |
| Approach | 17.53% | 73.64% |
| Contact | 4.84% | 70.48% |
| Opening | 4.47% | 64.10% |
| Open | 73.16% | 30.46% |

Phase는 observation과 `desired_goal_train`으로 정의되므로 HER relabeled goal을 포함한다. 따라서 `open`을 원래 환경 성공으로 해석해서는 안 된다.

## 5. Hit90 전후와 seed 7

Hit90 전후 결과는 시간적 기술 통계이며 인과 방향을 분리하지 않는다. Seed 7은 400–800k 구간부터 성공 seed와 분기하며, 마지막 400k support 평균은 `0.9928`, support≥0.95 snapshot 비율은 `99.15%`다. 이는 초기 접촉 탐색 실패 뒤 support가 비구분적으로 포화된 단일 failure case다.

## 6. Scanner와 통계 한계

원본 run은 Numba scanner를 요청했지만 diagnostic cohort에서 scanner enabled/warmed 값은 0이었다. Python fallback audit의 policy-support mismatch는 0이었으나, 이는 같은 512-candidate 제한 아래 구현 일치만 뜻한다. Local scan truncation이 높으므로 512/1024/full-scan label agreement가 별도로 필요하다.

Exploratory correlation은 성공 seed 9개만 사용하고 multiple-comparison correction을 하지 않았다. p-value는 가설 생성용이며 확증적 결과가 아니다.

## 7. Payload 제외

Checkpoint, raw `metrics.csv`, `metrics.jsonl`, run log 및 기타 JSONL은 포함하지 않았다. 원본 파일은 `source_input_manifest.json`의 SHA256으로만 식별한다.

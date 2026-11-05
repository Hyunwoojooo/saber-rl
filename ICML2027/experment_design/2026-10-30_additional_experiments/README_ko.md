# SABER 추가 실험 검토 — 2026-10-30

이 문서는 완료된 결과를 확인해 다음 연구 질문과 실행 후보를 정리한 **설계 제안**이다. 새 학습을 시작하거나 예약하지 않았다. 결과를 이미 본 뒤 작성했으므로 기존 자료에 대한 분석은 탐색적이다. 실행할 항목은 그 직전에 코드·설정·seed·종료 기준을 별도 protocol로 고정한다.

**우선순위는 정책 학습이 멈추는 경로를 확인하는 실험, support 배정의 직접 효과를 구분하는 실험, context 전이 실험 순으로 권한다.** MetaWorld goal 표현 점검은 이들과 병행할 수 있다. 학습을 더 많이 돌리는 것만으로 SABER에 유리한 결론이 보장되지는 않는다.

## 1. 현재 질문에 대한 답: 신뢰구간이 어느 정도여야 하는가?

95%는 흔히 쓰는 신뢰수준이다. 99%로 올려야 인정받는다는 기준은 없다. 같은 데이터에서 신뢰수준을 높이면 대개 구간은 더 넓어진다. 구간의 폭은 추정의 정밀도를 보여준다. [NIST 설명](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)

현재 차이를 해석할 때는 다음을 구분해야 한다.

| 질문 | 현재 답 |
| --- | --- |
| 이 결과를 논문에 보고할 수 있는가? | 가능하다. 평균·구간·실패 실행·적용 조건을 함께 보고한다. |
| 0을 포함하면 실험이 잘못됐는가? | 아니다. 우월성뿐 아니라 동등성도 확정하지 못한 결과일 수 있다. |
| SABER가 더 좋다는 강한 주장을 뒷받침하는가? | 이번 배정 비교에서는 일관되게 그렇지 않다. 추가 seed에서 평균 방향도 바뀌었다. |
| 신뢰수준을 95%에서 99%로 바꿔야 하는가? | 그럴 필요가 없다. 필요한 것은 질문에 맞는 정밀도와 재현성이다. |

예를 들어 assignment cohort의 SABER−TD AUC는 기존 1–10에서 **+0.117990, 95% CI [−0.054866, +0.324917]**, 추가 11–20에서 **−0.300553 [−0.530202, −0.083792]**, 두 실행 cohort를 50:50으로 합친 1–20에서 **−0.091281 [−0.237400, +0.051294]**다. 이것은 95%라는 숫자가 낮아서 생긴 문제가 아니다. 효과의 방향과 크기가 어떤 조건에서 유지되는지가 남은 질문이다.

이 percentile bootstrap 구간은 유한 표본의 근사치이며, 실행 cohort 간 환경 차이를 제거하지 않는다. “진짜 효과가 이 범위 안에 있을 확률이 95%”나 “SABER가 이길 확률이 95%”로 읽지 않는다. 통계적 문턱 하나를 연구의 합격 기준으로 삼지 않는다는 원칙도 중요하다. [ASA 원칙](https://www.amstat.org/docs/default-source/amstat-documents/p-valuestatement.pdf)

## 2. 이미 끝난 작업과 남은 공백

GitHub 자료는 `origin/main` **1c6f2872acb1bc9bccd81908a8ebcceab6392efc**에 고정했다. 원격 결과를 작업 브랜치에 합치지는 않았다. 아래 실행 집합은 소스·재사용·seed 범위가 달라 전부 독립 반복으로 합산하지 않는다.

| 작업 | 확인한 상태와 결과 | 다음 질문 |
| --- | --- | --- |
| Support / Random assignment / TD Top-k | 기존 1–10과 추가 11–20 완료. 통합 AUC는 S 0.480704, R 0.526545, TD 0.571985. 기존 양의 평균 차이가 추가 cohort에서 반전됐다. | 어떤 학습 경로에서 배정 방식의 효과가 달라지는가? |
| PAL | 기존 10-seed 비교 완료. S−PAL 평균 +0.107261, CI에 0 포함. | PAL을 새로 구현하는 제안은 중복이다. 필요하면 다른 task/backbone 한 조건으로 확장한다. |
| Tail / Random-512 | 온라인 10-seed와 양쪽 최종 checkpoint 진단 완료. Random retrieval−Tail AUC −0.222387 [−0.518372, +0.080932]. | 정적 support 누락률과 학습 유용성은 다른 지표다. 같은 실험을 다시 예약하지 않는다. |
| 가중치 강도 λ | 0.5 / 1 / 2, 설정당 3 seeds, 2M 완료. AUC 0.477973 / 0.745226 / 0.538023. 9개 cell 중 1개 재사용, 신규 8회. | 제한된 탐색 결과다. 광범위한 강건성이나 λ=1의 최적성을 입증하지 않는다. |
| Policy-query Q 진단 | HER/SABER 각 seed 1, 500k·1M·1.5M·2M 등에서 실시. Q 오차 개선이 시점에 따라 달랐다. 2M S의 MSE는 한 번 S 업데이트 후 183.2572→186.8298. | 같은 실험이 아직 없다고 제안하지 않는다. 대표성·query 범위·정책 개선과의 연결이 공백이다. |
| 최신 실패 원인 감사 | 추가 S의 최종 실패 6개 모두 검사한 query에서 tanh 도함수와 actor gradient가 0. Random 1개·TD 2개에도 같은 완전 포화 형태가 확인됐다. | 공통 실패 형태라는 사실과 SABER가 진입 위험에 미친 영향은 구분해야 한다. |
| DrawerOpen/Close SAC | HER/SABER/HGR/PER, task당 방법별 5 seeds 완료. Open S−HER +0.001457, Close −0.157312. 두 CI는 0 포함. | “SAC를 채워 넣으면 된다”는 단계는 지났다. 이득 부재와 실패 양상을 설명해야 한다. |
| Main-paper DrawerOpen DDPG 확장 | 별도 1–5/6–10 cohort의 HER/S/HGR/PER 완결 자료 존재. 통합 S−HER +0.250804 [−0.067643,+0.543933]. 일부 query control은 공개 snapshot에서 미완료. | 기존 HER 비교를 없는 것으로 취급하지 않는다. assignment cohort와 임의로 섞거나 같은 수치의 재등장을 독립 재현으로 세지 않는다. |
| N_local / context 전이 / 공정한 wall-clock | 이번 검토에서 해당 후속 실험의 완결 자료를 확인하지 못함. 기존 계획·구현 일부는 존재. | 아래 후보로 남긴다. 미확인은 모든 머신에서 미실행이라는 뜻이 아니다. |

세부 source 위치와 hash는 [evidence_manifest.json](evidence_manifest.json), 계획의 실행 단위와 예산은 [agenda.json](agenda.json)에 기록한다. 오래된 [10월 8일 세 설계 문서](../01_policy_query_value_diagnostic.md)의 “미실행”은 작성 당시 상태다.

## 3. 가장 먼저: 정책 학습이 언제 멈추는지 관찰하기

**쉽게 말하면:** 실패한 최종 모델의 손이 왜 계속 위로만 가는지는 확인했다. 이제 학습 중 어느 때 그렇게 굳었고, 그 직전에 critic 배정이 무엇을 바꿨는지 확인한다.

### 관찰과 가설

- 관찰: 추가 SABER의 실패 6개는 검사한 최종 original/HER query에서 actor gradient가 0이었다. 다만 모든 상태에서 영구적으로 0이라는 정리는 아니다.
- 가설 A: sparse-reward DDPG의 공통 불안정성이 주된 배경이고 loss 배정과의 관계는 약하다.
- 가설 B: support 배정이 critic의 행동 방향 gradient를 바꾸어, 일부 정책에서 포화 경로 진입을 촉진하거나 지연한다.
- 가설 C: native goal 표현이나 실행 환경과 배정 방식의 상호작용이 중요하다.
- 가설 D: 이번 두 cohort 차이는 작은 seed 집합의 변동을 상당 부분 반영한다.

DDPG의 sparse-reward 실패와 actor drift는 선행연구가 있다. 현재 관찰을 그 자체로 새로운 발견이라고 주장하지 않는다. 선행연구의 작은 환경에서는 critic의 행동 gradient 소실이 핵심인 반면, 현재 검사에서는 작지만 남은 critic gradient가 actor의 수치적 tanh 포화에서 차단되는 형태도 확인했다. 두 설명을 동일시하지 않는다. [Matheron 등](https://arxiv.org/abs/1911.11679)

### 제안 설계 A1: 초기부터 같은 계측으로 네 방법을 추적

| 항목 | 제안 |
| --- | --- |
| 대상 | DrawerOpen–DDPG |
| arm | HER / SABER / Context-Eligible Random assignment / TD-error Top-k |
| pilot | 새 seeds 101·102·103, 각 500k. 12회, 총 6M steps |
| 목적 | 계측 검증과 사건 순서 관찰. n=3으로 실패 확률이나 우월성을 확정하지 않음 |
| 본 연구 | pilot과 구분해 고정된 새 seed 집합과 2M 종료점을 실행 전에 결정. 예산·정밀도를 검토해 n 확정 |
| 보존 시점 | 초기화, 4,800 steps, 5k, 6k, 8k, 10k, 20k, 50k, 100k 이후 일정 간격 |
| 비교 단위 | 학습 seed. checkpoint·query·episode 수를 독립 학습 반복으로 세지 않음 |

모든 arm을 처음부터 같은 초기 seed·소프트웨어·장비 조건으로 실행한다. seed별 arm 실행 순서는 무작위화하고, 동시 실행 수와 계측량을 맞춘다. 과거처럼 방법별 전부를 다른 기간에 몰아 실행하는 계획은 피한다. 같은 seed가 학습 후 replay까지 같게 만든다는 주장은 하지 않는다.

5k부터 actor/critic 업데이트가 시작되고 10k까지 행동은 random이다. 따라서 **10k SABER checkpoint는 배정 개입 이전의 공통 상태가 아니다.** 공통 상태 분기가 필요하면 첫 업데이트 전 상태를 준비한다. 4,800은 horizon 200의 episode 경계 후보지만 실제 저장 순서·다음 reset RNG까지 확인해야 한다. simulator·환경 RNG·wrapper 상태까지 복원할 수 없는 파일을 온라인 분기의 동일 초기 상태로 사용하지 않는다. 단순한 fresh-run 비교에는 checkpoint 재개가 필요 없다.

### 측정할 값

- 고정 probe 상태에서 actor의 pre-tanh 값, |action|≥0.999 비율, float32에서 `1-action² == 0` 비율.
- 같은 query의 `∇a Q`, actor parameter gradient, 실제 Adam parameter 이동량. 샘플 단위 gradient와 batch 평균 gradient를 구분해 상쇄와 포화를 혼동하지 않는다.
- 실제 evaluation rollout의 손–손잡이 거리·접근·접촉, 원래 목표 성공 경험의 첫 시점, replay 행동 다양성.
- support-positive 비율, eligible 비율, informative update 비율, `k=|E|>0` 포화. 마지막 항목을 actor 출력 포화와 다른 이름으로 기록한다.
- 동일한 고정 probe bank와 별도 현재-policy rollout 둘 다 사용. 전자는 시간 비교, 후자는 실제 방문 분포를 설명한다.

정확한 actor gradient가 0인 시점과 거의 0인 시점을 구분하고, “붕괴 발생”을 정의할 probe 집합·지속 횟수·threshold는 pilot 이후 본 연구 전에 고정한다. 500k까지 사건이 없으면 그 범위에서 미발생한 것이며 장기 실패가 없다는 뜻은 아니다. 특정 arm만 유리하게 연장하거나 seed를 바꾸지 않는다.

**판단:** 시간적 선행만으로 인과를 확정하지 않는다. 초기에 배정에 따른 critic gradient 차이가 관측되면 A2의 동일 상태 개입으로 확인한다. 공통 실패가 보여도 SABER가 진입 빈도에 미치는 영향은 별도로 남는다.

## 4. 그다음: 같은 상태에서 배정만 바꾸면 정책에 유용한 방향이 달라지는가?

**쉽게 말하면:** 같은 시험지를 네 가지 방식으로 복습시켰을 때, 정책이 실제로 참고할 예측과 행동 방향이 어느 쪽에서 좋아지는지 비교한다.

### 제안 설계 A2: 초기 공통 checkpoint의 critic 분기

- A1의 개입 전 또는 사전에 지정한 checkpoint 전체를 사용한다. 실패한 실행의 best만 골라 “공통 초기 상태”라고 부르지 않는다.
- critic·Adam을 복제하고 actor·target·replay·HER minibatch·목표·Bellman target을 고정한다. E와 k도 같은 분기 입력에서 계산한다.
- uniform / support / random / TD Top-k로 critic만 1회, 별도로 고정된 10회 업데이트한다. `∇a Q(s,π(s),g)`와 독립 policy-query value 오차의 변화를 측정한다.
- 선택한 한 방법의 update 뒤 다른 방법을 연속 적용하지 않는다. 각 분기는 같은 원 상태에서 출발한다.
- actor-gradient 차이를 볼 때도 같은 actor를 쓴다. 실제 actor를 한 번 학습시키는 확장은 별도 분기에서 동일 actor/Adam으로 출발한다.
- 고정된 original goal 평가와 HER goal 평가는 분리한다. 실제 policy action의 transition/rollout으로 참조 return을 얻는다. replay action의 보상·다음 상태를 policy action의 정답으로 재사용하지 않는다.
- 현재 timeout bootstrap 정의에 맞춰 참조 return을 구성한다. 기존 방식처럼 긴 rollout을 쓰면 잔여 return bound를 보고한다. SAC까지 옮기면 entropy가 포함된 soft-Q와 reward-only return을 혼동하지 않는다.

과거 Q 진단은 이미 일부 시점에서 악화를 보여줬다. 이번 목적은 더 유리한 query를 고르는 것이 아니라, **초기 gradient 경로와 독립 query 예측을 함께 측정하는 것**이다. 여러 seed에서 사전 지정된 query bank를 사용하고, 불리한 시점·목표 도달 후 이탈 query도 보존한다. 기본 결과는 native Adam 업데이트로 제시하고, 실제 parameter 이동 norm을 맞춘 비교는 별도 진단으로 표시한다. 평균 coefficient=1만으로 이동량이 같아지지는 않는다.

평가 query가 모두 실패 return −50처럼 거의 상수면 측정은 유효하더라도 구분력이 제한된다. 그 사실을 보고하고, 이후 성공 checkpoint를 포함하는 bank는 성능을 보며 임의 선택하지 않고 고정 시점·모든 seed 규칙으로 만든다. Q값 오차가 줄어도 actor gradient 정확도가 자동으로 보장되지는 않는다.

**판단:** support의 추가 효과가 일관되면 적용 조건을 구체화한다. Random/TD와 유사하거나 악화되면 exact membership의 필수성 주장을 낮춘다. 단기 분기는 장기 AUC나 실패 확률의 대체 실험이 아니다.

### 선택적 A3: 완전히 굳은 actor를 풀면 학습이 재개되는가?

마지막 실패 checkpoint를 복제해 unchanged control과 actor의 **pre-tanh 값에 직접 작용하는** 제한된 개입 하나를 비교하는 복구 진단이다. 새 policy/target/optimizer 처리와 환경 reset·replay 보존 규칙을 명시한다. 결과는 과거 trajectory의 연장이 아니라 새 분기다.

이 실험은 포화가 실패 상태를 유지하는 데 관여하는지 묻는다. 최초 실패의 원인이나 원래 SABER의 우월성을 입증하지 않는다. tanh 뒤 action penalty나 단순 gradient clipping은 이미 0인 tanh 미분을 되살리지 못할 수 있다. 광범위한 actor reset/regularization 탐색을 먼저 시작하지 않는다. 안정화 변형을 성능 비교하려면 HER·SABER·다른 loss arm 모두에 같은 개입을 적용한 별도 연구로 만든다.

## 5. 병행 점검: MetaWorld 입력에 서로 다른 두 목표가 들어가는가?

최신 SAC 공개 source의 알려진 한계는 native observation 마지막 세 성분의 원래 목표와 HER로 바뀐 별도 goal이 함께 입력된다는 점이다. 로컬 DDPG wrapper도 observation을 그대로 보존하고 goal을 이어 붙인다. DDPG의 실제 native tail 값과 goal 표시 옵션은 먼저 확인해야 하며, 모든 환경에서 문제가 동일하다고 단정하지 않는다.

### 제안 B1: 먼저 학습 없는 입력 검사

같은 물리 상태에서 original/HER goal을 바꾼 input을 기록해 어떤 성분이 목표인지 확인한다. native observation의 설정·차원·goal 의미와 재계산 reward를 검사한다. goal을 원래 goal로 되돌리면 원 입력으로 돌아오는지도 본다. 단순히 두 숫자가 다르다는 이유만으로 성능 하락의 원인을 확정하지 않는다.

### 조건부 B2: representation × loss의 작은 matched 실험

| 요인 | 수준 |
| --- | --- |
| 입력 표현 | 기존 그대로 / native 목표 세 성분을 항상 0으로 가리고 별도 goal만 사용 |
| critic loss | HER / SABER |
| 시작 조건 | DrawerOpen–DDPG 또는 공동저자와 범위를 조정한 DrawerClose–SAC 한 조건 |
| pilot 예산 예시 | 2×2×3 seeds×500k = 12회, 6M steps |

입력 차원을 유지해 network 크기 차이를 줄인다. 가림은 collection·current/next state·HER·actor·target·support query·평가 모두에 일관되게 적용한다. 보상·success·HER 목표 추출은 바꾸지 않는다. 그 세 성분에 물리 상태 정보가 섞이지 않았는지 먼저 확인한다.

기존 표현과 다른 protocol이므로 별도 변형이다. 기존 불리한 SAC 결과를 지우거나 새 결과와 몰래 합치지 않는다. 핵심은 SABER의 효과 차이, 즉 `(S−HER)_masked − (S−HER)_legacy`이며, 한 방법만 가렸을 때의 개선을 비교 근거로 쓰지 않는다. pilot 길이는 correctness와 신호 탐색용이다. 2M 성능 효과를 말하려면 별도 고정된 본 평가가 필요하다.

## 6. 여전히 중요한 다음 실험: context를 줄여서 옮기기

이것이 사람 리뷰어의 “새 환경마다 수작업이 얼마나 필요한가?”에 가장 직접적으로 답한다. 기존 [공통 context 설계](../03_common_context_transfer.md)를 출발점으로 삼되 현재 실패/표현 문제와 실행 코드 차이를 반영해 새 버전을 고정한다.

- HER / 기존 SABER / 단순한 공통 context SABER의 세 arm.
- FetchPickAndPlace에서 개발하고, DrawerOpen 결과를 보고 규칙을 다시 바꾸지 않는 전이 평가.
- 관측 adapter, 손–물체 거리/성공 거리, 물체–목표 거리/성공 거리, 남은 시간 비율처럼 유지할 정보를 명시한다. 수작업 지식이 완전히 없다고 부르지 않는다.
- coarse key에서 제거한 phase/gripper 정보를 full key 바깥에서 계속 사용하지 않는지 검사한다. 후보의 full context는 query의 goal로 재계산한다.
- source pilot 예시: 3 arm×3 seeds×1M = 9회, 9M steps. target의 main budget은 source를 본 뒤, target 결과를 보기 전에 고정한다.
- 전이 결과가 낮으면 그대로 보고한다. task마다 bin을 다시 고르면 고정 규칙 전이의 증거가 아니다.

DDPG 포화가 모든 arm의 결과를 지배하면 대표 TD3 조건에서 확인하는 대안을 검토한다. TD3가 실패하지 않는다고 가정하지는 않는다. backbone을 바꾸는 경우 기존 DDPG 결과와 다른 cohort로 보고한다.

PointMaze는 조작 밖의 후순위 후보다. 기존 손–물체 context를 그대로 적용할 수 없으므로 추가 환경 숫자만 늘리지 않는다. 사용할 공통 표현부터 정해야 한다. 공식 sparse reward가 0/+1이고 continuing/goal-reset 설정도 있어 Drawer의 −1/0·종료 처리를 무심코 복사하지 않는다. [Farama 문서](https://robotics.farama.org/envs/maze/point_maze/)

## 7. 민감도·검색·비용에서 추가로 할 것

| 후보 | 정확히 확인할 것 | 우선 범위 |
| --- | --- | --- |
| N_local=4/8/16 | population 조건의 실제 threshold가 달라져도 효과가 유지되는가? | 기존 λ 실험 다음의 주 민감도 후보. 대표 task/backbone 하나에서 시작 |
| N_scan=128/512/2048 | 검색 누락·배정·시간의 trade-off | 먼저 같은 checkpoint의 정적 query를 비교. online 실험은 그 질문이 남을 때 |
| context bin을 성기게/촘촘하게 | 수동 경계에 의존하는가? | feature와 retrieval 규칙을 유지한 한 요인 변화 |
| context-only | policy action을 전혀 조회하지 않는 population 신호만으로 충분한가? | 기존 구현과 실행 여부를 먼저 감사. 소스만 보고 새 실험 또는 완료 실험이라고 단정하지 않음 |
| 다른 backbone의 mechanism control | 현재 DDPG 배정 결과가 다른 구조에서도 유지되는가? | TD3 대표 task 하나. SAC 일반 성능표를 다시 채우는 작업과 구분 |
| 공정한 wall-clock | support overhead를 포함해 실제 시간 대비 이득이 있는가? | HER/S/PAL, 같은 하드웨어·thread·동시 부하·계측. 먼저 고정 길이 microbenchmark |

N_local 판정은 `n≥N_local`과 `min(n,M_B)≥M_Q_min`을 함께 쓴다. M_Q_min=4에서 N_local 1과 4는 같은 조건이 될 수 있다. 또한 현재 frozen config는 요청값 8을 4로 바꿀 수 있다. 새 실험에서는 baseline parity를 유지하면서 명시값을 보존하는 별도 config 경로를 만들고 **requested=effective**를 검증한다. 변경 없이 CLI 숫자만 바꾼 실행은 sensitivity 근거가 아니다.

Wall-clock에는 index 구축·갱신·cache·조회·학습·동일 평가를 포함한다. 학습 없이 고정 replay로 측정한 lookup 속도와 online 총 비용은 구분한다. 지금 assignment arm의 가상 mask 계측과 PAL의 무계측 시간을 그대로 비교하지 않는다. Support 조회 비용이 큰 경우 matched-UTD 이득과 시간 효율 이득을 따로 보고한다.

## 8. 연구 순서와 잠정 예산

다음 표의 숫자는 **실행 예시**이며 검정력 계산으로 충분성이 확인된 n이 아니다. 항목 전체를 한꺼번에 실행하라는 뜻도 아니다. 기존 연구와의 재사용 감사가 끝나기 전에는 baseline 절감분을 약속하지 않는다.

| 순서 | 작업 | 신규 본학습 예산 예시 | 다음 결정 |
| --- | --- | ---: | --- |
| 0 | 완료 자료 통합, goal 입력 검사, checkpoint 계측 검증 | 0 | 측정이 맞고 중복 실행이 없는지 |
| 1 | A1 초기 actor/critic 경로 pilot | 12×500k = 6M | 사건 순서와 probe의 구분력을 확인 |
| 2 | A2 동일 상태 critic 분기 | 새 online training 0; 별도 rollout/update 진단 | support 배정이 gradient/Q 예측에 미치는 직접 차이를 확인 |
| 조건부 | B2 goal 표현 pilot | 12×500k = 6M | 표현과 loss 사이 상호작용을 조사 |
| 3 | 공통 context source pilot | 9×1M = 9M | 규칙을 고정하고 target 예산을 결정 |
| 후속 | N_local pilot | 3값×3 seeds×2M = 최대 18M | 실제 threshold가 바뀌는지와 성능 변동 확인 |

우선 실행 후보 하나를 고른다면 **A1의 계측 코드부터** 권한다. 단, B1의 입력 검사는 학습 없이 병행할 수 있다. 단순히 남는 컴퓨터를 채우기 위해 A1·B2·전이·민감도를 동시에 시작하면 결과 해석과 코드 버전 관리가 더 어려워진다.

본 평가의 seed 수는 “20이면 충분”처럼 정하지 않는다. 의미 있는 AUC 차이와 원하는 CI 정밀도, 실제 paired variance, 비용을 바탕으로 별도 결정한다. 기존·추가 cohort를 구분하고, seed를 유리한 구간이 나올 때까지 순차 추가하지 않는다. 평가 episode를 늘리는 것은 독립 학습 seed를 늘리는 것과 다르다. [RL 평가 연구](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)

## 9. 이번에는 우선하지 않을 것

- 21–30 seed를 바로 추가해 CI가 양수가 되는지 기다리기.
- 이미 끝난 PAL, λ=0.5/1/2 pilot, Tail-vs-Random-512, Drawer SAC 성능표를 “아직 없는 실험”으로 반복하기.
- 모든 task×backbone 조합에 동일 mechanism control을 일괄 추가하기.
- actor가 0 gradient인 final checkpoint의 replay만 바꿔 초기 실패 원인이 밝혀졌다고 주장하기.
- 실패 seed를 제외한 AUC나 가장 유리한 checkpoint를 primary로 바꾸기.
- 큰 support-positive 비율을 좋은 정책·정확한 Q의 지표로 사용하기.
- 기존 CPU/GPU 설정 기록이 없는 상태에서 “원래는 GPU였기 때문에”라고 설명하기.

원래 실행 환경이 확보되면 같은 seed의 장비/패키지 bridge는 별도로 가치가 있다. 현재는 default device와 실제 실행 device를 혼동하면 안 된다. frozen source 62개 보존본과 GitHub가 일치하고 core `code/`·`upstream/`가 동일하다는 감사는 있지만, wrapper 전체와 과거 실행 환경까지 동일하다는 뜻은 아니다.

## 10. 측정·통계·실행에서 지킬 공통 규칙

1. AUC는 해당 protocol의 10k–2M 전체 평가 grid와 정확한 분모를 유지한다. pilot AUC를 2M 결과와 직접 섞지 않는다.
2. loss·target·actor gradient·Adam 이동을 각각 기록한다. 학습 배치 residual 감소와 독립 Q 정확도를 구분한다.
3. 실제 evaluation geometry는 evaluation JSONL/새 전용 필드로 읽는다. 기존 `metrics.csv`의 일부 거리·phase 값은 마지막 HER minibatch에서 남은 값이었다. primary success/AUC가 잘못됐다는 뜻은 아니다.
4. 모든 finite zero-success run을 보존한다. 수치 오류와 성능 실패를 구분한다. 인프라 중단의 재시도는 별도 attempt로 처음부터 수행한다.
5. 새로운 diagnostic continuation은 원 main training의 재개 금지 규칙과 구분된 별도 코드·protocol에만 구현한다. 원본 checkpoint와 frozen 결과를 덮어쓰지 않는다.
6. 전체 config·패키지·장비·RNG·실행 명령·계측량·source hash를 보존한다. 진단 RNG는 replay/exploration stream을 소비하지 않는다.
7. primary contrast를 소수로 정하고 다중 비교와 탐색적 결과를 표시한다. 95% CI의 0 포함 여부만으로 모든 주장을 결정하지 않는다.

## 11. 원고에 돌아갈 결론의 형태

필요한 것은 “모든 곳에서 좋은 알고리즘”을 보장하는 실험이 아니다. **Support가 유용한 학습 배분 신호가 되는 조건과 실패하는 조건을 직접 구별하는 증거**다. 개선이 재현되지 않으면 그 결과를 포함해 방법·적용 범위·기여 문장을 바꿔야 한다. loss-side weighting 자체는 PAL 등 선행연구가 있으므로 일반적인 재가중의 신규성만 앞세우지 않는다. [Fujimoto 등](https://proceedings.neurips.cc/paper/2020/hash/a3bf6e4db673b6449c2f7d13ee6ec9c0-Abstract.html)

실험과 별개로 method의 짧은 예시·pseudocode·context 표, policy-action 조회와 replay-action 회귀를 구분한 그림, 진단 결과표를 준비한다. 이는 두 사람 리뷰어가 공통적으로 지적한 가독성 문제에 대한 작업이며 신규 학습으로 대신할 수 없다.

### 자료와 작성 방법

- 로컬 완료 자료: [추가 assignment 결과](/home/mimiclab-gigabyte-gen5/ICML2027/draweropen_ddpg_assignment_extension_20261020/outputs/final/REPORT.md), [실패 심층 감사](../../diagnostics/assignment_extension_deep_audit_20261030/outputs/REPORT_ko.md), [소스 재감사](../../diagnostics/assignment_source_reaudit_20261030/REPORT_ko.md), [λ 결과](/home/mimiclab-gigabyte-gen5/ICML2027/draweropen_ddpg_lambda_sensitivity_20261015/outputs/analysis/REPORT.md), [Q 오차 분석](/home/mimiclab-gigabyte-gen5/ICML2027/draweropen_ddpg_q_error_audit_20261015/FINDINGS_ko.md).
- 원격 완료 자료는 evidence manifest의 고정 commit과 path로 식별한다. 일부 원본 checkpoint는 로컬에만 있고 공개 archive에는 없다.
- 검색일: 2026-10-30. DDPG sparse-reward deadlock, PAL, RL 통계 평가의 원 논문과 공식 환경/통계 문서에 한정한 목적별 조회다. 체계적 문헌검색이나 신규성 부재의 증명이 아니다. 비공개 실험 수치는 외부 검색어로 보내지 않았다.
- `experimental-design` 및 `hypothesis-generation` 스킬을 사용해 비교 단위·고정 조건·대안 가설·불확실성·실행 전 고정 항목을 정리했다. 문서 구조 검증은 과학적 타당성이나 결과를 보증하지 않는다.
- 도구 방법론 참고: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://arxiv.org/abs/2609.00065). 최신 메타데이터를 2026-10-30 확인했다. 이는 SABER의 방법론·성능 근거가 아니다.

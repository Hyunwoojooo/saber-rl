# SABER AAAI 심사 대응 및 후속 실험 인계 보고서

- 대상 논문: **SABER: Support-Conditioned Adaptive Bellman Error Reweighting for Sparse-Reward Hindsight Experience Replay**
- 작성일: **2026-10-27**
- 실행 상태 기준: **2026-10-27 05:30:13 UTC / 14:30:13 KST**
- 수신 대상: ICML 재투고 준비에 참여하는 공동연구자
- 기준 원고: AAAI 제출본 9쪽 PDF. 1-7쪽 본문, 8-9쪽 참고문헌. 아래 페이지 번호는 이 PDF 기준이다.
- 공개 결과 기준: saber-rl/main 커밋 **a3c049464704ed2104f571ffc9b20925a339b51f**. 진행 중인 결과는 로컬 실행 기록을 별도로 확인했다.
- 범위: 전달받은 AAAI Phase 1 리뷰, 기존 본문의 수정 필요 사항, 9월 25일 이후 이 작업공간과 공개 아카이브에서 확인한 실험, 현재 실행·예약 상태. 다른 머신의 미공유 실행은 포함하지 않는다.

이 보고서는 연간 일정표가 아니라 **리뷰 지적과 증거 사이의 연결을 설명하는 인계 문서**다. 후속 실험의 유리한 결과와 불리한 결과를 모두 포함하며, 실험으로 직접 확인한 사실과 아직 검증하지 않은 설명을 구분한다. 완료 결과의 평균·표준편차·paired confidence interval(CI), 100회 신규 완료 학습의 seed별 지표를 문서 안에 수록했다. 전체 checkpoint 곡선과 세부 진단 CSV는 말미의 고정 커밋 링크에서 확인할 수 있다. 원시 체크포인트나 수백만 행 학습 로그를 문서에 복제하지는 않는다.

## 1. 핵심 요약

AAAI Phase 1에서 사람 리뷰어 두 명은 각각 **4점과 6점, 확신도는 모두 3**을 부여했다. 두 사람은 작은 개입 범위와 실험 통제를 인정했고, 공통적으로 가독성을 문제 삼았다. 부정 리뷰는 특히 신규성과 support 판정의 원리적 정당성을, 긍정 리뷰는 수작업 context와 민감도·후보 조회 규칙을 우려했다. AI 리뷰는 가까운 선행방법, mechanism의 불확실성, 통계·일반성·비용 문제를 추가로 지적했다. 결정은 Phase 2에 진입하지 못한 Reject였지만, 공개된 내용만으로 특정 리뷰가 최종 결정을 좌우했다고 단정할 수는 없다.

후속 작업의 핵심 질문은 다음 세 가지다.

1. **같은 HER 학습 조건에서 support 기반 critic weighting이 가까운 대안보다 어떤 추가 가치를 주는가?**
2. **정확한 sample 배정, context 조건, 조회 제한 중 무엇이 결과에 기여하며, 어디까지 분리해서 설명할 수 있는가?**
3. **방법이 도움이 되지 않거나 학습이 붕괴하는 조건까지 포함해 적용 범위를 설명할 수 있는가?**

현재까지 고유 신규 본학습 **100회가 완료**되었고, 모두 실행당 2M 환경 step이므로 완료 예산은 **2억 step**이다. HGR 5회가 진행 중이고 25회가 예약 대기 중이다. Smoke, 중단된 미완료 attempt, 과거 논문 결과의 재사용, 오프라인 분석은 신규 완료 횟수에 중복 산입하지 않았다.

DrawerOpen-DDPG의 새 배정 비교, PAL 비교, Tail/Random-512 비교에서는 SABER/Tail의 평균 AUC가 높았지만 각각의 주 비교 95% CI는 모두 0을 포함한다. 이는 긍정적인 평균 방향과 상당한 추정 불확실성이 함께 있다는 의미다. 반대로 새 SAC 결과에서는 HER 대비 이득 부재와 DrawerClose의 늦은 학습 붕괴가 확인되었다. 따라서 현 단계에서 가장 필요한 것은 우위를 포괄적으로 선언하는 것이 아니라, **본문을 더 쉽게 쓰면서 주장과 근거의 범위를 일치시키는 작업**이다.

## 2. 어떤 리뷰를 받았는가

### 2.1 심사 결과와 인정받은 부분

| 리뷰 | 인정한 부분 | 주요 우려 | 이 문서에서의 대응 위치 |
|---|---|---|---|
| wpgC, 4점, confidence 3 | HER 주변 파이프라인을 바꾸지 않는 작은 개입, 비교 통제, 경쟁력 있는 실험 성능 | 일반적인 reweighting 대비 신규성, binary heuristic의 정당성, 복잡한 수식·정의 전개 | 3-4절, 6.1-6.2절, 8절 |
| mK8N, 6점, confidence 3 | 기술적으로 타당한 개입, 충실한 비교·ablation·supplement; 주요 방법론적·실험적 결함은 발견하지 못했다고 명시 | §3.3 가독성, task별 context와 threshold, 일부 hyperparameter 민감도 부족, Tail 선택 근거, §4.3 숫자 나열 | 4절, 6.3-6.6절, 8절 |
| AI review, 별도 점수 없음 | 통제된 critic-loss 개입, mean-one normalization, AUC·paired 비교, 진단 설계 | 작은 표본과 변동성, PAL·actor-aware replay 관련성, exact occupancy의 기여 미분리, 좁은 적용 범위, 조회 누락, 시간 비용 | 3-9절 |

리뷰 요지는 연구자가 전달한 OpenReview 원문을 바탕으로 요약했다. 복사된 리뷰에서 일부 수식 기호가 누락되어 있어, 이를 임의의 직접 인용으로 복원하지 않았다. 구체적인 parameter 이름은 현재 원고의 정의와 연결해 설명한다.

### 2.2 리뷰를 정확히 받아들여야 하는 부분

- **가독성은 두 사람의 공통 문제 제기다.** 추가 실험만으로 해결되지 않는다.
- **“reweighting을 한다”는 사실만으로는 신규성이 충분하지 않다.** 기존 loss-side 방법과 비교하고, SABER 신호 자체의 구체적인 차이를 설명해야 한다.
- **“왜 이 support 조건인가”는 실질적인 질문이다.** binary gate, 수작업 context, 최소 개수, action bin, 가중치 강도가 성능과 어떤 관계인지 밝혀야 한다.
- **mechanism 비교의 CI가 0을 포함한다는 지적은 여전히 중요하다.** 효과가 없다는 증거는 아니지만, exact occupancy의 추가 가치를 확정적으로 주장하기 어렵게 한다.
- **Tail과 512개 제한은 구현 세부이면서 신호의 의미를 바꾸는 요소다.** 전체 replay의 support와 조회된 부분집합의 support를 구분해야 한다.
- **동일 UTD는 동일 시간 비용이 아니다.** 효율을 주장하려면 별도 조건 통제가 필요하다.

### 2.3 리뷰를 과도하게 확대해서는 안 되는 부분

기존 본문은 이미 support를 uncertainty·density·target accuracy 추정량이라고 주장하지 않았고, strict control의 CI가 0을 포함한다는 점과 HGR/PER의 우세 조건을 인정했다. 따라서 “원고가 모든 mechanism을 증명했다고 주장했다”는 식의 재서술은 부정확하다. 문제는 그 한계 인정에도 불구하고 **핵심 설계의 추가 가치를 충분히 설득했는가**다.

또한 기존 본문에는 FPP-TD3의 10-seed 확장과 HER convention 비교가 있었고, supplement에는 uniform replay를 유지하는 TD-error loss weighting, K 민감도, runtime 자료가 이미 있었다. “모든 실험이 5-seed였고 같은 개입 지점의 대조군이 전혀 없었다”는 식으로 정리하면 안 된다. 다만 이 자료들이 다른 task/backbone의 불확실성, PAL 직접 비교, policy-action 정확도까지 해결한 것은 아니다.

SAC Drawer 실험의 부재는 기존 본문이 공개한 한계였다. 사람 리뷰어가 이 누락 자체를 핵심 reject 사유로 명시한 것은 아니다. 후속 SAC 실험은 평가 범위를 채우기 위한 작업이며, 그 결과에서 새로운 적용 한계가 드러났다.

## 3. SABER의 현재 정의와 아직 열린 연결

### 3.1 방법의 핵심

HER로 목표를 바꾼 minibatch에서 SABER는 각 샘플의 관측·목표·남은 시간에 따른 이산 context를 만든다. 같은 coarse retrieval key에 해당하는 유효 replay 후보의 유지된 index 순서에서 최대 512개를 조회하고, 그 안에서 full context를 정확히 맞춘다. **Tail은 이미 support-positive인 샘플 중 마지막 512개를 고르는 것이 아니며, 시간적으로 가장 최근이라는 해석도 보장하지 않는다.**

같은 full context의 후보가 최소 4개이고, 그중 하나 이상이 현재 정책 행동과 같은 다차원 action bin에 속하면 support-positive로 판정한다. 기본 K는 5다. DDPG/TD3는 deterministic policy action, SAC는 stochastic actor의 squashed mean을 조회에 사용한다.

개념적인 가중치 정의는 다음과 같다.

~~~text
psi_i = 1{local_count_i >= N_local}
        * 1{policy_action_bin_i is occupied in the retrieved local context}

raw_weight_i = 1 + lambda_S * psi_i
Omega_i = raw_weight_i / mean_batch(raw_weight)

L_critic = mean_i [Omega_i * sum_k (Q_k(o_i, g_tilde_i, replay_action_i) - target_ik)^2]
~~~

기본 lambda_S=1이면 raw weight는 2 또는 1이며, 평균 1로 정규화한다. Critic의 Bellman regression은 **현재 정책 행동이 아니라 원래 replay action**에서 계산한다. Actor objective, Bellman target, HER 규칙, uniform replay sampling과 UTD는 유지한다. 단, 이는 목적함수·업데이트 규칙을 바꾸지 않는다는 뜻이지, 이후의 actor parameter나 replay trajectory까지 동일하다는 뜻은 아니다.

### 3.2 이미 명확한 성질과 아직 입증되지 않은 성질

| 구분 | 현재 설명 가능한 내용 |
|---|---|
| 정의에서 바로 확인되는 성질 | 모든 sample을 양의 계수로 유지한다. 평균 계수는 1이다. 모든 indicator가 같으면 uniform weighting으로 돌아간다. |
| 정의만으로 보장되지 않는 성질 | 수치 loss나 gradient norm이 동일하다, Q 오차가 줄어든다, actor gradient가 더 정확해진다, binary gate와 2:1 비율이 최적이다. |
| 기존 fixed-target 진단이 측정한 것 | 같은 학습 minibatch에서 한 번의 critic update 후 residual 감소가 어떻게 배분되는가. |
| 추가 검증이 필요한 것 | Replay action의 weighted update가 별도 policy-action 위치의 critic 예측과 이후 학습에 도움이 되는가. |

열린 연결은 다음과 같다.

~~~text
조회 범위에서 policy action 주변의 replay evidence가 관찰됨
    -> 다른 replay-action residual에 더 큰 가중치를 부여
    -> 함수 근사를 통해 policy-action 위치의 예측에도 이득
    -> actor 학습과 task 성능에 이득
~~~

첫 번째 관찰과 loss 배정은 구현·진단으로 확인할 수 있다. 뒤의 연결에는 추가 가정과 증거가 필요하다. 현재 one-update residual 진단만으로 이 전체 연결이 성립한다고 해석하면 안 된다.

### 3.3 가까운 선행방법과 구별할 질문

- **PAL:** uniform replay를 유지하면서 critic loss를 바꾸는 구체적인 선행방법이다. “큰 TD error 경험을 더 자주 뽑는다”는 설명은 PAL 자체에는 맞지 않는다. 그 설명은 sampling-side 방법인 PER/LAP와 구분해야 한다. [Fujimoto et al., 2020][PAL-PAPER]
- **PER/HGR:** replay transition 또는 hindsight goal 선택 분포를 바꾸는 baseline이다. SABER의 loss-side 개입과 다른 범위를 비교한다.
- **TD-error Top-k:** 이번 연구에서 선택 신호를 분리하기 위해 만든 count-matched control이다. PAL 재현이나 TD3 backbone을 뜻하지 않는다.
- **Actor Prioritized Experience Replay:** actor와 critic에 적합한 경험의 차이를 다룬 관련연구다. SABER와 동일한 알고리즘은 아니지만 actor를 고려한 replay 활용의 동기에서 비교할 필요가 있다. [Saglam et al., 2023][ACTOR-PAPER]
- **LaBER:** replay sampling을 gradient 추정의 importance sampling 문제로 전개한다. 문제·판단 기준·근사·구현 순서의 서술은 참고할 수 있지만, 그 분산 최소화 근거를 SABER의 변경된 weighted objective에 그대로 적용할 수는 없다. [Lahire et al., 2022][LABER-PAPER]

PAL 성능 비교를 추가했다는 사실만으로 신규성이 확보되지는 않는다. **HER 이후의 local policy-action support 신호가 더 단순한 신호와 어떤 차이를 만들고, 어떤 조건에서 의미가 있는가**가 방어해야 할 중심 질문이다.

## 4. 기존 본문에서 무엇을 어떻게 고쳐야 하는가

다음 항목은 수정 제안이며, 이 보고서를 작성하면서 제출된 PDF나 TeX를 변경하지 않았다. 위치는 [AAAI 제출 PDF][MAIN]를 기준으로 한다.

### 4.1 위치별 수정 목록

| 현재 위치 | 현재 서술·구성의 문제 | 구체적인 수정 |
|---|---|---|
| Abstract, p.1 | 5-seed DrawerOpen 개선 0.207→0.754가 대표 결과이며 “distinct design axis”가 크게 강조됨 | 최종 사용할 실행 집합을 명시하고 최신 결과로 갱신. Loss reweighting 자체의 최초성보다 support 신호의 정의·통제된 검증·조건별 차이를 중심으로 작성 |
| Introduction, pp.1-2 | 바뀌지 않는 pipeline 요소를 반복하고, support와 policy-query 이득 사이 연결이 직관에 머묾 | 기존 HER·PER·PAL의 역할을 먼저 인정한 뒤 남은 질문을 한 문단으로 제시. Replay-action fitting이 policy-action 평가로 전달된다는 부분은 가설로 표시 |
| Contribution bullets, p.2 | “post-HER loss allocation을 별도 축으로 정식화”가 가장 앞에 있음 | 구체적 signal, 이를 분리하는 controlled comparison, 적용 범위·실패 분석의 세 항목으로 재작성 |
| Related Work, p.2 | Fujimoto 2020 인용은 있으나 PAL과의 직접 연결이 약하고 actor-aware prior가 빠짐 | Sampling / loss allocation / actor-aware replay / offline support를 구분한 짧은 비교표. PAL 실험으로 바로 연결 |
| §3.1, pp.2-3 | 문제 설정, HER indexing, backbone 설명이 한 번에 길게 전개됨 | 원래 replay action과 policy-query action의 차이를 먼저 그림으로 설명. 기호는 실제 사용 직전에 정의하고 backbone의 공통·차이점은 작은 표로 압축 |
| §3.2-3.3, p.3 | C, coarse key, ordered candidates, scan set, exact subset, action-bin set을 거쳐야 핵심 규칙을 이해함 | “같은 context 경험이 충분한가 / policy action bin이 관찰됐는가”의 두 질문부터 설명. Indicator와 loss를 먼저 제시하고 index 유지·slot 검증 등은 뒤 또는 appendix로 이동 |
| Figure 1, p.1 | 두 action의 역할을 그림만으로 추적하기에는 제한적 | Support-query action 경로와 replay-action residual 경로를 별도 화살표로 표시. Regression action을 바꾸지 않는다는 점을 눈으로 확인 가능하게 구성 |
| §3.3, p.3 및 §4.5, p.7 | “bound lookup cost / bounded replay retrieval”이 전체 조회 비용의 엄격한 상한처럼 읽힐 수 있음 | “bounded post-retrieval exact-matching scan”으로 범위를 한정. Index maintenance와 candidate-list 처리는 별도 비용임을 명시 |
| §3.3-3.4, pp.3-4 | Context와 hyperparameter가 수작업인데 선택 이유·민감도가 충분히 연결되지 않음 | Task별 context 구성표를 본문에 두고 기본값·선택 근거·민감도 자료를 연결. 원리적으로 최적이라는 표현은 사용하지 않음 |
| §3.4, p.4 | Mean-one 성질과 동작 설명이 길고 반복됨 | Weight 식과 짧은 pseudocode로 압축. 평균 계수 보존과 gradient norm·원래 목적함수 gradient 보존은 다르다는 주의는 유지 |
| §4.1, p.4 | Primary 5-seed, Drawer SAC matched cohort 부재라는 구버전 상태 | 최종 분석의 seed 수와 실제 완료 조건으로 갱신. 구 fixed-alpha SAC와 새 auto-temperature SAC를 분리. 기존 1-5와 새 6-10의 기록 한계도 공개 |
| §4.2 및 Table 1, pp.5-6 | 새 PAL·SAC·seed 확장을 포함하지 않음 | HER/HGR/PER의 전체 성능과 가까운 loss-side 비교를 구별해 제시. PAL을 핵심 비교로 배치하되 DrawerOpen-DDPG 한 조건임을 표시. SAC의 불리한 결과도 포함 |
| §4.3, p.5 | Activation, enrichment, residual reduction이 숫자 중심 산문으로 나열됨 | 목적·측정량·관측·한계를 담은 compact diagnostic table과 한 개 핵심 그림으로 교체. 학습 batch residual과 별도 policy-action 오차를 구별 |
| §4.4 및 Table 2, pp.6-7 | 기존 네 control과 새 배정 비교의 역할이 혼동될 수 있음 | Query identity, query pairing, batch-wide random, context-eligible random, TD Top-k를 통제 요소별로 정리. Count-matched에서도 실행 간 실제 k·buffer는 다름을 명시 |
| §4.5 및 Conclusion, p.7 | 7개 조건·5-seed·Drawer SAC 부재에 맞춘 결론 | 확장 후 확인한 평균·CI·실패를 반영. “모든 task/backbone에서 개선” 또는 “Tail의 필요성 입증”으로 확대하지 않음 |
| Supplement와 code artifact | 새 결과와 원고 사이 claim·표·source 연결이 아직 통합되지 않음 | 실행 집합을 구분한 claim-to-artifact 표, 원고 수치 재생성, 독립 패키지 실행·익명성 점검. 현재 GitHub 아카이브를 제출용 완성 패키지로 간주하지 않음 |

### 4.2 권장하는 Method 전개

1. HER로 바뀐 목표에서 두 replay sample을 비교하는 간단한 예시를 보여준다.
2. “Local context가 충분한가”와 “policy-action bin이 관찰되는가”를 정의한다.
3. Indicator와 normalized critic loss를 제시한다.
4. 왜 이 배정을 시도하는지, 어떤 경험적 가설을 검증할 것인지 설명한다.
5. Coarse retrieval, Tail scan, exact matching, 계산 비용과 누락 가능성을 설명한다.
6. DDPG/TD3/SAC의 공통 적용과 SAC mean-action query 차이를 표로 정리한다.

Index 구현을 숨기자는 뜻이 아니다. **독자가 알고리즘의 목적을 이해한 뒤 정확한 구현을 따라가도록 순서를 바꾸는 것**이다.

### 4.3 기여 문장의 수정 방향

아래는 최종 결과에 맞춰 다듬을 문장 초안이다.

- “HER로 relabeling된 context에서 현재 정책 행동의 관찰된 local replay support를 이용하는 critic-loss allocation 규칙을 제시한다.”
- “Sampling·relabeling·update budget을 유지한 비교와 count-matched controls를 통해 loss 배정 신호의 효과를 평가한다.”
- “조회 방식, stochastic backbone, seed 변동성에 따른 이득과 한계를 함께 분석한다.”

마지막 두 항목은 “모든 비교에서 superiority를 입증했다”는 뜻이 아니다. 차이가 불확실한 조건과 불리한 조건을 함께 보고해야 한다.

## 5. 실험 계보와 공통 해석 규칙

### 5.1 고유 실행 수와 재사용 관계

| 실험군 | 새 본학습 | 재사용 | 현재 상태 |
|---|---:|---|---|
| 배정 비교, Support/Random/TD Top-k | 30 | 없음. 세 방법 모두 새 10-seed 실행 | 완료 |
| PAL | 10 | 위 Support 10회 | 완료 |
| 오프라인 조회·실패 진단 | 0 | Support checkpoint 10개와 배정 로그 30회 | 완료 |
| Online Random-512 | 10 | 위 Support 10회를 Tail 기준으로 재사용 | 완료, 양쪽 최종 checkpoint 진단도 완료 |
| Drawer SAC HER/SABER | 20 | 전원 중단 전 완료된 3회 보존 + 나머지 17회 fresh 실행 | 완료 |
| Drawer SAC HGR/PER | 20 | HER/SABER 20회는 비교용으로만 재사용 | 완료 |
| 원논문 DrawerOpen-DDPG seeds 6-10 확장 | 40 예정 중 10 완료 | 원논문 각 조건 seeds 1-5 | HGR 5회 실행, 이후 25회 대기 |
| 합계 | **100 완료, 5 실행, 25 대기** | 동일 실행을 중복 계수하지 않음 | 총 승인된 신규 본학습 130회 |

배정 실험의 SABER 평균 AUC는 0.652613이고, 원논문 1-5와 새 확장 6-10을 합친 평균은 0.651872다. 서로 가까워도 **다른 실행 집합**이다. 일부 seed의 수치가 같다는 이유만으로 독립성을 추가로 주장하거나 두 집합을 20개의 독립 seed로 합치지 않는다. 논문·분석에서 어느 source와 reference를 사용하는지 고정해야 한다.

이후 논의한 다른 task/backbone의 10-seed 통일은 연구 방향이지 현재 모든 조합이 예약되었다는 의미는 아니다. 실제 예약 목록은 7절에 한정한다.

### 5.2 공통 학습 설정과 지표

후속 Drawer 실험은 실행당 2M step, UTD=1, horizon=200, nominal HER ratio=0.8을 기본으로 한다. DDPG 추가 비교는 DrawerOpen-v3 sparse reward v2, batch 256, replay capacity 300,000, warmup 5,000, 초기 random action 10,000 step, gamma=0.98, tau=0.005를 사용한다. 전체 설정은 각 protocol과 recorded config가 기준이다.

- **AUC:** 10k, 20k, ..., 2M의 정확한 200개 평가 checkpoint에서 trapezoidal integration하고 **1,990,000**으로 나눈 normalized success AUC.
- **FQ-AUC:** 1.5M-2M 구간을 500,000으로 정규화한 값.
- **Final:** 2M checkpoint의 평가 성공률. 학습 episode 성공률이나 마지막 상태 성공률과 혼동하지 않는다.
- **Hit90:** 평가 성공률이 처음 0.9 이상인 checkpoint. 한 번 도달했다고 이후 성능이 유지되었다는 뜻은 아니다. 미도달은 NA로 두고 coverage를 함께 보고한다.
- **Mean +/- SD:** seed별 지표의 평균과 sample standard deviation. 표의 소수 자릿수는 표시용 반올림이다.
- **Paired 비교:** 같은 seed 번호의 방법 차이를 분석한다. 같은 seed라고 online buffer나 trajectory가 동일한 것은 아니다.
- **중간 결과:** 세 방법의 정확히 같은 학습 구간으로만 비교한다. 진행 중인 1.47M AUC를 완료된 2M AUC와 직접 비교하지 않는다.
- **성능 실패:** 유한한 loss로 예산을 완료했지만 성공률이 0인 실행도 결과에 포함한다. 프로세스 오류·non-finite·장비 중단과 구분한다.

### 5.3 통계 규칙

| 분석 | Seed 수 | Bootstrap | 분석 RNG | 구간 |
|---|---:|---:|---:|---|
| Support vs Random / TD Top-k | 10 pairs | 10,000 seed-row resamples | 20260925 | 95%, 두 주요 비교에 97.5% 보정 구간 추가 |
| SABER vs PAL | 10 pairs | 10,000 | 20260930 | 95%, 단일 사전 지정 primary contrast |
| Random-512 vs Tail | 10 pairs | 10,000 | 20261004 | 95% |
| SAC 두 task | 각 5 pairs | 10,000 | 2027 | 95% 및 비교 가족별 97.5%/98.75% |
| 원논문 seed 확장 | 1-5, 6-10은 각 5; pooled는 10 | 10,000 | 2027 | 95%, 실행 시기별 결과도 별도 보고 |

Percentile 구간은 작은 seed 수에서 근사적인 불확실성 요약이지 정확한 coverage 보장이 아니다. 배정 비교는 두 차이에 같은 resampled seed 행을 사용한다. SAC의 HER 대비 두 비교에는 97.5%, HGR/PER 대비 네 비교에는 98.75% 구간을 적용한 기록이 있다. HGR/PER 확장은 HER/SABER 결과를 본 뒤 추가한 탐색적 비교이므로 전체 네 방법을 처음부터 하나의 사전등록 실험이었다고 표현하지 않는다.

**0을 포함하는 CI는 “효과가 없다”, “동등하다”, “아무 의미가 없다”를 뜻하지 않는다.** 양의 평균만으로 우월성을 확정할 수도 없다. 반대로 0을 제외한다고 메커니즘이나 다른 환경으로의 일반화가 자동으로 증명되는 것도 아니다. 모든 결과를 보고하되 결과를 본 뒤 favorable seed·metric·task만 선택하지 않는다.

## 6. 완료 실험의 목적과 결과

### 6.1 Support vs Random vs TD-error Top-k 배정 비교

**왜 했는가.** 기존 strict context-eligible control의 CI가 0을 포함했고, 기존 continuous TD-error weighting과는 weight 형태·개수도 달랐다. 따라서 “같은 context eligibility와 SABER에서 유도한 high-weight 개수를 유지할 때, support-positive 위치에 weight를 배정하는 추가 가치가 있는가”를 더 직접적으로 평가했다. [설계와 결과][ASSIGN]

**통제한 것.** E는 context population 조건을 통과한 minibatch 위치 집합, M은 support-positive 위치, m=|E|, k=|M|이다. Support는 M을, Random은 E에서 k개를 균일 비복원 선택하며, TD Top-k는 E에서 pre-update absolute TD error가 큰 k개를 선택한다. 경계 동률은 무작위 처리한다. 모두 raw 2:1, mean-one normalization, critic-only다.

각 실행은 **자신의** policy·buffer·batch에서 E와 k를 계산한다. 실행 간 실제 E, k, buffer를 같게 고정한 실험이 아니며, 세 방법 모두 eligibility와 k를 통해 SABER 정보를 사용한다. 따라서 support 계산 전체를 제거한 대조군은 아니다.

**최종 성능.** 각 방법 10 seeds, 모두 2M 완료.

| 방법 | AUC mean +/- SD | FQ-AUC mean +/- SD | Final mean +/- SD | Hit90 도달 | 도달 seed 중앙값 |
| --- | --- | --- | --- | --- | --- |
| SABER | 0.6526 +/- 0.2484 | 0.8929 +/- 0.3138 | 0.890 +/- 0.313 | 9/10 | 470k |
| Random assignment | 0.5157 +/- 0.3594 | 0.6977 +/- 0.4815 | 0.700 +/- 0.483 | 7/10 | 550k |
| TD-error Top-k | 0.5346 +/- 0.3802 | 0.6936 +/- 0.4787 | 0.700 +/- 0.483 | 7/10 | 400k |

| 비교 | 평균 차이 | 95% CI | 97.5% CI | 양수 / 음수 / 동률 |
| --- | --- | --- | --- | --- |
| SABER - Random | +0.136935 | [-0.127451, +0.384334] | [-0.162888, +0.414626] | 7 / 3 / 0 |
| SABER - TD Top-k | +0.117990 | [-0.054866, +0.324917] | [-0.071018, +0.360704] | 4 / 5 / 1 |

**해석.** Support의 평균 AUC와 Hit90 coverage가 더 높다. 다만 주요 CI가 모두 0을 포함하고, TD Top-k 대비는 4개 seed에서 개선, 5개에서 악화, 1개에서 동률이다. 평균의 양의 방향은 배정 규칙의 추가 가치와 양립하지만, exact occupancy가 반드시 필요하거나 유일한 원인이라고 결론 내리기에는 부족하다.

**배정 진단.** 각 실행 안에서는 유효 update 수로 집계한 뒤 seed를 동일한 단위로 평균했다.

| 방법 | Eligible 비율 | Support-positive 비율 | 포화 update 비율 | Informative update 비율 |
| --- | --- | --- | --- | --- |
| SABER | 96.616% | 42.937% | 2.799% | 97.201% |
| Random assignment | 96.930% | 50.465% | 11.376% | 88.624% |
| TD-error Top-k | 96.736% | 50.808% | 13.422% | 86.578% |

Support arm의 informative update, 즉 0<k<m인 update에서 Support와 TD Top-k의 overlap/k는 **0.504095**, Random의 관측 overlap/k는 **0.427443**, 무작위 기준 기대값 k/m은 **0.427438**이었다. 두 선택 신호가 같지는 않으며, Random 선택은 기대 겹침과 일치한다. 이 진단 자체가 어느 신호의 성능 우위를 증명하지는 않는다.

k=0이면 모두 uniform weight다. k=m>0이면 세 방식 모두 E 전체를 고르므로 서로 같은 weight가 되지만, **E가 minibatch 전체일 때만 전체 uniform weight**다. k=0일 때 overlap/k는 NA다. Jaccard의 기대값을 기대 교집합 k²/m을 단순 대입한 값으로 취급하지 않는다.

**본문 반영.** 기존 §4.4를 query identity와 assignment 분리 질문으로 재구성하고, 새 세 방법의 성능·CI를 본문 compact table에 넣는다. 기존 네 control을 대체해 지우지 말고 각 control이 보존·제거한 요소를 구분한다. 남은 질문은 다른 조건에서의 재현성과 별도 policy-action 평가다.

### 6.2 HER+PAL 비교

**왜 했는가.** 일반적인 critic-loss reweighting의 신규성 우려와 PAL 비교 누락에 답하기 위해, uniform replay에서 loss를 바꾸는 구체적 선행방법과 비교했다. 이는 선택 신호를 고립시키는 6.1의 mechanism control과 역할이 다르다. [실행 설계][PAL-PROTOCOL], [결과][PAL]

PAL은 author implementation의 alpha=0.4, threshold/min_priority=1, detached minibatch normalizer를 사용했다. 실행 protocol의 piecewise loss, 0.5 배율, outer branch와 normalization을 유지했고 SABER의 2:1 weighted MSE로 대체하지 않았다. Author TD3 loss를 **기존 single-critic DDPG-HER**에 적용한 비교이며, 원 PAL 논문의 TD3 benchmark 전체 재현으로 표현하지 않는다. Uniform replay, HER, actor/target update와 환경 예산은 reference와 맞추고 PAL에는 support lookup을 수행하지 않았다.

PAL 10회가 새 실행이며, 비교 SABER 10회는 6.1의 Support arm이다.

| 방법 | AUC mean +/- SD | FQ-AUC mean +/- SD | Final mean +/- SD | Hit90 도달 | 도달 seed 중앙값 |
| --- | --- | --- | --- | --- | --- |
| SABER | 0.6526 +/- 0.2484 | 0.8929 +/- 0.3138 | 0.890 +/- 0.313 | 9/10 | 470k |
| HER+PAL | 0.5454 +/- 0.3028 | 0.7926 +/- 0.4179 | 0.800 +/- 0.422 | 8/10 | 620k |

- SABER−PAL AUC: **+0.107261**, 95% CI **[−0.153141, +0.357651]**.
- Seed별 방향: 양수 7, 음수 2, 동률 1.
- PAL 실행 설정은 PAL 학습 전에 고정했지만, 이 비교를 계획한 시점은 배정 실험의 중간 결과를 확인한 이후다.

**해석.** 이 조건에서 SABER의 평균은 높지만 PAL 대비 우위를 확정하기에는 구간이 넓다. 중요한 진전은 가까운 선행방법을 회피하지 않고 비교했다는 점이다. 성능 비교만으로 support rule의 신규성·필요성·일반성까지 확보한 것은 아니다. PAL과 SABER의 loss scale 차이는 해당 방법의 구현 일부로 보존했으며, 순수한 선택 신호 비교는 6.1이 담당한다.

기록된 평균 runtime은 PAL 약 19.78시간, SABER reference 약 25.79시간/seed다. 후자는 추가 mask 진단 등을 포함하므로 이 값만으로 알고리즘 자체의 공정한 시간 효율 순위를 정하지 않는다.

**본문 반영.** §4.2에 가까운 loss-side baseline 비교로 배치한다. Related Work의 Fujimoto 2020 설명에서 PAL 실험으로 바로 연결한다. 현재 근거 범위는 DrawerOpen-DDPG이고, 다른 task/backbone으로의 비교는 미실행이다.

### 6.3 Tail / Random / Full 오프라인 조회 진단

**왜 했는가.** “왜 Tail인가”, “512개 밖에 있는 support를 얼마나 놓치는가”에 답하기 위해 학습을 추가하지 않고 고정 checkpoint에서 조회 규칙만 바꿨다. [진단 결과][OFFLINE]

초기 분석은 SABER 최종 checkpoint 10개에서 seed당 1,024개의 고정 HER query를 사용하고 random retrieval을 5회 반복했다. 동일 query의 policy·replay·index·full-context 정의를 유지했다. Full은 해당 이산 context의 전체 후보 기준이며, 연속 공간의 참 support 또는 Q 정확도 정답은 아니다.

| 조회 방식 | Support+ | Eligible | Full 대비 mask 불일치 | Full-positive 중 누락 | CPU ms/query |
| --- | --- | --- | --- | --- | --- |
| tail | 53.711% | 97.881% | 36.563% | 41.046% | 3.952 |
| random | 62.535% | 99.730% | 27.738% | 31.173% | 3.991 |
| full | 90.273% | 99.980% | 0.000% | 0.000% | 6.322 |

위 값은 seed별 비율의 평균이다. Full-positive 중 누락률은 Full이 positive인 query만 분모로 쓰며, 전체 query의 mask 불일치율과 다르다. CPU 시간은 공통 참조 구현의 ms/query이지 production GPU/Numba 전체 비용이 아니다.

**해석.** 이 최종 checkpoint 집합에서는 Random-512가 Tail-512보다 Full-positive support를 덜 놓쳤다. 따라서 Tail의 negative를 “전체 replay에 support가 없음”으로 해석할 수 없다. 그러나 Full과의 agreement가 높다는 사실만으로 online 학습이 더 좋아진다는 결론은 나오지 않는다. 그 질문을 평가한 실험이 6.5다.

초기 사후 분석은 config의 thread_hint metadata를 학습 parameter 오류로 오인해 중단된 이력이 있다. 이는 checkpoint loader 단계의 오류였고 PAL 학습 결과의 오류가 아니었다. 독립 복구판에서 metadata 처리와 검증을 보강해 다시 실행했고, 학습을 재개하거나 성능 결과를 교체하지 않았다. [복구 기록][RECOVERY]

**본문 반영.** §3.3의 bounded scan 정의와 §4의 조회 분석에 숫자를 연결한다. “조회 범위에서 support가 확인되지 않음”과 “전체 replay에 support가 없음”을 구별한다.

### 6.4 실패와 support 포화 분석

**왜 했는가.** 평균만으로 보이지 않는 실패·변동성을 분석하고, support 신호가 유용한 선택을 하지 못하는 상태와 어떤 관계가 있는지 살폈다.

대상은 6.1의 **30회 전체**다. Zero-evaluation은 200개 평가 checkpoint 전부에서 성공률이 0인 경우다. 마지막 성공률만 0인 collapse와는 다르다. 포화는 **k=m>0**, 즉 모든 eligible 위치가 support-positive여서 배정 규칙 간 구별이 사라지는 update를 뜻한다.

| 방법 | Zero-evaluation seeds | 개수 | Zero 그룹 Support+ | 나머지 Support+ | Zero 그룹 포화 | 나머지 포화 |
| --- | --- | --- | --- | --- | --- | --- |
| SABER | 7 | 1/10 | 55.957% | 41.490% | 27.987% | 0.000% |
| Random assignment | 3, 8, 9 | 3/10 | 64.385% | 44.500% | 37.920% | 0.000% |
| TD-error Top-k | 6, 7, 8 | 3/10 | 70.294% | 42.457% | 44.740% | 0.000% |

Zero-evaluation 7회 모두에서 적어도 한 번 포화가 관측되었고, 나머지 23회에서는 포화가 관측되지 않았다. 처음 500k까지의 포화 비율은 이 zero 그룹들에서도 0이었다. 따라서 “초기 포화 때문에 처음부터 실패했다”는 설명으로 바로 연결하면 안 된다. 포화는 실패가 지속되며 replay가 좁아지는 결과일 수도 있고, 공통 원인의 결과일 수도 있다.

최초 포화를 포함한 기록 window의 끝 step은 SABER seed 7에서 1,104,999, Random seeds 3/8/9에서 각각 693,999 / 586,999 / 537,999, TD Top-k seeds 6/7/8에서 1,049,999 / 706,999 / 965,999였다. 이는 최초 사건의 정확한 단일 update가 아니라 집계 window 경계다.

**해석.** 실패 조건에서 신호가 포화될 수 있다는 관찰은 적용 한계와 후속 가설을 구체화한다. 하지만 성능으로 나눈 사후 그룹이므로 인과관계를 확정하거나, 그 7개 seed를 제외해 평균을 다시 주장하는 데 사용할 수 없다.

**본문 반영.** Failure/limitations 절에 전체 seed 곡선과 시간대별 activation을 함께 보여준다. 원인을 주장하려면 사전에 정한 동일 조건에서 별도 intervention을 검증해야 한다.

### 6.5 Tail-512 vs Random-512 online 학습

**왜 했는가.** 6.3에서 조회 mask가 다르다는 것을 확인했으므로, 후보 선택 방식의 차이가 실제 학습에도 영향을 주는지 평가했다. [성능·최종 조회 분석][ONLINE]

이 Random은 6.1의 **minibatch high-weight 위치 선택**이 아니라, **support를 계산하기 위해 읽을 replay 후보 선택**이다. 같은 coarse bucket의 유효 후보에서 최대 512개를 균일 비복원 선택하고 이후 exact context·action-bin 조건은 유지한다. Dedicated RNG를 사용하고, 같은 query batch의 bucket 단위 subset 공유를 protocol에 명시했다.

Random-512 10회를 새로 실행했고 Tail은 기존 Support 10회를 재사용했다.

| 방법 | AUC mean +/- SD | FQ-AUC mean +/- SD | Final mean +/- SD | Hit90 도달 | 도달 seed 중앙값 |
| --- | --- | --- | --- | --- | --- |
| SABER Tail-512 | 0.6526 +/- 0.2484 | 0.8929 +/- 0.3138 | 0.890 +/- 0.313 | 9/10 | 470k |
| SABER Random-512 | 0.4302 +/- 0.3772 | 0.5920 +/- 0.5098 | 0.590 +/- 0.509 | 6/10 | 550k |

원 분석 방향인 Random-512−Tail AUC는 **−0.222387**, 95% CI **[−0.518372, +0.080932]**다. 방향을 뒤집으면 Tail−Random은 **+0.222387 [−0.080932, +0.518372]**다. Tail이 높은 pair는 6/10, Random이 높은 pair는 4/10이다. 평가 성공이 한 번도 없었던 실행은 Tail 1/10, Random-512 4/10이다.

**해석.** 관측 평균은 Tail에 유리하지만 이 한 조건에서 Tail의 우위가 확정되었다고 하기는 어렵다. Random이 Full의 support를 더 잘 찾는 정적 결과와 Tail의 높은 online 평균은 모순이 아니다. 두 지표는 다른 질문이며 online 학습에서는 buffer·policy가 함께 바뀐다. 이를 “support를 덜 찾는 것이 좋다”는 인과적 설명으로 바꾸어서는 안 된다.

**Online 완료 후의 추가 정적 진단.** 양쪽 최종 checkpoint 20개를 대상으로 seed당 1,024 query, random 5회 반복 분석도 완료했다. 아래는 각 학습 arm 내 10개 seed의 평균이다.

| 학습 arm | 진단 조회 방식 | Support+ | Eligible | Full-positive 중 누락 |
| --- | --- | --- | --- | --- |
| tail | tail | 52.998% | 97.344% | 41.869% |
| tail | random | 62.316% | 99.617% | 31.484% |
| tail | full | 90.303% | 99.971% | 0.000% |
| random512 | tail | 71.416% | 98.623% | 25.353% |
| random512 | random | 77.018% | 99.736% | 19.221% |
| random512 | full | 94.316% | 100.000% | 0.000% |

Tail mask parity mismatch는 0, direct full-replay spot check는 320건이다. 초기 진단과 후속 진단은 query RNG와 random subset 공유 방식이 다르므로 수치를 섞지 않는다. 초기 query seed는 2026100200+s, 후속 query seed는 202610040+s다. 어느 쪽도 학습 초기·중기의 모든 정책과 replay를 복원한 분석은 아니다.

**본문 반영.** Lookup 구현의 선택 근거에 대한 실증 자료로 사용하되 “Tail이 최적/필수”라고 쓰지 않는다. 범위는 DrawerOpen-DDPG의 현재 scanner와 context다.

### 6.6 DrawerOpen / DrawerClose SAC 평가

**왜 했는가.** 기존 Table 1에는 Drawer SAC matched comparison이 없었다. FPP에서 사용한 SAC를 공통 기준으로 삼아 stochastic backbone까지 평가 범위를 채웠다. 이는 직접적인 리뷰 요청 한 항목을 넘어 기존 원고가 밝힌 범위 제한을 보완하는 실험이다.

새 실행은 **auto-temperature SAC**, stochastic Gaussian actor, twin critics, initial alpha=0.2, target entropy=-4를 사용한다. SABER는 squashed policy mean으로 support를 조회하고 두 critic loss에 같은 weight를 적용한다. 과거 fixed-alpha DrawerOpen prototype과 합치지 않는다.

#### DrawerOpen SAC

| 방법 | AUC mean +/- SD | FQ-AUC mean +/- SD | Final mean +/- SD | Hit90 도달 |
| --- | --- | --- | --- | --- |
| HER | 0.4470 +/- 0.4108 | 0.6008 +/- 0.5425 | 0.600 +/- 0.548 | 3/5 |
| SABER | 0.4485 +/- 0.4108 | 0.5992 +/- 0.5465 | 0.600 +/- 0.548 | 3/5 |
| HGR | 0.7066 +/- 0.0659 | 0.9952 +/- 0.0065 | 1.000 +/- 0.000 | 5/5 |
| PER-full | 0.7976 +/- 0.0722 | 0.9840 +/- 0.0149 | 1.000 +/- 0.000 | 5/5 |

#### DrawerClose SAC

| 방법 | AUC mean +/- SD | FQ-AUC mean +/- SD | Final mean +/- SD | Hit90 도달 |
| --- | --- | --- | --- | --- |
| HER | 0.2606 +/- 0.2846 | 0.2475 +/- 0.3959 | 0.200 +/- 0.447 | 5/5 |
| SABER | 0.1033 +/- 0.0565 | 0.0176 +/- 0.0098 | 0.000 +/- 0.000 | 5/5 |
| HGR | 0.2118 +/- 0.1314 | 0.0619 +/- 0.1237 | 0.030 +/- 0.067 | 5/5 |
| PER-full | 0.4873 +/- 0.3043 | 0.5909 +/- 0.4985 | 0.550 +/- 0.505 | 5/5 |

각 task에서 HER/SABER/HGR/PER 모두 seeds 1-5, 각 2M step이다. HER/SABER 20회와 HGR/PER 20회, 총 **40개의 고유 완료 실행**이다.

#### Paired AUC 비교

| Task | 비교 | 평균 차이 | 95% CI | 가족별 보정 CI | 보정 수준 | 양수 pair |
| --- | --- | --- | --- | --- | --- | --- |
| DrawerOpen | saber-her | +0.001457 | [-0.046633, +0.041283] | [-0.056683, +0.042412] | 97.5% | 2/5 |
| DrawerOpen | saber-hgr | -0.258090 | [-0.596533, +0.065075] | [-0.715980, +0.080302] | 98.75% | 3/5 |
| DrawerOpen | saber-per | -0.349095 | [-0.674724, -0.023467] | [-0.803266, -0.000101] | 98.75% | 1/5 |
| DrawerClose | saber-her | -0.157312 | [-0.419070, +0.025025] | [-0.440276, +0.038166] | 97.5% | 1/5 |
| DrawerClose | saber-hgr | -0.108467 | [-0.234950, +0.004271] | [-0.259849, +0.031307] | 98.75% | 1/5 |
| DrawerClose | saber-per | -0.383995 | [-0.623945, -0.144045] | [-0.660126, -0.070251] | 98.75% | 0/5 |

Adjusted 구간의 coverage 역시 작은 표본의 근사 bootstrap에 의존한다. 특히 DrawerOpen SABER−PER의 98.75% 상단은 −0.000101 수준으로 0에 매우 가깝다. “여러 설정에서 확실히 열등하다”는 식으로 범위를 확대하지 않는다.

**해석.**

- DrawerOpen에서 HER와 SABER의 평균은 매우 가깝고, HER 대비 추가 이득을 뚜렷하게 확인하지 못했다. 그렇다고 동등성이 입증된 것은 아니다.
- DrawerClose에서는 SABER 평균 AUC가 세 비교방법보다 낮고, 모든 seed의 final success가 0이다. Hit90은 전부 도달했지만 FQ-AUC가 0.018 수준이므로 “한 번 성공했다”와 “학습을 유지했다”를 구분해야 한다.
- 두 task 모두 SABER−PER의 95% CI가 음수 구간에 있다. 불리한 결과도 본문 요약에서 빠뜨리지 않는다.
- 이는 “SAC에서는 언제나 SABER가 실패한다”는 결론이 아니다. 기존 FPP-SAC 결과와 함께 보면 **task/backbone 조합에 따른 차이**를 설명할 필요가 생긴 것이다.

**정합성·원인 분석에서 남은 점.** Inherited MetaWorld observation의 마지막 세 성분에는 original goal이 남아 있고, HER는 별도로 붙이는 goal을 바꾼다. 비교한 네 방법 모두 같은 표현을 쓰지만 그 영향은 미확정이다. 이 점만으로 SAC 결과가 무효라고 하거나 반대로 전혀 영향이 없다고 단정하지 않는다. 수정이 필요하면 모든 비교방법에 같은 변경을 적용한 별도 matched experiment로 검증하고 기존 실행·불리한 결과는 보존한다.

Stochastic actor의 mean-action support와 실제 sampled action 간 차이, entropy/alpha 변화, critic 값·target 변화, goal 표현, phase coverage는 후속 분석 후보이지 현재 밝혀진 원인이 아니다.

**중단·재실행 이력.** 전원 중단 전 완료된 DrawerOpen HER seeds 1-3은 보존했다. 중단된 다섯 attempt는 step 0부터 다시 실행했고, recovery의 17 fresh runs와 보존된 3회가 두 task의 20회 HER/SABER 집합을 이룬다. Checkpoint continuation이나 잘린 curve 이어붙이기는 하지 않았다. HGR/PER는 별도로 20회 완료했다. [Open 결과][SAC-OPEN], [Close 결과][SAC-CLOSE]

**본문 반영.** §4.1의 “matched Drawer SAC가 없다”는 문장을 고치고 Table 1을 채운다. Conclusion의 범위를 task/backbone 의존 결과에 맞춘다. 실패 분석의 요약을 appendix에만 숨기지 않는다.

### 6.7 원논문 DrawerOpen-DDPG seeds 6-10 확장

**왜 했는가.** AAAI의 primary·mechanism 결과를 5개 seed에서 10개로 늘려 효과 추정과 seed 변동성을 더 잘 파악하기 위한 작업이다. 기존 결과를 유지하고 seeds 6-10만 추가한다. 6.1의 새 배정 실험과는 별도다. [완료된 HER/SABER artifact][EXT]

HER와 SABER는 각 5회 추가 실행을 완료했다. 원논문 seed 1-5와 새 6-10을 나누어 보고한 뒤 pooled 1-10도 제시한다.

| 실행 집합 | Paired n | HER AUC | SABER AUC | 차이 | 95% CI | 양수 pair |
| --- | --- | --- | --- | --- | --- | --- |
| historical_1_5 | 5 | 0.2073 +/- 0.2868 | 0.7537 +/- 0.0444 | +0.546357 | [+0.339874, +0.741005] | 5/5 |
| extension_6_10 | 5 | 0.5948 +/- 0.3343 | 0.5501 +/- 0.3289 | -0.044749 | [-0.478116, +0.456080] | 1/5 |
| pooled_1_10 | 10 | 0.4011 +/- 0.3577 | 0.6519 +/- 0.2459 | +0.250804 | [-0.067643, +0.543933] | 6/10 |

보조 지표는 다음과 같다. 이 구간은 primary AUC와 별도의 보조 분석이며 동시 유의성을 보장하지 않는다.

| 실행 집합 | 지표 | HER 평균 | SABER 평균 | 차이 | 95% CI |
| --- | --- | --- | --- | --- | --- |
| historical_1_5 | final_quarter_auc | 0.3927 | 0.9874 | +0.594700 | [+0.205300, +0.982800] |
| historical_1_5 | final_success | 0.3900 | 1.0000 | +0.610000 | [+0.210000, +1.000000] |
| extension_6_10 | final_quarter_auc | 0.7895 | 0.7929 | +0.003400 | [-0.577000, +0.596200] |
| extension_6_10 | final_success | 0.8000 | 0.7900 | -0.010000 | [-0.600000, +0.590000] |
| pooled_1_10 | final_quarter_auc | 0.5911 | 0.8901 | +0.299050 | [-0.092651, +0.689105] |
| pooled_1_10 | final_success | 0.5950 | 0.8950 | +0.300000 | [-0.105000, +0.700000] |

**해석.** Pooled AUC 차이는 양수이지만 기존 5-seed 결과보다 작아졌고 CI가 0을 포함한다. 새 6-10만 보면 HER가 더 높은 평균이며 SABER가 높은 pair는 1/5다. 추가 seed가 기존 효과를 더 선명하게 재현했다고 해석하면 안 된다. 반대로 이 결과만으로 효과가 없다고 결론낼 수도 없다. 실험을 확장하며 작은 표본에서 드러나지 않았던 변동성이 나타난 상황으로 보는 것이 적절하다.

HER seed 10과 SABER seed 7은 2M을 정상 완료했지만 평가 성공이 0이었다. 이들은 제외하지 않았다.

**과거 기록의 한계.** 원논문 HER/SABER의 일부 원 실행 config와 환경 lock은 남아 있지 않다. 새 실행은 보존된 canonical source·launcher 계약을 복원해 사용했으며, 과거 실행과 bit-for-bit 동일한 환경임을 입증한 것은 아니다. 현재 설정·source hash는 기록되어 있다. 따라서 1-5/6-10의 차이를 전적으로 seed만의 원인으로 단정하지 않고 실행 시기별 결과를 계속 병기한다.

**본문 반영.** Pooled 결과로 대표 수치·표·결론을 갱신하되 historical 결과도 appendix에 남긴다. 현재 HGR/PER와 네 mechanism control 확장은 미완료이므로 그 비교까지 10-seed로 완료된 것처럼 표시하지 않는다.

### 6.8 비용 기록과 기존 supplementary 자료의 위치

| 실행 집합 | 방법 | 기록된 steps/s mean +/- SD |
| --- | --- | --- |
| DrawerOpen DDPG 배정 | SABER | 21.60 +/- 1.24 |
| DrawerOpen DDPG 배정 | Random assignment | 21.78 +/- 1.44 |
| DrawerOpen DDPG 배정 | TD-error Top-k | 21.48 +/- 1.41 |
| DrawerOpen DDPG PAL | HER+PAL | 28.10 +/- 0.32 |
| DrawerOpen DDPG 조회 | Random-512 | 9.88 +/- 0.24 |
| DrawerOpen SAC | HER | 37.92 +/- 4.01 |
| DrawerOpen SAC | SABER | 20.16 +/- 2.83 |
| DrawerOpen SAC | HGR | 11.78 +/- 0.36 |
| DrawerOpen SAC | PER-full | 39.48 +/- 3.24 |
| DrawerClose SAC | HER | 37.75 +/- 4.71 |
| DrawerClose SAC | SABER | 21.19 +/- 1.64 |
| DrawerClose SAC | HGR | 11.79 +/- 0.39 |
| DrawerClose SAC | PER-full | 39.37 +/- 3.20 |

이 표는 실제 기록된 실행 속도이며, 서로 다른 시점·진단량·동시 부하를 통제한 효율 순위표가 아니다. 특히 배정 실험의 Random/TD arm도 counterfactual support·진단 mask를 계산한다. Support 계산이 전혀 없는 baseline의 비용을 이 표에서 추정하면 안 된다.

AAAI supplement에는 이미 FPP-TD3 seed 1, 20k-100k 구간의 동일 하드웨어 profile이 있었다. HER 8.063 ms/step, SABER 24.133 ms/step, support/weighting 15.028 ms/step이었다. 이 한 profile은 후속 신규 실험이 아니며, 다른 task나 현재 구현 전체의 배율로 일반화하지 않는다. Reviewer가 장점으로 언급한 “low complexity overhead”를 실제 전 구간 wall-clock 비용에 대한 확인으로 받아들이면 안 된다.

기존 K=3/5/7, actor/critic scope, continuous TD-error weighting, AER, REDQ, high-RR, FPP-TD3 확장·inclusive-future, fixed-target diagnostics는 AAAI 자료에 이미 포함된 근거다. 이번 신규 100회에 다시 세지 않는다. 특히 K=3/5/7의 FPP-TD3 AUC는 0.353/0.442/0.361이고 Support+는 0.396/0.219/0.148이었다. 이는 “인접 설정에도 둔감하다”는 근거가 아니라 해상도·활성화 비율의 역할을 더 살펴볼 이유다. [기존 supplementary source][SUPP]

## 7. 현재 실행 중인 실험과 예약

### 7.1 실제 실행 큐

기준은 **2026-10-27 14:30:13 KST**의 queue snapshot이다. 이 보고서는 실시간 dashboard가 아니다.

현재 큐는 원논문 DrawerOpen-DDPG의 아래 여덟 조건만 수행한다. 각 조건에서 seeds 6-10을 **5개 병렬** 실행하고, 전 seed 완료·검증 후 다음 조건으로 넘어간다.

| 순서 | 조건 | Seeds | 예산 | 상태 |
| --- | --- | --- | --- | --- |
| 1 | HER | 6-10 | 5 x 2M | 완료 |
| 2 | SABER critic-only | 6-10 | 5 x 2M | 완료 |
| 3 | Full HGR pair-priority + IS | 6-10 | 5 x 2M | 실행 중 |
| 4 | PER-full | 6-10 | 5 x 2M | 대기 |
| 5 | Replay-Action Query | 6-10 | 5 x 2M | 대기 |
| 6 | Shuffled-Policy Query | 6-10 | 5 x 2M | 대기 |
| 7 | Random-Support, batch-wide count-matched | 6-10 | 5 x 2M | 대기 |
| 8 | Context-Eligible Count-Matched | 6-10 | 5 x 2M | 대기 |

- 이 큐의 전체 예산: 40회, 80M step.
- 완료 10회, 실행 중 5회, 대기 25회.
- 기준 시각 누적: **27,366,000 / 80,000,000 step = 34.21%**.
- 현재 조건: HGR. 이후 PER, Replay-Action Query, Shuffled-Policy Query, batch-wide Random-Support, Context-Eligible Count-Matched 순서다.
- 마지막 strict random은 기존 Table 2의 context-eligible control이며, 새 TD-error Top-k를 다시 실행하는 큐가 아니다.

### 7.2 진행 중인 HGR 5개 seed

| Seed | 학습 step | 진행률 | 평가 step | 최근 성공률 | 최고 성공률 | 최근 200k steps/s | 잔여 시간 h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 1,470,400 | 73.52% | 1,470,000 | 1.00 | 1.00 | 13.04 | 11.28 |
| 7 | 1,475,600 | 73.78% | 1,470,000 | 1.00 | 1.00 | 13.14 | 11.09 |
| 8 | 1,471,600 | 73.58% | 1,470,000 | 1.00 | 1.00 | 13.09 | 11.21 |
| 9 | 1,472,800 | 73.64% | 1,470,000 | 0.00 | 0.00 | 13.10 | 11.18 |
| 10 | 1,475,600 | 73.78% | 1,470,000 | 0.00 | 0.10 | 13.12 | 11.10 |

최근 속도가 유지된다는 단순 추정으로 HGR 조건 종료까지 약 11.3시간, **10월 28일 01:47 KST 전후**다. 이는 snapshot 시각에 대한 조건부 추정이며, 장비·부하·평가 시간에 따라 달라질 수 있다.

이 시점 HER/SABER/HGR의 seeds 6-10을 모두 **10k-1,470k**의 같은 평가 grid로 잘라 계산한 중간 지표는 다음과 같다. AUC 분모는 1,460,000이다.

| 방법 | 공통 구간 AUC | 공통 마지막 평가 성공률 |
| --- | --- | --- |
| HER | 0.524041 | 0.800 |
| SABER | 0.462055 | 0.800 |
| HGR | 0.388630 | 0.600 |

HGR의 2M 최종 성능·paired CI는 아직 없다. 위 표를 기존 2M 성능표나 seed 1-10 완료 비교에 섞지 않는다. Seed 9는 지금까지 평가 성공이 없고 seed 10은 최고 0.10이었지만 최근 0이다. 기록된 핵심 loss·TD error·Q/target 값에서 non-finite는 발견되지 않았으며, 현재까지의 평가 grid도 누락이 없었다.

### 7.3 종료 후 자동으로 수행되는 것

모든 여덟 조건이 완료되면 큐의 analysis.py가 기존 1-5, 새 6-10, pooled 1-10의 지표·paired bootstrap 결과를 생성한다. 실패·non-finite·config 불일치·checkpoint grid 누락은 후속 조건 진입을 막으며, 성능이 낮다는 이유만으로 실패 처리하거나 재시작하지 않는다.

전원 재부팅 후 자동 이어달리기는 없다. 중단 시 확인과 명시적인 재실행 판단이 필요하다. 현재 큐 뒤에 PAL 추가 실행, 다른 task의 seed 확장, sensitivity, critic 분기, SAC 추가 seeds가 연결되어 있지는 않다.

## 8. 근본적인 대응을 위해 아직 해야 할 일

다음은 연구·원고 수정 과제다. **실행 중 또는 예약된 실험 목록이 아니다.** 유리한 결과를 보장하는 작업이 아니라 현재 열린 질문을 더 직접적으로 평가하는 방법이다.

| 미해결 질문 | 필요한 작업 | 어떤 근거가 생기면 지적에 더 직접 답할 수 있는가 | 결과가 불확실하거나 불리할 때 |
|---|---|---|---|
| 왜 이 support 기준을 쓰는가 | 같은 checkpoint에서 critic·optimizer를 복제하고 actor·target·학습 batch를 고정한 Support/Random/TD/가능하면 uniform 분기 진단 | 별도 policy-action 데이터에서 예측 개선을 직접 비교. Return을 쓸 경우 horizon·종료·bootstrap 정의를 맞춤 | Replay-action residual 재배분까지로 주장 범위를 제한. “정책이 사용하는 Q가 더 정확해졌다”는 주장을 하지 않음 |
| 현재 action과 replay action 사이의 간극 | 실제 policy action으로 얻은 전이 또는 고정 정책 rollout으로 held-out 평가 구성 | Replay action의 reward/next state를 다른 action의 정답으로 사용하지 않는 평가 | Fixed-target Bellman residual이면 간접 지표임을 표시. True Q error와 구분 |
| Binary gate·가중치·context 선택에 얼마나 의존하는가 | N_local, N_scan, lambda, context 구간의 사전 지정 민감도 | 성능과 activation·포화·조회 비용을 함께 측정. Reward·success threshold는 고정 | Robust하다는 표현을 낮추고 적용 가능한 범위·tuning 부담 공개 |
| 수작업 context가 새 task에도 통하는가 | 단순 공통 context 규칙과 feature mapping을 고정한 전이 | 새 task별 threshold 재튜닝 없이도 유용한지, mapping 작업량이 얼마나 드는지 | 같은 조작 계열의 제한적 적용으로 한정. 이미지·다른 action 차원까지 확장하지 않음 |
| PAL·mechanism 결과가 DrawerOpen-DDPG 밖에서도 의미 있는가 | 결과를 보기 전에 정한 추가 task/backbone에서 검증 | Favorable task 탐색이 아닌 사전 지정된 외적 타당성 자료 | 범위를 해당 조건에 한정. 추가 비교 자체를 novelty 증명으로 쓰지 않음 |
| SAC에서 왜 이득이 없거나 붕괴하는가 | Goal 표현 점검, entropy·alpha·mean-vs-sampled support·critic/phase 시계열 분석; 필요 시 matched 수정 실험 | 원인 후보가 실제 intervention과 연결되는지 | 관찰적 실패 분석으로 보고. SAC 결과를 제외하지 않음 |
| Seed 수로 인한 불확실성을 어떻게 줄일 것인가 | 기존 완료분을 확인한 뒤 합의한 seed 6-10 확장을 고정 예산으로 진행 | Effect size·분포·CI·pair 방향을 함께 보고하고 실행 집합을 추적 | 10-seed라도 불확실성을 인정. CI가 유리해질 때까지 추가하지 않음 |
| 같은 UTD의 이득이 실제 시간에도 남는가 | 같은 host·부하·계측 조건에서 HER/SABER/PAL 성능-시간 평가 | 전체 곡선, time-to-threshold와 미도달 실행, 비용을 함께 보고 | Interaction budget에서의 결과로 한정. 시간 효율 우위를 주장하지 않음 |
| 설명이 어렵다는 공통 지적 | 4절의 Method·figure·table 재구성, 독립 연구자 읽기 검토 | 처음 읽는 독자가 query action, regression action, 변경 objective, 근거 범위를 재구성 가능 | 추가 수식보다 예시·순서·기호 축소를 우선 |

일반적인 수렴정리를 반드시 새로 붙여야 한다는 결론은 아니다. 가정이 명확한 분석이나 설계된 실험으로 주장을 뒷받침하는 것이 중요하다. Mean-one, bounded positive weights 같은 성질은 정확히 설명하되 그것을 performance guarantee로 격상하지 않는다. ICML의 확인 가능한 2026 리뷰 지침도 이론 또는 실험에 의한 soundness와 명료한 originality 설명을 구분해 평가한다. 2027의 세부 규정은 별도로 확인해야 한다. [ICML 리뷰 지침][ICML-REVIEW]

LAP·LaBER 직접 성능 비교, 새로운 simulator benchmark, 새로운 support estimator 설계는 논의할 수 있지만 현재 필수·예약 목록으로 확정된 것은 아니다. 방법 자체를 바꾸면 기존 SABER와 별도 버전의 연구 질문·실행 집합이 되므로 결과를 섞지 않는다.

## 9. 결과를 본문에서 어떻게 해석할 것인가

### 9.1 현재 근거와 맞는 표현

| 사용할 수 있는 표현 | 피해야 할 표현 |
|---|---|
| 추가 DrawerOpen-DDPG 비교에서 SABER가 더 높은 평균 AUC를 보였으나 paired 구간은 넓다 | PAL·Random·TD Top-k를 모두 확실하게 능가했다 |
| Count-matched 비교는 weight 형태와 배정 신호를 더 분리해 평가한다 | Support의 정확한 occupancy만이 성능 개선의 직접적·유일한 원인이다 |
| Tail과 Random은 조회 누락률과 online 성능에서 다른 양상을 보였다 | Tail은 최적이고 Random은 항상 나쁘다 |
| 전체 이산 context 기준의 positive 일부를 제한된 scan이 놓쳤다 | psi=0이면 replay 전체에서 지원이 없다 |
| 평가 실패 실행에서 support 포화가 함께 관측됐다 | 포화가 학습 실패를 일으켰음을 증명했다 |
| Drawer SAC에서 성능 이득 부재와 collapse가 관찰되어 적용 범위를 점검해야 한다 | SAC는 원래 안 되는 알고리즘이므로 해당 결과를 제외한다 |
| Pooled 10-seed에서 평균 이득은 남지만 기존 5-seed보다 불확실성이 드러났다 | Seed를 늘리면 우위가 자동으로 증명된다 |
| Mean-one normalization은 평균 loss coefficient를 보존한다 | 원래 uniform objective의 gradient, gradient norm 또는 학습률 효과를 모두 보존한다 |
| 실행 시간과 steps/s는 계측된 조건의 기술 통계다 | 같은 UTD이므로 같은 계산 비용이고, 기록된 속도로 모든 방법의 효율을 순위화할 수 있다 |

### 9.2 Main body와 appendix의 역할

Main body에는 문제 설정, 핵심 support rule·loss, strongest relevant comparison, 핵심 mechanism 결과와 그 불확실성, 중요한 실패·비용 제한을 남긴다. PAL과 새 count-matched 결과는 novelty·정당성 지적에 직접 연결되므로 핵심 요약을 appendix에만 넣지 않는 편이 적절하다.

Appendix에는 full seed tables, 전체 curves, 실행 config·provenance, 자세한 hyperparameter grid, 모든 retrieval·phase 통계, 코드 검증을 둔다. Negative results를 appendix로 옮기더라도 본문에서 존재와 의미를 분명히 언급해야 한다.

### 9.3 원고를 읽는 연구자가 구분해야 할 세 가지

1. **관측:** 이 task/backbone/seed 집합에서 평균·CI·실패 패턴이 어떻게 나왔는가.
2. **해석 또는 가설:** Support 배정, 조회 제한, entropy, representation이 그 결과와 어떤 관련이 있을 수 있는가.
3. **아직 없는 증거:** 별도 policy-action 오차, 원인을 바꾸는 intervention, 고정 context 전이, 통제된 wall-clock 비교.

후속 실험은 평가 범위와 해석 가능한 자료를 늘렸지만, 모든 review concern을 해결했다고 표시할 단계는 아니다. 특히 novelty의 핵심인 구체적인 support 신호의 추가 가치는 평균 성능, mechanism 통제, 적용 범위, 필요한 수작업을 함께 보며 판단해야 한다.

## 10. 결과와 코드의 검증 및 인계 주의사항

### 10.1 이번 정리의 검증 범위

- 공개 후속 아카이브: 6개 실행·분석 family, 497개 source file, 461개 Python file의 parse, 총 555개 archive file 무결성 검사 통과.
- SAC compact artifacts: task별 20회, 각 4,000 evaluation points, config·완료 상태·지표·CI·manifest 재검증 통과.
- DDPG HER/SABER extension artifact: 신규 10회 + historical 10회, 4,000 evaluation points, source 42개, paired intervals 9개 재생성 검증 통과.
- 현재 큐: 기준 시각의 step까지 핵심 지표 finite, exact evaluation grid 확인. 이는 아직 끝나지 않은 실행의 미래 안정성을 보장하지 않는다.

이 검증은 저장된 자료의 정합성·수치 재생성을 확인한 것이다. 모든 알고리즘에 결함이 없거나, 통계적·인과적 결론이 참이거나, 깨끗한 임의 환경에서 전 학습을 즉시 재현할 수 있다는 보증은 아니다.

### 10.2 실행 결과를 바꾸지 않는 관리

새 protocol이나 수정 실험은 기존 frozen source·결과에 덮어쓰지 않는다. Historical config가 없으면 없다고 기록하고, 복원한 설정을 원 실행 config처럼 표시하지 않는다. 같은 seed 번호의 다른 실행을 추가 독립 관측으로 무조건 합치지 않는다. Low-success run은 정상 완료 결과로 보존한다.

현재 public archive는 실제 실행 source와 compact 결과를 모은 연구용 보관본이다. 일부 launcher에는 원래 gate·원시 자료·checkpoint 의존성이 남아 있다. 제출용 code supplement는 결과 재생성, clean installation, smoke, source provenance, 익명성, 라이선스·third-party notice를 별도로 점검해야 한다. 공개 Git SHA와 저장소 링크를 그대로 익명 제출물에 포함하지 않는다.

### 10.3 기준 자료의 식별

- 제출 PDF SHA256: **ca0d3168dcd4562c4ea0a35bc61b1bfe48aad824ec2a5ecd9ed713428bbba23c**
- 공개 결과 commit: **a3c049464704ed2104f571ffc9b20925a339b51f**
- 상태를 읽은 queue JSON의 SHA256: **38a68e79c224c909f08fd61e846cbee6dbb8e17746789c19649957b948179aa8**
- 진행 상태 원본: experiments_post_review/2026-10-24_draweropen_ddpg_main_seed_extension/outputs/queue_status.json
- 결과 원본: 같은 독립 실험 폴더의 outputs/production 및 각 protocol·snapshot manifest.
- 이 보고서 작성 과정에서는 학습 코드·queue·기존 원고·기존 보고서를 수정하지 않았다.

## 11. 완료된 신규 100회와 historical reference의 seed별 결과

아래는 모든 신규 완료 실행의 AUC/FQ-AUC/Final/Hit90다. 모든 신규 실행은 2M step이다. NA는 Hit90 미도달이다. Smoke·중단 attempt·재사용 reference는 신규 실행으로 다시 세지 않는다.

### 11.1 DrawerOpen DDPG 후속 비교 50회

Support 10회, Random assignment 10회, TD Top-k 10회, PAL 10회, Random-512 10회다. PAL·Tail 비교에서 재사용한 Support 결과는 아래 SABER 10행과 동일하므로 다시 추가하지 않았다.

| 방법 | Seed | AUC | FQ-AUC | Final | Hit90 |
| --- | --- | --- | --- | --- | --- |
| SABER | 1 | 0.776633 | 1.000000 | 1.000 | 460k |
| Random assignment | 1 | 0.869347 | 1.000000 | 1.000 | 250k |
| TD-error Top-k | 1 | 0.807538 | 0.999000 | 1.000 | 390k |
| SABER | 2 | 0.744975 | 0.985000 | 1.000 | 520k |
| Random assignment | 2 | 0.739447 | 0.999000 | 1.000 | 530k |
| TD-error Top-k | 2 | 0.792965 | 0.997000 | 1.000 | 420k |
| SABER | 3 | 0.789070 | 0.987500 | 0.950 | 450k |
| Random assignment | 3 | 0.000000 | 0.000000 | 0.000 | NA |
| TD-error Top-k | 3 | 0.512814 | 0.988000 | 1.000 | 920k |
| SABER | 4 | 0.818342 | 0.992000 | 1.000 | 350k |
| Random assignment | 4 | 0.690704 | 1.000000 | 1.000 | 600k |
| TD-error Top-k | 4 | 0.809799 | 0.998000 | 1.000 | 400k |
| SABER | 5 | 0.646734 | 1.000000 | 1.000 | 720k |
| Random assignment | 5 | 0.725879 | 0.994000 | 1.000 | 500k |
| TD-error Top-k | 5 | 0.773367 | 0.991000 | 1.000 | 460k |
| SABER | 6 | 0.765578 | 0.989000 | 1.000 | 470k |
| Random assignment | 6 | 0.714824 | 0.993000 | 1.000 | 590k |
| TD-error Top-k | 6 | 0.000000 | 0.000000 | 0.000 | NA |
| SABER | 7 | 0.000000 | 0.000000 | 0.000 | NA |
| Random assignment | 7 | 0.732663 | 0.993000 | 1.000 | 550k |
| TD-error Top-k | 7 | 0.000000 | 0.000000 | 0.000 | NA |
| SABER | 8 | 0.576382 | 0.998000 | 1.000 | 830k |
| Random assignment | 8 | 0.000000 | 0.000000 | 0.000 | NA |
| TD-error Top-k | 8 | 0.000000 | 0.000000 | 0.000 | NA |
| SABER | 9 | 0.570980 | 0.977500 | 0.950 | 850k |
| Random assignment | 9 | 0.000000 | 0.000000 | 0.000 | NA |
| TD-error Top-k | 9 | 0.808543 | 0.997000 | 1.000 | 400k |
| SABER | 10 | 0.837437 | 1.000000 | 1.000 | 340k |
| Random assignment | 10 | 0.683920 | 0.998000 | 1.000 | 660k |
| TD-error Top-k | 10 | 0.841206 | 0.966000 | 1.000 | 330k |
| HER+PAL | 1 | 0.715075 | 0.998000 | 1.000 | 590k |
| HER+PAL | 2 | 0.000000 | 0.000000 | 0.000 | NA |
| HER+PAL | 3 | 0.000000 | 0.000000 | 0.000 | NA |
| HER+PAL | 4 | 0.818342 | 0.999000 | 1.000 | 400k |
| HER+PAL | 5 | 0.557035 | 0.983000 | 1.000 | 850k |
| HER+PAL | 6 | 0.689698 | 0.998000 | 1.000 | 640k |
| HER+PAL | 7 | 0.809296 | 0.995000 | 1.000 | 410k |
| HER+PAL | 8 | 0.691206 | 0.962000 | 1.000 | 600k |
| HER+PAL | 9 | 0.509045 | 0.993000 | 1.000 | 930k |
| HER+PAL | 10 | 0.663819 | 0.998000 | 1.000 | 650k |
| SABER Random-512 | 1 | 0.780905 | 0.998000 | 1.000 | 420k |
| SABER Random-512 | 2 | 0.000000 | 0.000000 | 0.000 | NA |
| SABER Random-512 | 3 | 0.000000 | 0.000000 | 0.000 | NA |
| SABER Random-512 | 4 | 0.656030 | 0.999000 | 1.000 | 670k |
| SABER Random-512 | 5 | 0.569095 | 0.949500 | 1.000 | 870k |
| SABER Random-512 | 6 | 0.846482 | 1.000000 | 1.000 | 350k |
| SABER Random-512 | 7 | 0.720854 | 0.989000 | 0.900 | 550k |
| SABER Random-512 | 8 | 0.000000 | 0.000000 | 0.000 | NA |
| SABER Random-512 | 9 | 0.728894 | 0.985000 | 1.000 | 550k |
| SABER Random-512 | 10 | 0.000000 | 0.000000 | 0.000 | NA |

### 11.2 DrawerOpen SAC 20회

| 방법 | Seed | AUC | FQ-AUC | Final | Hit90 |
| --- | --- | --- | --- | --- | --- |
| HER | 1 | 0.824874 | 1.000000 | 1.000 | 340k |
| HER | 2 | 0.002010 | 0.008000 | 0.000 | NA |
| HER | 3 | 0.001256 | 0.005000 | 0.000 | NA |
| HER | 4 | 0.748744 | 0.995000 | 1.000 | 430k |
| HER | 5 | 0.658291 | 0.996000 | 1.000 | 570k |
| SABER | 1 | 0.731407 | 1.000000 | 1.000 | 590k |
| SABER | 2 | 0.000251 | 0.001000 | 0.000 | NA |
| SABER | 3 | 0.000000 | 0.000000 | 0.000 | NA |
| SABER | 4 | 0.803518 | 0.995000 | 1.000 | 360k |
| SABER | 5 | 0.707286 | 1.000000 | 1.000 | 480k |
| HGR | 1 | 0.723618 | 0.995000 | 1.000 | 550k |
| HGR | 2 | 0.660050 | 1.000000 | 1.000 | 690k |
| HGR | 3 | 0.800251 | 0.998000 | 1.000 | 430k |
| HGR | 4 | 0.719598 | 0.999000 | 1.000 | 550k |
| HGR | 5 | 0.629397 | 0.984000 | 1.000 | 770k |
| PER-full | 1 | 0.864824 | 0.960000 | 1.000 | 260k |
| PER-full | 2 | 0.844221 | 0.982000 | 1.000 | 320k |
| PER-full | 3 | 0.776131 | 0.999000 | 1.000 | 450k |
| PER-full | 4 | 0.820101 | 0.993000 | 1.000 | 360k |
| PER-full | 5 | 0.682663 | 0.986000 | 1.000 | 620k |

### 11.3 DrawerClose SAC 20회

| 방법 | Seed | AUC | FQ-AUC | Final | Hit90 |
| --- | --- | --- | --- | --- | --- |
| HER | 1 | 0.102261 | 0.030000 | 0.000 | 40k |
| HER | 2 | 0.136181 | 0.104000 | 0.000 | 50k |
| HER | 3 | 0.123744 | 0.106000 | 0.000 | 30k |
| HER | 4 | 0.767588 | 0.953000 | 1.000 | 30k |
| HER | 5 | 0.173367 | 0.044500 | 0.000 | 40k |
| SABER | 1 | 0.192211 | 0.025000 | 0.000 | 110k |
| SABER | 2 | 0.116834 | 0.025000 | 0.000 | 20k |
| SABER | 3 | 0.064070 | 0.019000 | 0.000 | 30k |
| SABER | 4 | 0.095477 | 0.018000 | 0.000 | 40k |
| SABER | 5 | 0.047990 | 0.001000 | 0.000 | 40k |
| HGR | 1 | 0.107035 | 0.002000 | 0.000 | 40k |
| HGR | 2 | 0.166332 | 0.023000 | 0.000 | 30k |
| HGR | 3 | 0.279648 | 0.002000 | 0.000 | 30k |
| HGR | 4 | 0.407915 | 0.282500 | 0.150 | 20k |
| HGR | 5 | 0.097990 | 0.000000 | 0.000 | 40k |
| PER-full | 1 | 0.305779 | 0.061000 | 0.000 | 60k |
| PER-full | 2 | 0.821357 | 0.958000 | 1.000 | 70k |
| PER-full | 3 | 0.069347 | 0.029000 | 0.000 | 50k |
| PER-full | 4 | 0.709548 | 0.965000 | 0.900 | 250k |
| PER-full | 5 | 0.530528 | 0.941500 | 0.850 | 1060k |

### 11.4 원논문 DrawerOpen DDPG seed 확장 신규 10회

위 11.1의 SABER 실행과 별도 집합이다.

| 방법 | Seed | AUC | FQ-AUC | Final | Hit90 |
| --- | --- | --- | --- | --- | --- |
| HER | 6 | 0.782663 | 0.993000 | 1.000 | 450k |
| HER | 7 | 0.752513 | 0.960000 | 1.000 | 470k |
| HER | 8 | 0.752010 | 0.999000 | 1.000 | 510k |
| HER | 9 | 0.686935 | 0.995500 | 1.000 | 660k |
| HER | 10 | 0.000000 | 0.000000 | 0.000 | NA |
| SABER | 6 | 0.765578 | 0.989000 | 1.000 | 470k |
| SABER | 7 | 0.000000 | 0.000000 | 0.000 | NA |
| SABER | 8 | 0.576382 | 0.998000 | 1.000 | 830k |
| SABER | 9 | 0.570980 | 0.977500 | 0.950 | 850k |
| SABER | 10 | 0.837437 | 1.000000 | 1.000 | 340k |

### 11.5 Pooled 분석에 사용한 원논문 reference 10회

아래는 AAAI 이후 신규 100회에 포함되지 않는다. 11.4와 결합해 6.7의 historical/extension/pooled 분석을 구성한다.

| 방법 | Seed | AUC | FQ-AUC | Final | Hit90 |
| --- | --- | --- | --- | --- | --- |
| HER | 1 | 0.460302 | 0.970000 | 1.000 | 1090k |
| HER | 2 | 0.000000 | 0.000000 | 0.000 | NA |
| HER | 3 | 0.000000 | 0.000000 | 0.000 | NA |
| HER | 4 | 0.000000 | 0.000000 | 0.000 | NA |
| HER | 5 | 0.576256 | 0.993500 | 0.950 | 830k |
| SABER | 1 | 0.757286 | 1.000000 | 1.000 | 510k |
| SABER | 2 | 0.786935 | 0.977000 | 1.000 | 420k |
| SABER | 3 | 0.705025 | 1.000000 | 1.000 | 580k |
| SABER | 4 | 0.713065 | 0.960000 | 1.000 | 500k |
| SABER | 5 | 0.806030 | 1.000000 | 1.000 | 400k |

## 12. 자료 출처와 다음 연구자에게 넘길 판단

아래 GitHub 링크는 이동하는 main이 아니라 본 보고서 기준 commit에 고정했다. 따라서 이후 결과가 추가되어도 이 보고서의 근거를 다시 확인할 수 있다.

| 자료 | 위치·역할 |
|---|---|
| AAAI 리뷰 | 연구자가 전달한 OpenReview submission 36289의 Phase 1 리뷰·결정문. 사람 리뷰 4/6점, confidence 3/3 |
| 제출 본문 | [AAAI PDF][MAIN] |
| 기존 supplementary | [Technical Supplement source][SUPP] |
| 배정 비교 | [README와 source/results][ASSIGN] |
| PAL 비교 | [README와 source/results][PAL], [고정 protocol][PAL-PROTOCOL] |
| 오프라인 조회·실패 | [상세 분석 보고서][OFFLINE], [복구 기록][RECOVERY] |
| Online 조회 비교 | [README][ONLINE], [완료 후 양쪽 checkpoint 진단][ONLINE-FINAL] |
| DrawerOpen SAC | [완료 결과·protocol·provenance][SAC-OPEN] |
| DrawerClose SAC | [완료 결과·protocol·provenance][SAC-CLOSE] |
| 원논문 DDPG 확장 완료분 | [HER/SABER 완료 artifact][EXT] |
| 전체 공개 후속 목록 | [ICML_27 index][INDEX] |
| 과거 중간 기록 | [10월 6일 보고서][OLD-REPORT]. 당시 진행률·예약 상태는 현재 상태가 아님 |

다음 공동연구자가 먼저 확인할 사항은 다음과 같다.

1. 기존 원고의 contribution과 Method를 4절 방향으로 고칠지 합의한다. **가독성 수정은 추가 학습 종료를 기다릴 필요가 없다.**
2. 본문 각 표에 사용할 실행 집합을 고정하고, 6.1의 Support와 6.7의 pooled SABER를 혼용하지 않는다.
3. SAC의 표현상 주의점을 점검한 뒤, 기존 결과의 범위와 필요한 별도 검증을 정한다. 원인을 확인하기 전 유리한 설명을 채택하지 않는다.
4. 현재 여덟 조건 확장을 완료한 후 결과를 모두 반영한다. 다음 실험은 8절의 질문·필요 데이터·예산을 먼저 정하고 승인한다.
5. 논문의 중심 주장에 반드시 필요한 추가 증거와, 있으면 좋은 범위 확장을 구별한다. 불확실한 결과도 연구 기록과 원고의 한계에 반영한다.

**종합 판단:** 후속 작업으로 가까운 선행방법 비교, 더 엄격한 배정 대조군, 조회의 근사성, 실패·포화, Drawer SAC 범위, seed 확장 자료가 확보되었다. 이 자료들은 리뷰에 더 구체적으로 답할 수 있게 하지만, exact support criterion의 필요성이나 광범위한 우월성을 확정하지는 않는다. 다음 원고는 **구체적인 support 규칙, 통제된 평가, 조건별 이득과 한계**를 중심으로 재구성하는 것이 현재 근거와 가장 잘 맞는다.

[MAIN]: https://github.com/Hyunwoojooo/saber-rl/blob/a3c049464704ed2104f571ffc9b20925a339b51f/paper/SABER_Support_Conditioned_Adaptive_Bellman_Error_Reweighting.pdf
[SUPP]: https://github.com/Hyunwoojooo/saber-rl/blob/a3c049464704ed2104f571ffc9b20925a339b51f/paper/technical_supplement.tex
[ASSIGN]: https://github.com/Hyunwoojooo/saber-rl/blob/a3c049464704ed2104f571ffc9b20925a339b51f/ICML_27/experiments/draweropen_ddpg_saber_vs_random_vs_td_topk/README.md
[PAL]: https://github.com/Hyunwoojooo/saber-rl/blob/a3c049464704ed2104f571ffc9b20925a339b51f/ICML_27/experiments/draweropen_ddpg_saber_vs_pal/README.md
[PAL-PROTOCOL]: https://github.com/Hyunwoojooo/saber-rl/blob/a3c049464704ed2104f571ffc9b20925a339b51f/ICML_27/experiments/draweropen_ddpg_saber_vs_pal/source/PROTOCOL.md
[OFFLINE]: https://github.com/Hyunwoojooo/saber-rl/blob/a3c049464704ed2104f571ffc9b20925a339b51f/ICML_27/experiments/draweropen_ddpg_retrieval_and_failure_diagnostics/results/outputs/final/REPORT.md
[RECOVERY]: https://github.com/Hyunwoojooo/saber-rl/blob/a3c049464704ed2104f571ffc9b20925a339b51f/ICML_27/experiments/draweropen_ddpg_retrieval_and_failure_diagnostics/source/RECOVERY.md
[ONLINE]: https://github.com/Hyunwoojooo/saber-rl/blob/a3c049464704ed2104f571ffc9b20925a339b51f/ICML_27/experiments/draweropen_ddpg_saber_tail_vs_random512/README.md
[ONLINE-FINAL]: https://github.com/Hyunwoojooo/saber-rl/blob/a3c049464704ed2104f571ffc9b20925a339b51f/ICML_27/experiments/draweropen_ddpg_saber_tail_vs_random512/results/outputs/final_retrieval/report.json
[SAC-OPEN]: https://github.com/Hyunwoojooo/saber-rl/blob/a3c049464704ed2104f571ffc9b20925a339b51f/metaworld_Drawer_Open_SAC/paper_artifacts/drawer_open_sac_her_saber_hgr_per_2m/README.md
[SAC-CLOSE]: https://github.com/Hyunwoojooo/saber-rl/blob/a3c049464704ed2104f571ffc9b20925a339b51f/metaworld_Drawer_Close_SAC/paper_artifacts/drawer_close_sac_her_saber_hgr_per_2m/README.md
[EXT]: https://github.com/Hyunwoojooo/saber-rl/blob/a3c049464704ed2104f571ffc9b20925a339b51f/metaworld_Drawer_Open_DDPG/paper_artifacts/drawer_open_ddpg_her_saber_seed_extension_2m/README.md
[INDEX]: https://github.com/Hyunwoojooo/saber-rl/blob/a3c049464704ed2104f571ffc9b20925a339b51f/ICML_27/README.md
[OLD-REPORT]: https://github.com/Hyunwoojooo/saber-rl/blob/a3c049464704ed2104f571ffc9b20925a339b51f/ICML_27/2026-10-06_post_review_experiment_status_ko.md
[PAL-PAPER]: https://proceedings.neurips.cc/paper/2020/hash/a3bf6e4db673b6449c2f7d13ee6ec9c0-Abstract.html
[ACTOR-PAPER]: https://www.jair.org/index.php/jair/article/view/14819
[LABER-PAPER]: https://proceedings.mlr.press/v162/lahire22a.html
[ICML-REVIEW]: https://icml.cc/Conferences/2026/ReviewerInstructions

# SABER 원고 오류 검수 및 Peer Review

- 검토 대상 문서: `saber_aaai27_draft_kr(1).pdf`
- 문서 성격: 학회 제출용 연구논문 초안
- 분야: 컴퓨터·AI·강화학습, sparse-reward goal-conditioned reinforcement learning, HER, off-policy actor-critic, robotic manipulation

---

# 오류 검수 결과

## 1. 전체 검토 범위

문서 유형: 학회 제출용 연구논문 초안
추정 분야: 컴퓨터·AI·강화학습, sparse-reward goal-conditioned reinforcement learning, HER, off-policy actor-critic, robotic manipulation
총 검토 페이지: PDF 9페이지 전체
주요 검토 기준: 강화학습 개념·수식·알고리즘 정의의 정확성, 실험 설계와 주장 간 일치성, 표 수치와 본문 설명 일치성, baseline 공정성, 통계 보고, 재현 가능성, 참고문헌·용어·조판 일관성
검토 시 특히 주의한 항목: SABER의 support predicate 정의, critic-only Bellman loss의 논리적 타당성, HER/PER/TD3/SAC 비교 구조, Table 1–5의 수치·명명·지표 정의, MetaWorld DrawerOpen random-control 해석, AAAI-style 논문 형식과 조판 상태

## 2. 오류 목록

### 오류 1

위치: SABER 문제 설정과 핵심 mechanism 설명 전반
PDF 페이지: 1, 3, 4
본문 페이지: 미표기
장/절: Introduction / Motivation / Critic Objective / Theoretical Interpretation
그림/표/프로그램/수식 번호: 수식 (7), (8), (12), (13), (18)

오류 유형: 내용 오류 / 분야별 전문 오류 / 수식·숫자·단위 오류 / 독자 관점 오류

심각도: 높음

원문:
“현재 policy action이 local replay action support 바깥에 있다면, critic은 그 action region에서 extrapolation에 의존할 수 있다.”
“aπ_i = πθ(o_i, g^tr_i)”
“L^SABER_Q(ϕ) = … Ω_i (Qϕ(x_i, a_i) − y_i)^2”
“support affects Bellman regression, not direct policy optimization.”

문제 설명:
논문은 reliability를 “current policy action”이 local replay-action support 안에 있는지로 판정한다. 그러나 실제 critic loss는 current policy action `aπ_i`가 아니라 replay action `a_i`에서의 Bellman error `Qϕ(x_i, a_i) − y_i`에 weight를 곱한다. 즉, support predicate가 평가하는 action과 loss가 fitting하는 action이 다르다. 이 연결고리가 명확히 증명되거나 실험적으로 검증되지 않으면 “policy action support가 critic Bellman fitting reliability를 의미한다”는 핵심 주장에 논리적 공백이 생긴다.

수정안:
다음 중 하나로 보완해야 한다.
“Although the critic regression is evaluated at replay actions, SABER uses the local support of the current policy action as a proxy for policy-improvement-relevant target reliability. We empirically validate this proxy by measuring the correlation between ψ_i and post-update Q-error around πθ(o_i, g_i).”
또는 support predicate를 replay action 기준으로 재정의한다.
“ψ_i = 1[b(a_i) ∈ B_i]” 또는 `Qϕ(x_i, πθ(x_i))` 근방의 auxiliary diagnostic을 추가한다.

비고:
현재 형태에서는 novelty 자체보다 mechanism 해석이 취약해 보인다. top-tier review에서는 major weakness로 지적될 가능성이 크다.

---

### 오류 2

위치: Support Indicator
PDF 페이지: 4
본문 페이지: 미표기
장/절: SABER / Support Indicator
그림/표/프로그램/수식 번호: 수식 (14)

오류 유형: 수식·숫자·단위 오류 / 분야별 전문 오류 / 독자 관점 오류

심각도: 높음

원문:
“ψ_i = 1 [|L_i| ≥ N_min ∧ |A^sup_i| ≥ M_min ∧ b^π_i ∈ B_i]”

문제 설명:
`A^sup_i`가 수식 이전에 정의되어 있지 않다. 직전 절에서는 `L_i`, `B_i`, `aπ_i`, `bπ_i`만 정의되어 있다. 독자는 `A^sup_i`가 unique action bin 집합인지, support 가능한 action set인지, `B_i`와 같은 것인지 판단할 수 없다.

수정안:
수식 (14) 이전에 다음 정의를 추가한다.
`A^sup_i = {b(a_j) : j ∈ L_i}` 또는 `A^sup_i = unique(B_i)`
그 다음 수식 (14)를 다음처럼 정리한다.
`ψ_i = 1[|L_i| ≥ N_min ∧ |unique(B_i)| ≥ M_min ∧ b^π_i ∈ B_i]`

비고:
알고리즘 재현 가능성을 직접 해치는 정의 누락이다.

---

### 오류 3

위치: Support Indicator threshold 설명
PDF 페이지: 4
본문 페이지: 미표기
장/절: SABER / Support Indicator
그림/표/프로그램/수식 번호: 수식 (14)

오류 유형: 수식·숫자·단위 오류 / 코드·실습·절차 오류 / 분야별 전문 오류

심각도: 높음

원문:
“대표 threshold는 M_B = 16, M_Q_min = 4, N_local_min = 4, N_gap_min = 128, λ_S = 1.0 …”

문제 설명:
수식 (14)는 `N_min`, `M_min`을 사용하지만, 본문 설명은 `M_B`, `M_Q_min`, `N_local_min`, `N_gap_min`을 사용한다. 기호 체계가 맞지 않고 각 threshold가 무엇을 제어하는지도 설명되어 있지 않다. 특히 `N_gap_min`은 horizon gap인지, replay index gap인지, local pool 간격인지 알 수 없다.

수정안:
수식과 본문 기호를 통일한다. 예:
`N_min = N_local_min = 4`
`M_min = M_Q_min = 4`
`M_B = 16`은 action bin 개수라면 “number of action bins per dimension” 또는 “total action-bin vocabulary size”로 명시한다.
`N_gap_min = 128`은 “minimum temporal/index gap used to avoid near-duplicate transitions” 등으로 정의한다.

비고:
구현자나 reviewer가 SABER를 재현할 수 없는 상태다.

---

### 오류 4

위치: Local Context
PDF 페이지: 3
본문 페이지: 미표기
장/절: SABER / Local Context
그림/표/프로그램/수식 번호: 수식 (9), (10), (11)

오류 유형: 분야별 전문 오류 / 코드·실습·절차 오류 / 독자 관점 오류

심각도: 높음

원문:
“이 context는 observation, training goal, horizon information을 coarse bin 또는 hash로 변환한 것이다.”
“FetchPickAndPlace에서는 hand-object distance, object-goal distance, object height, gripper gap, horizon bin 등이 context에 들어갈 수 있다.”

문제 설명:
`coarse bin 또는 hash`라고만 되어 있고 실제 bin width, normalization, distance metric, hash collision 처리, task별 feature 조합이 주어지지 않는다. SABER의 핵심은 local support lookup인데, local context가 정확히 정의되지 않으면 논문 method가 아니라 구현 관찰에 가까워진다.

수정안:
각 task별로 다음 표를 추가한다.
`Task / Context features / Bin width / Normalization / Horizon bin / Action binning / Minimum pool size / Lookup complexity`
본문에는 다음 문장을 추가한다.
“Unless otherwise stated, all continuous context features are normalized by the task-specific observation range and discretized into K equal-width bins. The exact feature list and bin widths are reported in Table X.”

비고:
현재는 “들어갈 수 있다” 수준의 서술이라 방법 정의로 부족하다.

---

### 오류 5

위치: Preliminaries / HER training goal
PDF 페이지: 3
본문 페이지: 미표기
장/절: Preliminaries / Goal-Conditioned MDP
그림/표/프로그램/수식 번호: 수식 (3), (7)

오류 유형: 내용 오류 / 분야별 전문 오류 / 코드·실습·절차 오류 / 독자 관점 오류

심각도: 높음

원문:
“g^tr_i ∼ H(τ_i).”
“r(o_i, a_i, o′_i, g^tr_i)”

문제 설명:
HER relabeling distribution `H(τ_i)`가 future strategy인지, final strategy인지, episode strategy인지, 몇 개의 hindsight goal을 샘플링하는지 정의되어 있지 않다. 또한 sparse reward가 `0/1`인지 `-1/0`인지, success threshold가 무엇인지 제시되지 않는다. Fetch 계열 task에서는 reward convention과 success threshold가 성능 수치 해석에 직접 영향을 준다.

수정안:
다음 정의를 추가한다.
“HER uses the future strategy with k = X relabeled goals per transition. The sparse reward is r = 0 if ||achieved_goal − desired_goal|| < ε_success and r = −1 otherwise.”
또는 사용하는 구현의 정확한 convention을 명시한다.

비고:
실험 수치의 재현성과 baseline 공정성에 영향을 준다.

---

### 오류 6

위치: Bellman Regression / Actor objective
PDF 페이지: 3–4
본문 페이지: 미표기
장/절: Preliminaries / Critic Objective
그림/표/프로그램/수식 번호: 수식 (7), (18)

오류 유형: 분야별 전문 오류 / 수식·숫자·단위 오류 / 독자 관점 오류

심각도: 높음

원문:
“TD3와 SAC에서는 target 계산 방식이 backbone에 맞게 달라지지만 … SABER는 이 regression loss의 sample weight만 바꾼다.”
“SAC에서는 entropy-regularized actor objective를 유지한다.”

문제 설명:
논문은 DDPG, TD3, SAC 모두에서 실험했다고 주장하지만, 수식은 DDPG-style target만 제시한다. TD3의 clipped double Q, target policy smoothing, delayed actor update, SAC의 entropy term과 stochastic policy target이 SABER weight와 어떻게 결합되는지 명시되어 있지 않다.

수정안:
TD3와 SAC에 대해 별도 target 수식을 추가한다. 예:
TD3: `y_i = r_i + γ(1 − d_i) min_k Q̄_k(o′_i, g_i, π̄(o′_i, g_i) + ε)`
SAC: `y_i = r_i + γ(1 − d_i)(min_k Q̄_k(o′_i, g_i, a′) − α log πθ(a′|o′_i,g_i))`
그 후 “Ω_i is applied identically to each critic loss term”이라고 명시한다.

비고:
backbone-compatible method라는 contribution을 뒷받침하려면 필수 보완이다.

---

### 오류 7

위치: Table 1
PDF 페이지: 6
본문 페이지: 미표기
장/절: Main Result / FetchPickAndPlace
그림/표/프로그램/수식 번호: Table 1

오류 유형: 수식·숫자·단위 오류 / 그림·표·도식 오류 / 내용 오류

심각도: 치명적

원문:
“TD3 HER: Best Success 0.880 ± 0.144, First Hit ≥ 0.90 980k”

문제 설명:
`First Hit ≥ 0.90`이 평균 success curve가 0.90을 처음 넘은 step을 의미한다면, 같은 row의 `Best Success` 평균은 최소 0.90 이상이어야 한다. 그런데 Table 1에서는 TD3 HER의 Best Success가 0.880으로 표시되어 있어 지표 정의 또는 수치 중 하나가 맞지 않는다. 만약 First Hit가 “어느 한 seed가 0.90을 넘은 시점”이라면 column명이 잘못되었고, 만약 평균 기준이라면 수치가 잘못되었다.

수정안:
다음 중 하나로 수정해야 한다.
1. 평균 기준이면 `First Hit ≥ 0.90 = N/A` 또는 실제 step으로 재계산.
2. seed 기준이면 column명을 `Earliest seed hit ≥ 0.90` 또는 `#Seeds hit ≥ 0.90`로 변경.
3. smoothing 기준이면 “smoothed mean curve” 등 계산 방식을 명시.

비고:
표의 신뢰도를 직접 훼손하는 수치·정의 불일치다.

---

### 오류 8

위치: Tasks and Metrics / Tables
PDF 페이지: 4, 6–8
본문 페이지: 미표기
장/절: Experiments / Tasks and Metrics
그림/표/프로그램/수식 번호: Table 1–5

오류 유형: 내용 오류 / 그림·표·도식 오류 / 독자 관점 오류

심각도: 높음

원문:
“주요 metric은 final success, best success, success AUC, first hit threshold, runtime steps/sec, drop best-final이다.”

문제 설명:
`drop best-final`은 주요 metric이라고 되어 있지만 Table 1–5에 보고되지 않는다. `first hit threshold`도 Table 1에만 있고 FetchSlide에는 없으며, DrawerOpen에서는 `Hit90`이라는 다른 표기가 사용된다. `success AUC`와 `Last-Quarter AUC`의 계산 방식도 정의되어 있지 않다.

수정안:
모든 task 표에 같은 metric을 일관되게 제공하거나, 사용하지 않는 metric은 본문에서 제거한다.
예:
“Metrics are final success, best success, normalized success AUC, and steps/sec. First-hit and best-final drop are reported only for FPP in Table 1.”
또는 모든 표에 `Drop Best-Final`, `First Hit ≥ 0.90`을 추가한다.

비고:
독자가 task 간 성능을 비교하기 어렵다.

---

### 오류 9

위치: Comparison with PER-Style Weighting / Table 2
PDF 페이지: 5–6
본문 페이지: 미표기
장/절: Experiments / Comparison with PER-Style Weighting
그림/표/프로그램/수식 번호: Table 2

오류 유형: 내용 오류 / 분야별 전문 오류 / 코드·실습·절차 오류

심각도: 높음

원문:
“PER-style weighting은 AUC에서 HER보다 높지만 SABER보다 낮고…”
“DDPG+HER+PER-style … DDPG+HER+PER-full …”

문제 설명:
PER-style weighting의 정확한 수식, priority 계산, importance sampling correction 사용 여부, α/β hyperparameter, TD-error clipping 여부가 없다. 또한 PER 비교는 FPP DDPG 하나에만 제시되어 RQ3의 “SABER의 이득은 TD-error/PER-style weighting으로 설명될 수 있는가?”라는 질문을 충분히 검증하지 못한다.

수정안:
PER-style loss를 다음처럼 명시한다.
`w_i^PER ∝ (|δ_i| + ε)^α`, normalized per batch, with or without IS correction.
그리고 최소한 FPP의 TD3/SAC, FetchSlide의 TD3/SAC에도 PER-style 또는 full PER comparison을 추가한다.

비고:
현재 PER 비교는 “다른 신호를 사용한다”는 정성 주장에는 도움이 되지만, mechanism 배제 증거로는 약하다.

---

### 오류 10

위치: Actor-Scope and Random Controls / Table 5
PDF 페이지: 5, 8
본문 페이지: 미표기
장/절: Experiments / Actor-Scope and Random Controls
그림/표/프로그램/수식 번호: Table 5

오류 유형: 내용 오류 / 분야별 전문 오류 / 그림·표·도식 오류

심각도: 높음

원문:
“Random-Support가 Critic-Only와 비슷하거나 더 높은 AUC를 보였다는 점은 exact structured support rule만이 유일한 원인이라는 강한 claim을 약화시킨다.”
“B3-Random-Support … Success AUC 0.426 ± 0.392”
“B3-Critic-Only … Success AUC 0.361 ± 0.338”

문제 설명:
논문은 이 limitation을 정직하게 언급하지만, Abstract와 Conclusion에서는 여전히 SABER를 support-aware objective로 강하게 요약한다. Random-Support가 Critic-Only보다 높은 AUC를 보인다면, 이득이 structured local support 때문인지, 단순 reweighting regularization 때문인지 분리되지 않는다.

수정안:
Abstract와 Conclusion의 claim을 낮춘다.
수정 예:
“DrawerOpen controls suggest that part of the gain may come from reweighting-induced regularization rather than the exact structured support predicate; we therefore interpret support-awareness as a plausible but not fully isolated mechanism.”

비고:
저자가 이미 인지한 limitation이므로, 주장 강도 조절이 핵심이다.

---

### 오류 11

위치: Main Result / Abstract / Conclusion
PDF 페이지: 1, 5, 6
본문 페이지: 미표기
장/절: Abstract / Main Result / Conclusion
그림/표/프로그램/수식 번호: Table 1

오류 유형: 내용 오류 / 수식·숫자·단위 오류 / 독자 관점 오류

심각도: 높음

원문:
“FPP에서는 SABER가 세 backbone 모두에서 final success와 success AUC를 개선한다.”
“DDPG에서는 final success가 0.940에서 0.960으로 소폭 상승…”

문제 설명:
seeds 1–5, 즉 n=5 결과인데 유의성 검정, confidence interval, per-seed curve, bootstrap test가 없다. 특히 DDPG FPP final success 0.940 ± 0.042 vs 0.960 ± 0.042는 차이가 작고 error bar가 겹친다. “개선한다”를 확정 표현으로 쓰기에는 통계적 근거가 부족하다.

수정안:
“mean final success and AUC are higher”로 표현을 낮추고, paired seed test 또는 bootstrap confidence interval을 추가한다.
수정 예:
“On FPP, SABER improves mean success AUC across all three backbones; final success gains are strongest for TD3 and SAC and modest for DDPG.”

비고:
학회 reviewer는 “statistical significance and seed robustness”를 요구할 가능성이 높다.

---

### 오류 12

위치: Table 3
PDF 페이지: 7
본문 페이지: 미표기
장/절: Secondary Result / FetchSlide
그림/표/프로그램/수식 번호: Table 3

오류 유형: 그림·표·도식 오류 / 편집·조판·교정 오류 / 독자 관점 오류

심각도: 중간

원문:
“DDPG HER+SABER”
“TD3 SABER”
“SAC SABER”

문제 설명:
같은 Table 3 안에서 DDPG는 `HER+SABER`로 쓰고, TD3/SAC는 `SABER`로만 쓴다. 본문은 `TD3+SABER`, `SAC+SABER`라고 말하지만, 논문 전체 claim은 HER baseline 위에 SABER를 붙인다는 구조다. 표기만 보면 TD3/SAC는 HER 없이 SABER만 적용한 것처럼 읽힐 수 있다.

수정안:
Table 3의 Method column을 모두 통일한다.
`DDPG HER`, `DDPG HER+SABER`, `TD3 HER`, `TD3 HER+SABER`, `SAC HER`, `SAC HER+SABER`

비고:
작은 표기 문제처럼 보이나, baseline 비교 구조를 오해하게 만든다.

---

### 오류 13

위치: Table 4 / Table 5
PDF 페이지: 7–8
본문 페이지: 미표기
장/절: MetaWorld DrawerOpen
그림/표/프로그램/수식 번호: Table 4, Table 5

오류 유형: 수식·숫자·단위 오류 / 그림·표·도식 오류

심각도: 중간

원문:
Table 4: “TD3+HER+SABER … AUC 0.362 ± 0.338”
Table 5: “B3-Critic-Only … Success AUC 0.361 ± 0.338”

문제 설명:
Table 4의 TD3+HER+SABER와 Table 5의 B3-Critic-Only가 같은 critic-only SABER 결과라면 AUC가 0.362와 0.361로 다르다. 반올림 차이라면 반올림 기준이 달라졌고, 서로 다른 run이면 이를 명시해야 한다.

수정안:
같은 데이터라면 둘 다 `0.361 ± 0.338` 또는 둘 다 `0.362 ± 0.338`로 통일한다.
다른 데이터라면 Table 5 caption에 “B3 uses a separate run/configuration from Table 4”라고 명시한다.

비고:
수치가 작아도 table consistency 검수에서 걸리는 항목이다.

---

### 오류 14

위치: Table 5
PDF 페이지: 8
본문 페이지: 미표기
장/절: Actor-Scope and Random Controls
그림/표/프로그램/수식 번호: Table 5

오류 유형: 그림·표·도식 오류 / 편집·조판·교정 오류 / 독자 관점 오류

심각도: 중간

원문:
“B3-Critic-Only”, “B3-Actor-Only”, “B3-Random-Support”, “B3-Shuffled-Weight”

문제 설명:
`B3`가 무엇을 의미하는지 본문에 정의되어 있지 않다. binning configuration인지, backbone variant인지, experimental batch인지 알 수 없다.

수정안:
Table 5 caption 또는 본문에 다음을 추가한다.
“B3 denotes the third binning/support configuration used for DrawerOpen; all B3 variants share the same local context and λ_S unless otherwise stated.”
정의할 내용이 없다면 `B3-` prefix를 제거한다.

비고:
표 단독으로 읽는 reviewer가 실험 조건을 이해하기 어렵다.

---

### 오류 15

위치: Tasks and Metrics
PDF 페이지: 4
본문 페이지: 미표기
장/절: Experiments / Tasks and Metrics
그림/표/프로그램/수식 번호: 해당 없음

오류 유형: 내용 오류 / 편집·조판·교정 오류 / 독자 관점 오류

심각도: 중간

원문:
“FetchPush는 complete rows만 appendix 또는 보조 evidence로 제한한다.”

문제 설명:
업로드된 9페이지 문서에는 FetchPush 결과표나 appendix가 없다. 현재 원고만 기준으로 보면 독자가 존재하지 않는 appendix를 참조하게 된다.

수정안:
현재 버전에 appendix가 없다면 문장을 삭제한다.
수정 예:
“FetchPush results are omitted from the main paper because the current experimental table is incomplete.”
또는 appendix를 실제로 추가한다.

비고:
제출본에서 appendix가 함께 제출되지 않으면 명백한 편집 흔적이다.

---

### 오류 16

위치: MetaCog-RL Lineage
PDF 페이지: 2
본문 페이지: 미표기
장/절: Related Work / MetaCog-RL Lineage
그림/표/프로그램/수식 번호: 해당 없음

오류 유형: 내용 오류 / 분야별 전문 오류 / 편집·조판·교정 오류

심각도: 중간

원문:
“원래 MetaCog-RL은 RL 실패 원인을 ignorance, incompetence, impossibility로 나누고…”

문제 설명:
MetaCog-RL이 별도 선행연구라면 참고문헌에 citation이 필요하다. 현재 References에는 MetaCog-RL 관련 항목이 없다. 만약 저자의 미공개 내부 framework라면, 익명 제출에서 lineage를 길게 설명하는 것이 오히려 novelty와 anonymity를 흐릴 수 있다.

수정안:
공개 선행연구라면 citation을 추가한다.
미공개 아이디어라면 “This work was motivated by a broader diagnostic view of RL failures…” 정도로 축약하고 고유명 `MetaCog-RL`은 제거한다.

비고:
관련연구 섹션의 근거 없는 framework 설명으로 보일 수 있다.

---

### 오류 17

위치: Tasks and Metrics / Tables
PDF 페이지: 4, 6–7
본문 페이지: 미표기
장/절: Experiments
그림/표/프로그램/수식 번호: Table 1–4

오류 유형: 분야별 전문 오류 / 코드·실습·절차 오류 / 독자 관점 오류

심각도: 높음

원문:
“runtime steps/sec”
Table 1–3의 “Steps/sec” column

문제 설명:
Steps/sec는 hardware, CPU/GPU, 병렬 환경 수, replay buffer 구현, support lookup 자료구조, evaluation 포함 여부에 크게 좌우된다. 그러나 실험 환경 정보가 없다. FetchSlide에서 steps/sec가 119.08에서 26.85로 크게 감소하는 등 wall-clock claim이 중요한데, 재현 가능한 조건이 빠져 있다.

수정안:
실험 설정에 다음을 추가한다.
“Experiments used [CPU], [GPU], MuJoCo version, Gym/Gymnasium-Robotics version, number of parallel environments, evaluation frequency, and whether steps/sec includes evaluation.”
support lookup complexity와 caching 여부도 명시한다.

비고:
runtime overhead를 limitation으로 제시하는 논문에서는 필수 정보다.

---

### 오류 18

위치: References / Tasks
PDF 페이지: 4, 9
본문 페이지: 미표기
장/절: Experiments / References
그림/표/프로그램/수식 번호: References

오류 유형: 분야별 전문 오류 / 편집·조판·교정 오류

심각도: 중간

원문:
FetchPickAndPlace, FetchSlide, FetchPush를 benchmark로 사용하지만 References에는 Fetch Robotics/Gym/Gymnasium-Robotics 환경 citation이 없다.

문제 설명:
Fetch 환경은 논문 실험의 핵심 benchmark다. HER 논문 인용만으로는 사용한 environment suite, version, reward convention, success function 출처가 충분히 특정되지 않는다.

수정안:
OpenAI Gym Robotics 또는 Gymnasium-Robotics / Plappert et al. multi-goal RL benchmark citation을 추가하고, 환경명을 version까지 표기한다.
예: `FetchPickAndPlace-v1` 또는 `FetchPickAndPlaceDense-v?`가 아니라 sparse reward version임을 명시한다.

비고:
환경 버전 차이는 success 수치 재현에 영향을 줄 수 있다.

---

### 오류 19

위치: Related Work / Experiments
PDF 페이지: 2, 5–6
본문 페이지: 미표기
장/절: Related Work / Experiments
그림/표/프로그램/수식 번호: Table 2

오류 유형: 분야별 전문 오류 / 내용 오류 / 독자 관점 오류

심각도: 높음

원문:
Related Work는 HER, DDPG, TD3, SAC, PER, BCQ, CQL 중심으로 구성되어 있다.

문제 설명:
sparse goal-conditioned RL의 직접 경쟁축인 HER prioritization, curriculum-guided HER, multi-step HER, high replay ratio + HER 계열 baseline이 거의 반영되지 않았다. 특히 2024년 TMLR/OpenReview 논문은 sparse-reward goal-conditioned Robotics tasks에서 HER와 REDQ/high replay ratio를 결합해 이전 SOTA 대비 약 2배 sample efficiency를 보고하므로, 본 논문의 baseline 맥락에서 매우 중요하다.

수정안:
Related Work와 baseline에 최소한 다음 축을 추가한다.
CHER, energy-based hindsight prioritization, HGR, MHER/MHER(λ), REDQ+HER/high replay ratio.
실험 baseline으로는 적어도 FPP에서 REDQ+HER 또는 high replay ratio HER baseline을 포함한다.

비고:
top-tier conference review에서 “missing strong baselines”로 지적될 가능성이 크다.

---

### 오류 20

위치: Abstract / Related Work / Tables
PDF 페이지: 1–8
본문 페이지: 미표기
장/절: 전반
그림/표/프로그램/수식 번호: Table 3–5

오류 유형: 편집·조판·교정 오류 / 독자 관점 오류

심각도: 중간

원문:
“Meta-World DrawerOpen”
“MetaWorld DrawerOpen”
“TD3 SABER”
“TD3+HER+SABER”
“Hit90” / “First Hit ≥ 0.90”

문제 설명:
동일 개념의 표기가 혼용된다. official benchmark 명칭은 보통 `Meta-World`로 표기되는데 본문은 `MetaWorld`와 `Meta-World`를 섞어 쓴다. 또한 `Hit90`과 `First Hit ≥ 0.90`은 의미가 다른 지표처럼 보이나 정의가 없다.

수정안:
용어표를 정한다.
`Meta-World DrawerOpen`으로 통일.
`HER+SABER` 표기를 모든 backbone에 통일.
`Hit90`은 “number of seeds whose final or best success reaches 0.90”처럼 정의한다.

비고:
용어 혼용은 내용 오류는 아니지만 reviewer의 신뢰도를 낮춘다.

---

### 오류 21

위치: Abstract 및 본문 전반
PDF 페이지: 1–6
본문 페이지: 미표기
장/절: Abstract / Introduction / Experiments
그림/표/프로그램/수식 번호: 해당 없음

오류 유형: 편집·조판·교정 오류

심각도: 낮음

원문:
“FetchPickAndPlace 에 서 SABER 는 …”
“TD3+HER, SAC+HER 모 두 에 서 …”
“control 도 강 하 게 나 타 나 …”

문제 설명:
영문 term과 한국어 조사 사이의 spacing이 과도하게 벌어져 있다. 일부는 PDF 조판의 justification 문제로 보이나, 실제 독자에게는 띄어쓰기 오류처럼 보인다.

수정안:
Korean draft라면 다음처럼 붙여 쓴다.
“FetchPickAndPlace에서 SABER는 …”
“TD3+HER, SAC+HER 모두에서 …”
“control도 강하게 나타나 …”
LaTeX에서는 `\xspaceskip`, CJK spacing, line breaking 설정을 점검한다.

비고:
반복되는 조판 문제다.

---

### 오류 22

위치: Tables-only pages
PDF 페이지: 7–8
본문 페이지: 미표기
장/절: Tables
그림/표/프로그램/수식 번호: Table 3–5

오류 유형: 그림·표·도식 오류 / 편집·조판·교정 오류

심각도: 중간

원문:
Page 7은 Table 3 뒤에 큰 공백이 있고 하단에 Table 4가 배치된다. Page 8은 Table 5 하나만 있는 거의 빈 페이지다.

문제 설명:
표 float 배치가 비효율적이다. 학회 제출 페이지 제한이 있는 원고에서는 이 정도 공백이 내용 압축과 readability 모두에 불리하다. Table 4 caption도 page 하단에 위치해 앞뒤 흐름이 끊긴다.

수정안:
Table 3–5를 하나의 ablation/results table로 통합하거나 appendix로 이동한다. Main paper에는 핵심 table만 남기고 DrawerOpen controls는 compact table 또는 bar plot으로 정리한다.

비고:
조판 단계에서 반드시 재배치해야 한다.

## 3. 의심 항목

위치: PDF 페이지 3, 수식 (10)
의심되는 문제: `L_i = {j ∈ D : C(o_j, g^tr_i, h_j) ≈ C(o_i, g^tr_i, h_i)}`에서 모든 replay item j에 대해 query sample의 goal `g^tr_i`를 사용한다.
확인이 필요한 이유: goal-conditioned context lookup에서는 의도적으로 query goal을 기준으로 모든 state를 재평가할 수 있지만, HER relabeling과 replay buffer 저장 goal의 관계가 명시되어 있지 않다.
저자 또는 편집자가 확인해야 할 사항: `C(o_j, g^tr_i, h_j)`가 의도된 cross-goal lookup인지, 아니면 `g^tr_j`를 써야 하는지 확인.

위치: PDF 페이지 4, 수식 (20)
의심되는 문제: `Var(ϵ_i | ψ_i = 1) < Var(ϵ_i | ψ_i = 0)` 가설이 실험적으로 직접 검증되지 않았다.
확인이 필요한 이유: 논문은 theorem이 아니라고 밝히지만, SABER의 핵심 해석이 이 가설에 기대고 있다.
저자 또는 편집자가 확인해야 할 사항: support-positive/negative별 TD target variance, post-update TD-error, Q calibration error를 추가할 수 있는지 확인.

위치: PDF 페이지 8, Table 5
의심되는 문제: Random-Support가 Critic-Only보다 높은 AUC를 보인다.
확인이 필요한 이유: structured support가 핵심 mechanism인지 단순 reweighting regularization인지 불분명하다.
저자 또는 편집자가 확인해야 할 사항: random weight distribution matching, same positive-rate control, shuffled-within-context control을 추가.

위치: PDF 페이지 1, 전체 언어
의심되는 문제: AAAI-style anonymous submission인데 본문이 한국어와 영어 혼합이다.
확인이 필요한 이유: 실제 AAAI 제출본은 영어 원고가 요구될 가능성이 높다. 이 파일이 내부 Korean draft라면 문제 없지만, 제출본이라면 형식상 치명적이다.
저자 또는 편집자가 확인해야 할 사항: 제출 대상 언어 규정과 최종 영문 변환 여부 확인.

위치: PDF 페이지 6–8, 모든 표
의심되는 문제: `±`가 standard deviation인지 standard error인지 confidence interval인지 불명확하다.
확인이 필요한 이유: seeds 1–5 결과에서 통계 해석이 달라진다.
저자 또는 편집자가 확인해야 할 사항: table caption에 `mean ± std over 5 seeds` 등으로 명시.

위치: PDF 페이지 6, Table 1
의심되는 문제: `First Hit ≥ 0.90`의 계산 기준이 평균 curve인지 seed별 기준인지 불명확하다.
확인이 필요한 이유: TD3 HER row에서 Best Success와 충돌한다.
저자 또는 편집자가 확인해야 할 사항: first-hit 산출 코드와 smoothing 여부 확인.

위치: PDF 페이지 5–6, PER comparison
의심되는 문제: PER-style baseline이 저자 구현인지 표준 PER인지 불명확하다.
확인이 필요한 이유: PER와 SABER 비교의 공정성을 판단할 수 없다.
저자 또는 편집자가 확인해야 할 사항: priority formula, α/β, IS correction, replay sampling 여부 확인.

위치: PDF 페이지 9, References
의심되는 문제: DDPG와 PER가 arXiv preprint로만 제시되어 있다.
확인이 필요한 이유: DDPG와 PER는 널리 알려진 conference-paper version이 있어 bibliographic consistency 차원에서 보완 가능하다.
저자 또는 편집자가 확인해야 할 사항: venue 표기와 citation style을 AAAI 양식에 맞게 정리.

## 4. 반복 오류 패턴

특정 기호가 정의 없이 등장함: `A^sup_i`, `M_B`, `M_Q_min`, `N_gap_min`, `B3` 등.

실험 metric 정의가 반복적으로 부족함: `success AUC`, `Last-Quarter AUC`, `Hit90`, `First Hit ≥ 0.90`, `drop best-final`의 계산 방식이 불명확하다.

표와 본문 claim의 강도가 맞지 않음: n=5 결과와 큰 variance에도 “개선했다”, “호환된다”, “mechanism을 보여준다”는 표현이 상대적으로 강하다.

baseline 설명이 반복적으로 부족함: PER-style, full PER, actor-only, actor+critic, random-support, shuffled-weight의 구현 세부사항이 없다.

동일 방법 명칭이 여러 방식으로 혼용됨: `HER+SABER`, `SABER`, `TD3+HER+SABER`, `B3-Critic-Only`.

영문 term과 한국어 조사 사이 spacing 오류가 반복됨: “FetchPickAndPlace 에 서”, “control 도”, “모 두 에 서” 등.

표 float 조판이 비효율적으로 반복됨: Table 3–5가 넓은 공백을 만들며 후반 페이지의 정보 밀도가 낮다.

## 5. 수정 우선순위

순위: 1
위치: PDF 페이지 1, 3, 4
오류 요약: current policy action support를 기준으로 하지만 critic loss는 replay action에서 계산되는 핵심 mechanism 불일치.
먼저 고쳐야 하는 이유: 논문의 central claim과 technical soundness에 직접 영향을 준다.
권장 수정 방향: proxy assumption을 명확히 쓰고, support-positive/negative별 Q-error, TD-error 감소량, policy-action Q calibration 분석을 추가.

순위: 2
위치: PDF 페이지 6, Table 1
오류 요약: TD3 HER의 Best Success 0.880인데 First Hit ≥ 0.90이 980k로 표시됨.
먼저 고쳐야 하는 이유: 표 수치 자체의 신뢰성을 훼손한다.
권장 수정 방향: First Hit 정의와 산출 코드를 재검산하고 표를 수정.

순위: 3
위치: PDF 페이지 4, 수식 (14)
오류 요약: `A^sup_i`와 threshold 기호가 정의되지 않음.
먼저 고쳐야 하는 이유: SABER를 구현할 수 없는 상태다.
권장 수정 방향: support set, action bin, local pool threshold를 수식과 표로 명시.

순위: 4
위치: PDF 페이지 3–4
오류 요약: TD3/SAC target과 SABER weight 결합 수식이 없음.
먼저 고쳐야 하는 이유: 세 backbone 호환성을 contribution으로 주장하기 때문이다.
권장 수정 방향: DDPG, TD3, SAC 각각의 critic target과 loss를 별도 제시.

순위: 5
위치: PDF 페이지 5–6, Table 2
오류 요약: PER-style/full PER baseline 정의와 범위가 부족함.
먼저 고쳐야 하는 이유: “TD-error prioritization으로 설명되지 않는다”는 claim의 핵심 근거다.
권장 수정 방향: PER 수식, hyperparameter, IS correction, TD3/SAC 추가 비교를 제시.

순위: 6
위치: PDF 페이지 8, Table 5
오류 요약: Random-Support가 Critic-Only와 유사하거나 더 좋음.
먼저 고쳐야 하는 이유: support-aware mechanism claim을 약화시킨다.
권장 수정 방향: Abstract/Conclusion claim을 낮추고 positive-rate matched random control을 추가.

순위: 7
위치: PDF 페이지 4, 6–8
오류 요약: metric 정의와 보고가 불일치함.
먼저 고쳐야 하는 이유: 결과 해석과 task 간 비교가 어렵다.
권장 수정 방향: 모든 metric 정의를 명시하고 표 column을 통일.

순위: 8
위치: PDF 페이지 4–7
오류 요약: 환경 version, reward convention, hardware, steps/sec 측정 조건 누락.
먼저 고쳐야 하는 이유: 재현 가능성과 runtime claim을 해친다.
권장 수정 방향: implementation details subsection을 추가.

순위: 9
위치: PDF 페이지 7–8
오류 요약: Table 3 method 표기, Table 4/5 AUC, Table 5 B3 prefix 불일치.
먼저 고쳐야 하는 이유: 독자가 실험 조건을 잘못 이해할 수 있다.
권장 수정 방향: 표기와 수치를 한 번에 교정.

순위: 10
위치: PDF 페이지 1–8
오류 요약: 한국어·영어 혼합 조판 spacing과 용어 혼용.
먼저 고쳐야 하는 이유: 최종 제출 품질과 readability에 영향을 준다.
권장 수정 방향: 용어집을 만들고 CJK/LaTeX spacing을 재설정.

## 6. 최종 집계

전체 오류 개수: 22
치명적 오류 개수: 1
높은 심각도 오류 개수: 11
내용 오류 개수: 9
분야별 전문 오류 개수: 12
수식·숫자·단위 오류 개수: 7
코드·실습·절차 오류 개수: 5
그림·표·도식 오류 개수: 8
편집·조판·교정 오류 개수: 8
독자 관점 오류 개수: 11
의심 항목 개수: 8
반복 오류 패턴 개수: 7

## 7. 편집자용 종합 의견

이 문서에서 가장 시급하게 수정해야 할 문제는 SABER의 핵심 mechanism을 뒷받침하는 논리와 실험 근거다. 현재 논문은 “current policy action이 replay-action support 안에 있는지”를 기준으로 critic Bellman error를 reweighting하지만, 실제 loss는 replay action에서 계산된다. 이 둘 사이의 연결을 명확히 하지 않으면 논문의 중심 개념인 “hindsight Bellman target reliability gap”이 직관적 주장에 머문다.

단순 오탈자가 아니라 내용 신뢰도에 영향을 주는 문제는 Table 1의 First Hit/Best Success 불일치, undefined symbol, PER baseline의 불완전한 정의, Random-Support 결과의 mechanism 약화다. 특히 Table 1의 TD3 HER row는 즉시 재검산해야 한다.

독자나 학습자가 실제로 혼동할 가능성이 큰 부분은 `Hit90`, `First Hit`, `success AUC`, `drop best-final` 같은 metric 정의, `B3` prefix, `HER+SABER`와 `SABER` 표기 혼용이다. 이 논문은 연구논문이므로 문장보다 실험 정의와 수식의 재현 가능성이 우선이다.

조판 과정에서는 Table 3–5의 float 배치, Abstract의 한영 spacing, `MetaWorld`/`Meta-World` 표기, 영문 term과 한국어 조사 결합을 다시 확인해야 한다. 현재 후반 2페이지는 공백이 많아 학회 제출 원고로는 정보 밀도가 낮다.

다음 교정 단계의 체크리스트는 다음과 같다.
1. 수식 (14)의 모든 기호 정의.
2. TD3/SAC target 수식 추가.
3. Table 1 First Hit 재계산.
4. metric 정의 표 추가.
5. PER-style/full PER 수식과 hyperparameter 추가.
6. Random-Support 결과를 반영해 claim 약화.
7. environment version, reward convention, hardware 명시.
8. Table 3–5 표기 통일.
9. FetchPush/appendix 언급 삭제 또는 appendix 추가.
10. Related Work에 sparse GCRL 최신 baseline 추가.

---

# Peer Review

## Stage 1: Internal Compression

### Core Claims (H_core)

논문은 HER가 relabeled Bellman target의 quantity는 늘리지만, 그 target이 현재 policy action 근방에서 reliable한지는 보장하지 않는다고 주장한다. SABER는 local state-goal-horizon context에서 current policy action이 replay-action support를 갖는지를 판정하고, support-positive sample의 critic Bellman error에 더 큰 weight를 주는 critic-only reweighting objective다. 주요 empirical claim은 FetchPickAndPlace에서 DDPG/TD3/SAC 모두 final success와 success AUC가 증가하고, FetchSlide와 DrawerOpen에서는 장점과 한계가 함께 나타난다는 것이다.

### Method (M)

SABER는 mini-batch sample마다 `c_i = C(o_i, g_i^tr, h_i)`를 만들고, replay buffer에서 유사 context pool `L_i`를 찾은 뒤, action bins `B_i` 안에 current policy action bin `b_i^π`가 있는지 확인한다. `ψ_i = 1`이면 raw weight `1 + λ_S`, 아니면 `1`을 주고 batch-normalized weight `Ω_i`를 critic Bellman loss에만 곱한다. Actor objective와 replay sampling distribution은 baseline과 동일하게 유지한다.

### Evidence (E)

FPP Table 1은 DDPG, TD3, SAC에서 HER+SABER의 mean final success와 AUC가 HER보다 높다고 보고한다. Table 2는 DDPG FPP에서 PER-style과 full PER보다 SABER AUC가 높다고 보고한다. FetchSlide Table 3은 TD3/SAC에서는 개선, DDPG에서는 악화를 보인다. DrawerOpen Table 4–5는 SABER가 TD3+HER보다 Hit90 seed count를 높이지만 Random-Support도 강하다는 limitation을 보여준다.

## Stage 2: External Context Construction

Sub-domain: sparse-reward goal-conditioned RL, hindsight relabeling, off-policy actor-critic, robotic manipulation, support-aware value learning.

Relevant prior work: HER는 sparse binary rewards에서 실패 trajectory를 achieved goal 기준으로 relabeling하는 대표 방법이다. DDPG, TD3, SAC는 continuous-control off-policy actor-critic의 핵심 baseline이며, TD3는 function approximation error와 overestimation을 줄이기 위해 twin critics와 delayed update를 사용하고, SAC는 maximum-entropy off-policy actor-critic이다. HER 계열 직접 관련 work로는 Curriculum-guided HER, Energy-Based Hindsight Experience Prioritization, Hindsight Goal Ranking, MHER/MHER(λ) 등이 있다. Offline RL의 support/extrapolation 문제는 BCQ와 CQL 계열에서 핵심적으로 다뤄졌다. Meta-World는 50개 robotic manipulation tasks로 구성된 benchmark다.

Current SOTA context: sparse-reward goal-conditioned Robotics 계열에서는 high replay ratio와 regularization을 HER와 결합한 REDQ-style 접근이 2024년 기준 강한 관련 baseline으로 보고되며, 해당 논문은 12개 sparse-reward goal-conditioned Robotics tasks에서 이전 SOTA 대비 약 2배 sample efficiency를 주장한다. 따라서 SABER의 현재 baseline set은 DDPG/TD3/SAC+HER만으로는 충분히 강하다고 보기 어렵다.

Domain narrative:
이 분야는 HER의 relabeling으로 sparse reward 문제를 완화하는 단계에서, 어떤 episode/goal/transition을 더 유용하게 재사용할지 선택하는 prioritization·curriculum 방향으로 발전해 왔다. 동시에 offline RL에서는 dataset support 밖 action의 Q extrapolation 문제가 별도 축으로 발전했다. SABER는 이 두 흐름 사이에 위치한다. HER처럼 goal을 새로 만들지는 않고, PER처럼 TD-error로 sampling하지도 않으며, offline RL처럼 policy를 constrain하지도 않는다. contribution은 “support-aware critic loss weighting”이라는 좁은 objective에 있다. 다만 현재 논문은 direct sparse GCRL SOTA와의 비교가 부족하고, Random-Support 결과 때문에 support-specific novelty가 완전히 분리되지 않는다.

## Stage 3: Adversarial Baseline & Evaluation Check

Task: FetchPickAndPlace, FetchSlide, MetaWorld DrawerOpen sparse goal-conditioned manipulation.

Dataset/Environment: online off-policy replay buffer 기반. 그러나 Gym/Gymnasium-Robotics version, Meta-World version, reward convention, success threshold, evaluation episodes가 누락되어 있다.

Metrics: final success, best success, success AUC, first hit, last-quarter AUC, Hit90, steps/sec. 그러나 AUC 계산법, Hit90 정의, `±` 의미, smoothing 여부가 누락되어 있다.

Missing baselines: REDQ+HER/high replay ratio, CHER, energy-based HER prioritization, HGR, MHER/MHER(λ), possibly JaxGCRL-style modern GCRL baselines. PER-style baseline도 DDPG FPP에만 있어 충분하지 않다.

Suspicious evaluation setups: Table 1의 TD3 HER row에서 Best Success 0.880인데 First Hit ≥0.90이 존재하는 것은 metric 정의 또는 수치 오류다. Table 4와 Table 5의 동일해 보이는 SABER AUC도 0.362와 0.361로 다르다.

## Stage 4: Multi-Aspect Skeptical Q&A

### Question 1: SABER의 novelty가 기존 HER prioritization과 충분히 다른가?

Paper-based answer: SABER는 TD-error나 goal relabeling distribution이 아니라 local replay-action support를 사용하고, replay sampling은 바꾸지 않는다고 주장한다.
Verification: 기존 CHER, energy-based prioritization, HGR 등은 “어떤 hindsight experience를 더 재사용할지”를 다뤘고, SABER는 critic loss weighting이라는 차이가 있다. 그러나 논문은 이들 직접 관련 HER prioritization 계열을 충분히 비교하지 않는다.
Verdict: Weak.

### Question 2: support-positive predicate가 critic Bellman target reliability를 제대로 측정하는가?

Paper-based answer: current policy action이 local replay support 안에 있으면 extrapolation risk가 낮다고 본다.
Verification: 실제 critic loss는 replay action `a_i`에서 계산된다. policy action support와 replay-action Bellman error reliability 사이의 연결은 실험적으로 검증되지 않았다.
Verdict: Weak.

### Question 3: 수식 정의는 재현 가능한가?

Paper-based answer: `L_i`, `B_i`, `b_i^π`, `ψ_i`, `Ω_i`를 제시한다.
Verification: `A^sup_i`, binning rule, threshold mapping, context feature discretization이 빠져 있어 구현이 어렵다.
Verdict: Incorrect/Incomplete.

### Question 4: 세 backbone 호환성 claim은 충분히 뒷받침되는가?

Paper-based answer: DDPG, TD3, SAC 모두에서 FPP 성능 향상을 보인다고 한다.
Verification: FPP 결과는 평균상 유리하지만 TD3/SAC target 수식과 구현 세부가 없다. FetchSlide에서는 DDPG가 악화된다.
Verdict: Weak-to-Moderate.

### Question 5: PER-style baseline으로 RQ3가 검증되는가?

Paper-based answer: DDPG FPP에서 SABER가 PER-style/full PER보다 AUC가 높다.
Verification: PER-style 정의가 없고 DDPG FPP 하나뿐이다. TD3/SAC 및 FetchSlide에서 검증되지 않았다.
Verdict: Weak.

### Question 6: 결과가 통계적으로 설득력 있는가?

Paper-based answer: seeds 1–5의 mean ± 값을 제시한다.
Verification: n=5, significance test 없음, per-seed curve 없음, 일부 결과 variance가 크다. DDPG FPP final gain은 작다.
Verdict: Weak.

### Question 7: Random-Support control은 SABER의 mechanism을 지지하는가?

Paper-based answer: 저자는 Random-Support가 강하므로 limitation이라고 인정한다.
Verification: Table 5에서 Random-Support AUC가 Critic-Only보다 높다. 이는 structured support predicate가 핵심 원인이라는 claim을 약화한다.
Verdict: Weak.

### Question 8: runtime cost를 공정하게 보고했는가?

Paper-based answer: steps/sec를 보고하고 wall-clock 효율을 항상 주장하지 않는다고 말한다.
Verification: hardware와 measurement protocol이 없어 재현성이 낮다. FetchSlide slowdown은 매우 크다.
Verdict: Weak.

### Question 9: 일반화 가능성은 충분한가?

Paper-based answer: FPP, FetchSlide, DrawerOpen을 사용한다.
Verification: DrawerOpen은 variance가 크고, FetchSlide DDPG는 악화된다. FetchPush는 언급만 있고 결과가 없다.
Verdict: Weak-to-Moderate.

### Question 10: 논문 claim의 적정 강도는 어떤가?

Paper-based answer: SABER는 “general-purpose RL algorithm”이 아니라 좁은 critic-side objective라고 낮춰 말한다.
Verification: 이 좁은 claim은 적절하지만 Abstract의 support-aware mechanism 표현은 Random-Support 결과에 비해 다소 강하다.
Verdict: Moderate.

## Stage 5: Evidence-Based Review Synthesis

### Summary

이 논문은 sparse-reward goal-conditioned HER setting에서 relabeled Bellman targets의 quantity와 reliability를 구분해야 한다는 문제의식에서 SABER를 제안한다. SABER는 local replay-action support를 사용해 critic Bellman error를 sample-wise reweighting하는 critic-only objective다. FPP에서는 세 backbone 모두에서 평균 성능 향상이 보고되지만, FetchSlide와 DrawerOpen은 task/backbone dependence와 mechanism ambiguity를 보여준다.

### Strengths

문제 framing은 흥미롭다. HER가 relabeling quantity를 늘리는 반면, critic이 어떤 relabeled target을 더 신뢰해야 하는지 묻는 관점은 명확하고 좁다. Critic-only design은 actor distribution bias와 sampling bias를 분리하려는 좋은 설계다. 저자가 FetchSlide DDPG 실패와 Random-Support limitation을 숨기지 않고 논문에 포함한 점은 신뢰도를 높인다.

### Weaknesses

Major weakness는 support predicate와 critic loss 사이의 technical gap이다. current policy action support를 측정하지만 실제 regression은 replay action에서 수행된다. 두 번째 major weakness는 baseline insufficiency다. REDQ+HER/high replay ratio, CHER, MHER, HGR, energy-based HER prioritization 등 직접 관련 baseline이 없다. 세 번째 major weakness는 metric/reporting 문제다. Table 1의 First Hit inconsistency, undefined AUC/Hit90, `±` 의미 누락, n=5 통계 검정 부재가 있다. 네 번째 major weakness는 Random-Support가 Critic-Only와 비슷하거나 더 좋은 결과를 보인다는 점이다.

### Novelty & Significance

Novelty는 “goal relabeling이나 replay sampling이 아니라 critic-side support-aware loss reweighting”이라는 좁은 지점에 있다. 이 차별점은 의미가 있지만, structured support가 실제 성능 원인인지 random reweighting regularization인지 아직 충분히 분리되지 않았다.

### Technical Soundness

현재 수식과 알고리즘 정의는 불완전하다. `A^sup_i`, binning, threshold, context hash, TD3/SAC target, PER-style baseline이 명확하지 않다. 실험 결과는 가능성을 보여주지만 top-tier acceptance 수준의 mechanism validation에는 부족하다.

### Suggestions

SABER를 강화하려면 우선 support predicate와 critic reliability의 연결을 empirical diagnostic으로 증명해야 한다. support-positive vs negative의 TD-error 감소, Q calibration, policy-action neighborhood error, ψ와 |δ| correlation/top-k overlap을 추가하라. Baseline은 REDQ+HER/high replay ratio와 HER prioritization 계열을 포함하라. 모든 metric 정의, environment version, reward convention, hardware, seeds, statistical test를 명시하라. Abstract와 Conclusion은 “support-aware mechanism이 완전히 입증되었다”가 아니라 “support-aware weighting and possibly reweighting-induced regularization”으로 claim을 낮추는 편이 안전하다.

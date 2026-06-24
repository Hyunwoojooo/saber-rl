# 오류 검수 결과

## 1. 전체 검토 범위

문서 유형: 학회 제출용 연구논문 초안 / 리뷰용 PDF 빌드
추정 분야: 컴퓨터·AI, 강화학습, sparse-reward goal-conditioned reinforcement learning, HER, off-policy actor-critic, robotic manipulation
총 검토 페이지: 최신 기준 원고 `saber_aaai27_draft_kr_review.pdf` PDF 12페이지 전체. 보조자료로 `AAAI27_RESULT_MATRIX_KR.md`, `AAAI27_GPT_REVIEW_PACKET_KR.md`, `AAAI27_CLAIM_REVISION_KR.md`를 대조함. 리뷰 패킷은 최신 기준 원고가 `.tex`와 그 기반 PDF라고 명시한다. fileciteturn5file1
주요 검토 기준: RL 논문 기술 정확성, Bellman target·critic loss·support predicate 수식 정합성, HER/PER/TD3/SAC baseline 공정성, 실험 metric 정의, 수치·표·본문 일관성, 재현 가능성, 관련연구 누락, anonymous submission 조판 품질
검토 시 특히 주의한 항목: current policy action support와 replay-action critic residual의 연결, FPP/FetchSlide/DrawerOpen 결과 해석, Random-support control의 claim 약화 효과, runtime overhead와 steps/sec 해석, appendix·artifact matrix와 본문 간 불일치, 표·긴 파일명·한영 혼합 조판 상태

## 2. 오류 목록

### 오류 1

위치: Theoretical Interpretation / Motivation
PDF 페이지: 4, 8
본문 페이지: 4, 8
장/절: Motivation: Target Quantity Is Not Target Reliability / Theoretical Interpretation
그림/표/프로그램/수식 번호: 수식 (28)

오류 유형: 내용 오류 / 분야별 전문 오류 / 독자 관점 오류

심각도: 높음

원문:
“이 가설은 theorem으로 주장하지 않는다. 본 논문에서는 이를 reliability-weighted Bellman regression interpretation으로 제시하고, 실험적으로 검증한다.”
“E[R_i^fit | ψ_i = 1] < E[R_i^fit | ψ_i = 0].”
“This is not a theorem in the current paper. It is an interpretation and empirical hypothesis.” fileciteturn4file13 fileciteturn5file8

문제 설명:
본문은 support-positive sample이 더 낮은 effective fitting risk를 갖는다는 가설을 제시하지만, 현재 실험은 주로 최종 success/AUC와 일부 ablation에 머문다. sample-level diagnostic, Q calibration, ψ와 TD-error의 correlation, support-positive/negative별 update 전후 TD-error 감소량은 아직 없으며, 원고도 이 분석이 필요하다고 말한다. 따라서 “실험적으로 검증한다”는 표현은 현재 증거보다 강하다.

수정안:
“본 논문에서는 이를 reliability-weighted Bellman regression interpretation으로 제시하고, downstream performance와 ablation으로 간접 평가한다. ψ_i가 실제 fitting risk를 낮추는지에 대한 sample-level diagnostic은 최종본에서 추가 검증한다.”

비고:
논문의 핵심 mechanism claim에 해당한다. reject risk가 높은 항목이다.

---

### 오류 2

위치: Support Indicator
PDF 페이지: 6
본문 페이지: 6
장/절: SABER / Support Indicator
그림/표/프로그램/수식 번호: 수식 (20)

오류 유형: 수식·숫자·단위 오류 / 분야별 전문 오류 / 코드·실습·절차 오류

심각도: 높음

원문:
“m_i = min(|L_i|, M_B), ψ_i = 1[|L_i| ≥ N_local ∧ m_i ≥ M_Q ∧ b_i^π ∈ B_i].”
“여기서 M_B는 support candidate cap, M_Q는 minimum available support count, N_local은 minimum exact local pool size다.” fileciteturn3file2

문제 설명:
현재 설정값은 `M_Q = 4`, `N_local = 4`이다. 이 경우 `m_i = min(|L_i|, 16) ≥ 4`는 사실상 `|L_i| ≥ 4`와 거의 동일해 `|L_i| ≥ N_local` 조건과 중복된다. 또한 `B_i`는 action bin set이므로 support diversity를 보려면 `|B_i|` 또는 unique action bin count를 써야 한다. 현재 수식은 “available support count”가 local transition 수인지, unique action bin 수인지 불분명하다.

수정안:
의도가 local transition 수 확인이라면 중복 조건을 제거한다.
`ψ_i = 1[|L_i| ≥ N_local ∧ b_i^π ∈ B_i]`

의도가 action support diversity 확인이라면 다음처럼 수정한다.
`ψ_i = 1[|L_i| ≥ N_local ∧ |unique(B_i)| ≥ M_Q ∧ b_i^π ∈ B_i]`

비고:
SABER 구현과 논문 method 정의의 핵심 수식이다. 현재 상태로는 reviewer가 `M_Q`의 의미를 의심할 수 있다.

---

### 오류 3

위치: Local Context / FetchPickAndPlace phase definition
PDF 페이지: 5
본문 페이지: 5
장/절: SABER / Local Context
그림/표/프로그램/수식 번호: 수식 (14)

오류 유형: 분야별 전문 오류 / 코드·실습·절차 오류 / 독자 관점 오류

심각도: 중간

원문:
“Phase는 d_hand,obj > d_reach이면 approach, object height가 z_lift 이하이고 gripper gap이 threshold보다 크면 grasp, object height가 z_lift 이하이면 lift, 나머지는 place로 둔다.” fileciteturn4file5

문제 설명:
`d_reach`, `z_lift`, `threshold`의 수치가 제시되지 않는다. 또한 pick-and-place phase 명명 관점에서 `object height ≤ z_lift`일 때 `lift`, 그 외를 `place`로 두는 로직은 독자가 직관적으로 받아들이기 어렵다. 보통 object가 아직 충분히 lift되지 않았을 때 lift phase라고 볼 수는 있지만, `place`가 “lift 이후”인지 “goal placement phase”인지 설명이 필요하다. gripper gap이 threshold보다 클 때 `grasp`로 둔 것도 open gripper 상태와 혼동될 수 있다.

수정안:
다음처럼 phase predicate를 표로 명시한다.

`phase = approach if d_hand,obj > d_reach`
`phase = grasp_attempt if d_hand,obj ≤ d_reach and z_obj ≤ z_lift and gripper_gap > g_open`
`phase = lift if d_hand,obj ≤ d_reach and z_obj ≤ z_lift and gripper_gap ≤ g_open`
`phase = place if z_obj > z_lift`

그리고 `d_reach`, `z_lift`, `g_open` 값을 명시한다.

비고:
이 항목은 성능 수치보다 재현성과 method clarity에 영향을 준다.

---

### 오류 4

위치: Local Context / Tasks and Metrics / Reproducibility
PDF 페이지: 5, 8–10
본문 페이지: 5, 8–10
장/절: SABER / Experiments
그림/표/프로그램/수식 번호: 수식 (13)–(16), Table 3–5

오류 유형: 분야별 전문 오류 / 코드·실습·절차 오류 / 독자 관점 오류

심각도: 높음

원문:
“FetchPickAndPlace main experiments에서 사용한 exact context는 다음과 같다.”
“FetchSlide는 secondary evidence로 사용한다.”
“Meta-World DrawerOpen은 external stress-test와 actor/random/shuffled ablation 분석에 사용한다.” fileciteturn4file5 fileciteturn5file10

문제 설명:
FPP context는 비교적 상세히 정의되지만, FetchSlide와 DrawerOpen에서 어떤 local context, bin boundary, phase predicate, action binning을 썼는지는 본문에 없다. Table 3–5는 FetchSlide와 DrawerOpen 결과를 핵심 limitation/stress-test로 사용하므로, 해당 task들의 context 설계도 최소한 compact table로 본문이나 appendix에 있어야 한다.

수정안:
본문 또는 appendix에 다음 표를 추가한다.

`Task / Context features / Phase predicate / Bin boundaries / Action bin K / N_scan / N_local / M_Q / λ_S / support_scan_mode`

비고:
support predicate가 task-specific context에 의존한다고 스스로 인정하므로, task별 context 정의 누락은 재현 가능성 측면에서 높음 심각도다.

---

### 오류 5

위치: Reproducibility Snapshot 직후
PDF 페이지: 9
본문 페이지: 9
장/절: Experiments / Reproducibility Snapshot
그림/표/프로그램/수식 번호: 해당 없음

오류 유형: 편집·조판·교정 오류 / 코드·실습·절차 오류 / 독자 관점 오류

심각도: 중간

원문:
“A full Korean reproducibility appendix with task-specific context definitions is maintained in AAAI27_REPRODUCIBILITY_APPENDIX_KR.md.” fileciteturn5file3

문제 설명:
PDF 렌더링상 긴 파일명이 오른쪽 여백에서 잘리거나 과밀하게 보인다. 또한 학회 제출 원고에서 로컬 Markdown 파일명을 본문에 그대로 두면 독자가 appendix를 실제로 찾을 수 없다. 현재 업로드 묶음에도 해당 appendix 파일은 제공되지 않았고, review packet은 보조 문서 목록으로만 언급한다. fileciteturn5file0

수정안:
본문에서는 파일명을 삭제하고 appendix 번호로 바꾼다.
“Full task-specific context definitions are provided in Appendix A.”
PDF 조판에서는 `\path{}` 또는 `\url{}`을 쓰되 줄바꿈 가능하게 처리한다.

비고:
조판 오류와 재현성 참조 오류가 동시에 발생한다.

---

### 오류 6

위치: Reproducibility Snapshot / Experiments
PDF 페이지: 8–9
본문 페이지: 8–9
장/절: Experiments
그림/표/프로그램/수식 번호: Table 1–5

오류 유형: 분야별 전문 오류 / 코드·실습·절차 오류 / 독자 관점 오류

심각도: 높음

원문:
“모든 main Fetch 실험은 uniform replay와 future-goal HER를 사용한다. HER ratio는 0.8이며…”
“Success state는 Bellman target에서 terminal로 강제하지 않고, timeout transition에는 bootstrapping을 허용한다.” fileciteturn5file10

문제 설명:
HER ratio와 terminal convention은 제시되었지만, Fetch/Gym/Gymnasium-Robotics version, Meta-World version, reward convention, success threshold, evaluation environment name, sparse/dense reward variant가 명시되지 않았다. Fetch 계열 실험은 환경 버전과 success threshold에 따라 성공률 해석이 달라질 수 있다.

수정안:
다음 정보를 추가한다.
`Gymnasium-Robotics version`, `FetchPickAndPlace-v?`, `FetchSlide-v?`, `Meta-World version`, `success threshold ε`, `reward = 0/-1 또는 1/0`, `max episode length`, `evaluation episodes`, `seed mapping`.

비고:
현재 실험 표가 논문의 주된 증거이므로 환경 convention 누락은 높은 심각도다.

---

### 오류 7

위치: Tasks and Metrics / Runtime Cost
PDF 페이지: 8–11
본문 페이지: 8–11
장/절: Experiments / Runtime Cost
그림/표/프로그램/수식 번호: Table 1–3

오류 유형: 수식·숫자·단위 오류 / 분야별 전문 오류 / 독자 관점 오류

심각도: 높음

원문:
“Steps/sec는 training wall-clock throughput이며 support lookup overhead를 포함한다.”
“SABER는 local support lookup 때문에 runtime overhead가 있다.” fileciteturn5file10 fileciteturn4file7

문제 설명:
steps/sec는 hardware, CPU/GPU, simulator parallelism, Python implementation, replay indexing, support lookup cache, evaluation 포함 여부에 크게 좌우된다. 그러나 hardware/software 측정 조건이 없다. 특히 FetchSlide DDPG는 119.08 steps/sec에서 26.85 steps/sec로 크게 감소하므로 runtime claim을 하려면 측정 조건이 필수다.

수정안:
“Steps/sec was measured on [CPU/GPU], [number of env workers], [MuJoCo version], [Python/PyTorch version], excluding/including evaluation, using deterministic-tail support lookup without/with caching.”
또한 Table 1–3 caption에 “training-only throughput”인지 “training+evaluation throughput”인지 명시한다.

비고:
sample efficiency와 wall-clock efficiency를 분리하겠다는 논문 의도는 좋지만, 측정 조건이 없으면 수치 신뢰도가 낮아진다.

---

### 오류 8

위치: Main Result: FetchPickAndPlace
PDF 페이지: 9
본문 페이지: 9
장/절: Experiments / Main Result
그림/표/프로그램/수식 번호: Table 1

오류 유형: 내용 오류 / 수식·숫자·단위 오류 / 독자 관점 오류

심각도: 높음

원문:
“FPP에서는 SABER가 세 backbone 모두에서 final success와 success AUC를 개선한다.”
Table 1: DDPG HER `0.940 ± 0.042`, DDPG HER+SABER `0.960 ± 0.042`; TD3와 SAC도 seeds 1–5 기준으로 보고됨. fileciteturn4file1 fileciteturn4file2

문제 설명:
n=5 seed 결과에 대해 standard deviation만 있고 paired test, bootstrap confidence interval, per-seed curve가 없다. DDPG final success 차이 0.02는 sd와 비교해 작다. “개선한다”는 문장은 평균값 기준으로는 맞지만, 통계적 강건성을 암시할 수 있다.

수정안:
“FPP에서는 SABER가 세 backbone 모두에서 mean final success와 mean success AUC를 높인다.”
추가로 paired bootstrap CI, Wilcoxon signed-rank 또는 seed-paired difference table을 제시한다.

비고:
top-tier ML 리뷰에서는 “only 5 seeds, no significance”가 반복적으로 지적된다.

---

### 오류 9

위치: PER-style comparison
PDF 페이지: 7, 9
본문 페이지: 7, 9
장/절: TD-error and PER-style Loss Weighting Baselines / Comparison with PER-Style Weighting
그림/표/프로그램/수식 번호: 수식 (26), (27), Table 2

오류 유형: 내용 오류 / 분야별 전문 오류 / 코드·실습·절차 오류

심각도: 높음

원문:
“따라서 현재 표는 SABER의 이득이 PER-style weighting이나 prioritized replay만으로 설명되기 어렵다는 제한적인 근거로 사용한다.” fileciteturn4file2

문제 설명:
PER-style/full PER 비교는 FPP DDPG 하나에만 있다. Result matrix도 FPP TD-error는 약 630k partial run으로 1M main row에 쓰면 안 된다고 명시한다. fileciteturn3file5 따라서 RQ3를 충분히 닫기에는 부족하다. “제한적인 근거”라고 낮춘 것은 좋지만, RQ3의 제목과 contribution에서는 아직 PER/TD-error와의 차이를 비교·분석했다고 읽힌다.

수정안:
RQ3 답변을 더 좁힌다.
“On FPP-DDPG only, SABER outperforms our PER-style and full-PER baselines; broader backbone/task coverage and sample-level overlap diagnostics remain open.”
가능하면 TD3 또는 SAC FPP에도 PER-style baseline을 추가한다.

비고:
현재 상태에서는 “missing baseline/diagnostic”로 분류된다.

---

### 오류 10

위치: PER-style loss formula
PDF 페이지: 7
본문 페이지: 7
장/절: TD-error and PER-style Loss Weighting Baselines
그림/표/프로그램/수식 번호: 수식 (26), (27)

오류 유형: 분야별 전문 오류 / 코드·실습·절차 오류 / 수식·숫자·단위 오류

심각도: 중간

원문:
“δ_i = |Q_ϕ(x_i, a_i^r) − y_i|.”
“p_i = (δ_i + ε_PER)^α_PER.” fileciteturn4file9

문제 설명:
PER-style loss weighting에서 δ_i를 현재 critic으로 계산하면, 구현상 weight에 gradient가 흐를 수 있다. PER priority나 loss weight는 보통 TD-error를 priority signal로 사용하되, critic loss gradient에서는 detach/stop-gradient 처리한다. 원고에는 `stopgrad(δ_i)` 또는 priority update timing이 없다.

수정안:
`δ_i = stopgrad(|Q_ϕ(x_i, a_i^r) − y_i|)`
또는 “All TD-error based weights are computed under no_grad and are not differentiated through.”라고 명시한다.

비고:
코드 재현성과 baseline 공정성에 직접 영향을 준다.

---

### 오류 11

위치: PER-style comparison
PDF 페이지: 9
본문 페이지: 9
장/절: Comparison with PER-Style Weighting
그림/표/프로그램/수식 번호: Table 2

오류 유형: 내용 오류 / 수식·숫자·단위 오류 / 독자 관점 오류

심각도: 중간

원문:
“Full PER는 한 seed collapse 때문에 평균 final success가 크게 낮아진다.” fileciteturn4file2

문제 설명:
본문에는 per-seed 결과나 collapse를 보여주는 curve가 없다. 평균과 표준편차만으로 한 seed collapse를 추론할 수는 있지만, 저자가 그렇게 단정하려면 per-seed evidence가 필요하다.

수정안:
“Full PER shows high variance and a low mean final success; per-seed curves suggest one collapsed run.”
또는 appendix에 per-seed table/curve를 추가한다.

비고:
원인 설명을 단정하지 말고 evidence 수준에 맞춰 낮춰야 한다.

---

### 오류 12

위치: Actor-Scope and Random Controls / Limitations
PDF 페이지: 10–11
본문 페이지: 10–11
장/절: Actor-Scope and Random Controls / Limitations
그림/표/프로그램/수식 번호: Table 5

오류 유형: 내용 오류 / 분야별 전문 오류 / 독자 관점 오류

심각도: 높음

원문:
“Critic-only는 actor update를 바꾸지 않기 때문에, 논문의 main mechanism인 Bellman fitting reliability를 가장 깨끗하게 보여준다.”
Table 5: Critic-only SABER AUC `0.362 ± 0.338`, Actor+critic weighting AUC `0.432 ± 0.395`, Random-support AUC `0.426 ± 0.392`. fileciteturn5file12

문제 설명:
Table 5에서는 Actor+critic과 Random-support가 Critic-only보다 AUC 평균이 높다. 저자가 “clean mechanism”을 의미론적으로 쓰는 것은 가능하지만, 표 수치만 보면 critic-only가 performance dominance를 보이지 않는다. “가장 깨끗하게 보여준다”는 표현은 mechanism isolation과 performance를 혼동하게 만들 수 있다.

수정안:
“Critic-only is the most isolating design for attributing the intervention to critic regression, although it is not the best-performing variant in DrawerOpen.”
한국어로는:
“Critic-only는 성능 우위가 아니라 mechanism 분리를 위한 main variant다.”

비고:
Random-support가 강한 현재 결과에서는 claim을 더 낮춰야 한다.

---

### 오류 13

위치: Related Work
PDF 페이지: 2–3, 12
본문 페이지: 2–3, 12
장/절: Related Work / References
그림/표/프로그램/수식 번호: References

오류 유형: 분야별 전문 오류 / 내용 오류 / 독자 관점 오류

심각도: 높음

원문:
Related Work는 HER, DDPG/TD3/SAC, PER, BCQ/CQL, MetaCog-RL 중심으로 구성되어 있다. References도 8개 항목만 포함한다. fileciteturn4file6 fileciteturn5file9

문제 설명:
sparse goal-conditioned RL/HER 개선 논문에서 직접 비교해야 하는 HER prioritization·goal selection·curriculum 계열이 거의 빠져 있다. 예를 들어 CHER는 실패 경험 선택을 goal proximity와 curiosity 기준으로 조절하고, energy-based hindsight prioritization은 robotic manipulation에서 hindsight experience를 우선순위화하며, HGG는 hindsight goal을 생성하는 curriculum 계열이다. MHER는 dynamics model을 사용해 virtual achieved goal을 생성하는 model-based HER 변형이다. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2019/hash/83715fd4755b33f9c3958e1a9ee221e1-Abstract.html?utm_source=chatgpt.com))

수정안:
Related Work에 다음 하위 절을 추가한다.
`Hindsight Goal Selection and Prioritization`
포함 대상: CHER, Energy-Based Hindsight Experience Prioritization, HGG, HGR, MHER/MHER(λ), high replay ratio HER/REDQ+HER.

비고:
novelty risk가 큰 항목이다.

---

### 오류 14

위치: Experiments / Baselines
PDF 페이지: 8–10
본문 페이지: 8–10
장/절: Experiments
그림/표/프로그램/수식 번호: Table 1–3

오류 유형: 분야별 전문 오류 / 내용 오류 / 독자 관점 오류

심각도: 높음

원문:
Main baseline은 DDPG+HER, TD3+HER, SAC+HER이며, FPP DDPG에서 PER-style/full PER를 추가 비교한다. fileciteturn4file1

문제 설명:
현재 baseline set은 classic off-policy backbone 중심이다. 최근 sparse-reward goal-conditioned RL에서는 high replay ratio와 regularization을 HER와 결합한 REDQ-style 접근이 강한 비교 대상으로 보고되어 있다. 해당 OpenReview 논문은 high replay ratio와 regularization을 sparse-reward goal-conditioned tasks로 확장하는 방향을 다룬다. ([openreview.net](https://openreview.net/forum?id=7uml1PLgAN&utm_source=chatgpt.com)) 따라서 SABER가 sample efficiency를 주장하려면 REDQ+HER/high replay ratio baseline이 매우 중요하다.

수정안:
FPP 최소 1개 task에서 다음 baseline을 추가한다.
`REDQ+HER / high replay ratio HER`, `CHER`, `Energy-Based Prioritization`, 가능하면 `HGG` 또는 `MHER`.

비고:
AAAI/NeurIPS/ICLR reviewer가 “weak baselines”로 지적할 가능성이 높다.

---

### 오류 15

위치: MetaCog-RL Lineage
PDF 페이지: 3
본문 페이지: 3
장/절: Related Work / MetaCog-RL Lineage
그림/표/프로그램/수식 번호: 해당 없음

오류 유형: 내용 오류 / 편집·조판·교정 오류 / 독자 관점 오류

심각도: 중간

원문:
“SABER는 MetaCog-RL의 full diagnosis framework를 구현한 것이 아니다. 원래 MetaCog-RL은 RL 실패 원인을 ignorance, incompetence, impossibility로 나누고…” fileciteturn4file12

문제 설명:
MetaCog-RL이 공개된 선행연구라면 citation이 필요하다. 공개 선행연구가 아니라 내부 framework라면 anonymous submission에서 고유명으로 길게 설명하는 것이 불필요하거나 익명성 리스크가 될 수 있다.

수정안:
공개 논문이면 References에 추가한다.
비공개 내부 framework면 다음처럼 줄인다.
“This work is motivated by a narrower diagnostic view of practice-oriented RL failures, but we do not rely on the broader framework.”

비고:
현재 Related Work의 citation balance를 해친다.

---

### 오류 16

위치: Tasks and Metrics
PDF 페이지: 8
본문 페이지: 8
장/절: Experiments / Tasks and Metrics
그림/표/프로그램/수식 번호: 해당 없음

오류 유형: 편집·조판·교정 오류 / 독자 관점 오류

심각도: 중간

원문:
“FetchPush는 complete rows만 appendix 또는 보조 evidence로 제한한다.” fileciteturn5file10

문제 설명:
현재 검토 PDF에는 FetchPush appendix가 포함되어 있지 않다. Result matrix에는 FetchPush 일부 결과가 appendix evidence로 가능하다고 되어 있지만, TD3 HER baseline이 없는 등 full table로 쓰기 어렵다고 정리되어 있다. fileciteturn5file16 본문만 읽는 reviewer는 존재하지 않는 appendix를 기대하게 된다.

수정안:
appendix를 실제로 제출하지 않을 경우 문장을 삭제한다.
appendix를 제출할 경우 “FetchPush results are reported in Appendix B as partial supporting evidence, not as a complete cross-backbone comparison.”으로 제한한다.

비고:
편집 흔적성 문장이다.

---

### 오류 17

위치: Research Questions
PDF 페이지: 8
본문 페이지: 8
장/절: Experiments / Research Questions
그림/표/프로그램/수식 번호: RQ4

오류 유형: 내용 오류 / 독자 관점 오류

심각도: 중간

원문:
“RQ4. Critic-only weighting이 actor-only 또는 actor+critic weighting보다 더 깨끗한 mechanism을 제공하는가?” fileciteturn5file8

문제 설명:
“더 깨끗한 mechanism”은 측정 가능한 metric이 아니다. RQ라면 어떤 근거로 clean하다고 판정할지 명시해야 한다. 예를 들어 actor objective unchanged, sampling unchanged, positive-rate matched controls, variance, AUC, failure mode 중 무엇을 기준으로 하는지 불명확하다.

수정안:
“RQ4. Does critic-only weighting isolate the effect of support-derived weights on critic regression better than actor-only or actor+critic weighting, as measured by unchanged actor objective, lower variance, and control comparisons?”

비고:
research question 자체의 operational definition 문제다.

---

### 오류 18

위치: Tasks and Metrics
PDF 페이지: 8
본문 페이지: 8
장/절: Experiments / Tasks and Metrics
그림/표/프로그램/수식 번호: Table 1–5

오류 유형: 수식·숫자·단위 오류 / 독자 관점 오류

심각도: 중간

원문:
“Success AUC는 evaluation success curve의 normalized trapezoidal area이며, 현재 표에서는 task 내부 비교에만 사용한다.” fileciteturn5file8

문제 설명:
normalized trapezoidal area라고 했지만 수식이 없다. 시작 checkpoint를 0으로 보간하는지, first evaluation부터 적분하는지, evaluation interval이 균등한지, denominator가 `T_end − T_start`인지 `1M`인지가 불명확하다.

수정안:
다음 수식을 추가한다.
`AUC = (1/T) Σ_k 0.5(s_k + s_{k+1})(t_{k+1} − t_k)`
`T`가 1M인지 evaluation span인지 명시한다.

비고:
수치 재현성 문제다.

---

### 오류 19

위치: Bellman Regression
PDF 페이지: 4
본문 페이지: 4
장/절: Preliminaries / Bellman Regression
그림/표/프로그램/수식 번호: 수식 (7), (10), (12)

오류 유형: 수식·숫자·단위 오류 / 편집·조판·교정 오류

심각도: 낮음

원문:
DDPG target은 `r(o_i, a_i^r, o'_i, g_i^tr)`로 쓰지만, TD3/SAC target은 `r_i`를 사용한다. fileciteturn4file11

문제 설명:
`r_i`가 앞서 정의되지 않았다. 의미는 명백하지만, formal notation에서는 `r_i = r(o_i, a_i^r, o'_i, g_i^tr)`를 한 번 정의해야 한다.

수정안:
수식 (7) 앞에 추가한다.
`Let r_i = r(o_i, a_i^r, o'_i, g_i^tr).`

비고:
낮은 심각도의 notation 정리 항목이다.

---

### 오류 20

위치: Reproducibility Snapshot / Bellman Regression
PDF 페이지: 4, 8
본문 페이지: 4, 8
장/절: Preliminaries / Reproducibility Snapshot
그림/표/프로그램/수식 번호: 수식 (7), (10), (12)

오류 유형: 분야별 전문 오류 / 코드·실습·절차 오류 / 독자 관점 오류

심각도: 중간

원문:
“Success state는 Bellman target에서 terminal로 강제하지 않고, timeout transition에는 bootstrapping을 허용한다. 즉, success_terminates=false, timeout_bootstrap=true…” fileciteturn5file10

문제 설명:
수식에서는 `γ(1 − d_i)`를 사용한다. 그런데 `d_i`가 success termination을 포함하지 않는지, timeout에서 어떤 값인지, environment termination과 truncation을 어떻게 나누는지가 명확하지 않다. 특히 Gymnasium 계열은 `terminated`와 `truncated`를 구분하므로, `d_i` semantics가 중요하다.

수정안:
“d_i denotes true environment termination excluding success and excluding time-limit truncation; timeout transitions are bootstrapped.”
또는 `d_i = terminated_i ∧ not timeout_i`처럼 명시한다.

비고:
실제 코드 재현에 영향을 준다.

---

### 오류 21

위치: Evaluation protocol
PDF 페이지: 8–11
본문 페이지: 8–11
장/절: Experiments
그림/표/프로그램/수식 번호: Table 1–5

오류 유형: 수식·숫자·단위 오류 / 분야별 전문 오류

심각도: 중간

원문:
“각 evaluation checkpoint의 success는 20 evaluation episodes의 environment success rate다.”
“모든 ± 값은 seeds 1–5에 대한 sample standard deviation이다.” fileciteturn5file8

문제 설명:
20 evaluation episodes는 성공률 추정의 binomial noise가 클 수 있다. seed variance와 evaluation episode variance가 분리되어 있지 않다. 특히 DrawerOpen은 variance가 매우 크므로, 20 episode evaluation만으로 Hit90 seed count를 판단하면 noise가 커질 수 있다.

수정안:
`100 evaluation episodes` 또는 `20 episodes × multiple evaluation checkpoints smoothing`을 사용한다. 최소한 “Hit90 is computed from raw or smoothed evaluation success”를 명시한다.

비고:
성능 곡선과 first-hit metric 신뢰도에 영향을 준다.

---

### 오류 22

위치: References
PDF 페이지: 12
본문 페이지: 12
장/절: References
그림/표/프로그램/수식 번호: References

오류 유형: 편집·조판·교정 오류 / 분야별 전문 오류

심각도: 중간

원문:
References에는 HER, TD3, BCQ, SAC, CQL, DDPG, PER, Meta-World만 포함되어 있다. fileciteturn5file9

문제 설명:
실험에 FetchPickAndPlace/FetchSlide를 사용하지만 Fetch Robotics/Gym/Gymnasium-Robotics benchmark reference가 없다. 또한 직접 관련 HER 계열 방법의 reference가 없다. 참고문헌 범위가 논문의 주장 범위보다 좁다.

수정안:
Fetch/Gymnasium-Robotics 또는 Plappert et al. multi-goal RL benchmark citation을 추가하고, Related Work에서 추가한 HER variants를 모두 reference에 반영한다.

비고:
related work novelty와 reproducibility 모두에 관련된다.

---

### 오류 23

위치: 전체 원고
PDF 페이지: 1–12
본문 페이지: 1–12
장/절: 전체
그림/표/프로그램/수식 번호: 해당 없음

오류 유형: 편집·조판·교정 오류 / 독자 관점 오류

심각도: 낮음

원문:
원고는 한국어 문장과 영어 technical term이 혼합되어 있으며 review packet은 “최종 제출 전 영어로 번역할 예정”이라고 설명한다. fileciteturn5file1

문제 설명:
현재 PDF가 실제 AAAI 제출본이라면 한국어 본문은 형식상 부적합할 가능성이 높다. 다만 review packet이 “GPT 리뷰용 build이며, 최종 AAAI 제출 형식은 아니다”라고 명시하므로 현재 단계에서는 낮은 심각도의 편집 항목으로 둔다.

수정안:
최종 제출 전 영어 원고로 통일하고, Korean-only explanatory wording은 appendix/internal note로 이동한다.

비고:
현재 review-only draft라면 문제 아님. 최종 submission에서는 치명적 형식 문제로 바뀔 수 있다.

---

### 오류 24

위치: 전체 원고 / 표 배치
PDF 페이지: 9–12
본문 페이지: 9–12
장/절: Experiments / Discussion / References
그림/표/프로그램/수식 번호: Table 1–5

오류 유형: 그림·표·도식 오류 / 편집·조판·교정 오류

심각도: 중간

원문:
Table 1–2가 page 9 상단에 연속 배치되고, Table 3–4가 page 10 상단에 배치되며, Table 5가 page 11 상단에 배치된다. Page 12는 References 이후 여백이 크다. fileciteturn4file1 fileciteturn4file8

문제 설명:
표가 본문 narrative를 끊고, page 9의 긴 appendix 파일명은 오른쪽 여백을 침범한다. 학회 논문에서는 table float와 본문 설명의 인접성이 중요하다. 현재는 결과 표가 후반부에 몰려 있고, 일부 설명이 표 뒤에 나와 독자가 table과 해석을 왕복해야 한다.

수정안:
Table 1은 Main Result 바로 뒤, Table 3은 FetchSlide 설명 직후, Table 5는 Actor-Scope 설명 직후로 이동한다. 긴 appendix 파일명은 삭제하거나 footnote로 처리한다. References 앞 여백은 표 압축 또는 related work 보강에 사용한다.

비고:
조판 완성도와 리뷰어 readability에 영향을 준다.

## 3. 의심 항목

위치: PDF page 5, Local Context
의심되는 문제: FetchPickAndPlace phase predicate에서 `z_lift 이하이면 lift`, 그 외 `place`라는 조건이 실제 구현 의도와 직관적 phase naming 사이에서 어긋날 수 있다.
확인이 필요한 이유: context key는 SABER support lookup의 핵심이므로 phase misclassification이 성능과 해석 모두에 영향을 준다.
저자 또는 편집자가 확인해야 할 사항: 실제 코드의 phase predicate, threshold 값, phase 이름과 알고리즘 의미가 일치하는지 확인.

위치: PDF page 6, Support Indicator
의심되는 문제: `M_Q`가 unique action-bin count인지 local transition count인지 불명확하다.
확인이 필요한 이유: support diversity를 보려는 것인지 sample count를 보려는 것인지에 따라 수식이 달라진다.
저자 또는 편집자가 확인해야 할 사항: `M_Q`의 구현 코드와 논문 수식을 일치시킬 것.

위치: PDF page 7, PER-style formula
의심되는 문제: TD-error weight 계산에서 gradient detach 여부가 문서에 없다.
확인이 필요한 이유: detach하지 않으면 baseline objective가 단순 reweighting이 아니라 다른 gradient term을 갖게 된다.
저자 또는 편집자가 확인해야 할 사항: PyTorch 구현에서 `with torch.no_grad()` 또는 `.detach()`를 사용하는지 확인.

위치: PDF page 8, RQ4
의심되는 문제: “clean mechanism”의 판정 기준이 성능인지 causal isolation인지 불명확하다.
확인이 필요한 이유: Table 5에서는 performance 기준으로 critic-only가 최고가 아니다.
저자 또는 편집자가 확인해야 할 사항: RQ4를 mechanism isolation 질문으로 재정의할 것.

위치: PDF page 9, Table 1
의심되는 문제: Hit90이 raw evaluation curve 기준인지 smoothing curve 기준인지 불명확하다.
확인이 필요한 이유: 20 evaluation episodes만 사용하면 threshold crossing이 noise에 민감하다.
저자 또는 편집자가 확인해야 할 사항: Hit90 산출 코드와 smoothing 여부.

위치: PDF page 10–11, DrawerOpen
의심되는 문제: Random-support가 critic-only와 유사하거나 더 높게 나타난 원인이 positive-rate, weight distribution, seed noise 중 무엇인지 불명확하다.
확인이 필요한 이유: structured support mechanism claim의 핵심 약점이다.
저자 또는 편집자가 확인해야 할 사항: positive-rate matched random, within-context shuffled, cross-context shuffled control을 추가.

위치: PDF page 12, References
의심되는 문제: MetaCog-RL citation이 없고, 관련 HER variants가 빠져 있다.
확인이 필요한 이유: novelty framing과 anonymity에 영향을 줄 수 있다.
저자 또는 편집자가 확인해야 할 사항: 공개 선행연구면 citation, 내부 framework면 삭제 또는 익명화.

## 4. 반복 오류 패턴

1. Mechanism claim은 낮추었지만, 일부 문장에는 여전히 “검증한다”, “보여준다”, “깨끗하게 보여준다”처럼 evidence보다 강한 표현이 남아 있다.

2. Support predicate 관련 기호와 threshold는 개선되었으나, `M_Q`, `m_i`, `|L_i|`, `|B_i|`의 역할 구분이 여전히 약하다.

3. 실험 metric 정의가 이전보다 좋아졌지만, AUC 수식, Hit90 산출 방식, evaluation noise 처리, smoothing 여부가 반복적으로 빠져 있다.

4. Runtime overhead를 여러 번 논의하지만, hardware/software/측정 protocol이 반복적으로 누락된다.

5. Related Work와 baseline이 classic HER/DDPG/TD3/SAC/PER 중심으로 반복되어, 직접 관련 HER improvement 계열이 부족하다.

6. Appendix·external file reference가 본문에 남아 있어 최종 submission package와 연결되지 않는 위험이 반복된다.

7. Table 중심 결과 제시는 풍부하지만, per-seed curve, significance test, diagnostic plot이 부족해 수치 claim의 강건성을 확인하기 어렵다.

## 5. 수정 우선순위

순위: 1
위치: PDF page 6, 수식 (20)
오류 요약: support indicator의 `m_i`, `M_Q`, `|L_i|`, `B_i` 의미가 불명확하고 조건이 중복될 수 있음.
먼저 고쳐야 하는 이유: SABER method의 핵심 정의다.
권장 수정 방향: `M_Q`를 unique action-bin threshold로 쓸지 local sample count로 쓸지 결정하고 수식을 코드와 일치시킬 것.

순위: 2
위치: PDF page 8–11, Theoretical Interpretation / Actor-Scope
오류 요약: support reliability mechanism이 직접 검증되지 않았고 Random-support가 강함.
먼저 고쳐야 하는 이유: 논문의 novelty와 causal story를 좌우한다.
권장 수정 방향: ψ–TD-error correlation, top-k overlap, support-positive/negative별 TD-error 감소량, positive-rate matched random control을 추가.

순위: 3
위치: PDF page 2–3, 12
오류 요약: 직접 관련 HER prioritization/goal-selection/SOTA baseline 관련연구가 누락됨.
먼저 고쳐야 하는 이유: top-tier reviewer가 novelty와 baseline weakness를 먼저 지적할 가능성이 높다.
권장 수정 방향: CHER, EBP, HGG, HGR, MHER, REDQ+HER/high replay ratio를 Related Work와 baseline 후보에 반영.

순위: 4
위치: PDF page 8–10, Experiments
오류 요약: baseline set이 classic HER backbones 중심이며 REDQ+HER/high replay ratio 등 강한 최신 baseline이 없음.
먼저 고쳐야 하는 이유: sample efficiency claim의 설득력을 좌우한다.
권장 수정 방향: FPP에서 최소 1개 strong modern GCRL baseline을 추가하거나, unavailable이면 limitation으로 명확히 인정.

순위: 5
위치: PDF page 8–9
오류 요약: environment version, reward convention, success threshold, hardware 측정 조건 누락.
먼저 고쳐야 하는 이유: 재현성과 runtime claim을 동시에 해친다.
권장 수정 방향: Reproducibility Snapshot을 확장하고 task별 configuration table을 추가.

순위: 6
위치: PDF page 9, Table 1
오류 요약: n=5 평균 개선만으로 “개선한다”는 표현이 강함.
먼저 고쳐야 하는 이유: 통계적 강건성 부족은 ML 리뷰에서 주요 감점이다.
권장 수정 방향: paired seed difference, bootstrap CI, per-seed curve 추가. 문장은 “mean improves”로 낮춤.

순위: 7
위치: PDF page 7, 수식 (26), (27)
오류 요약: PER-style baseline에서 TD-error detach/priority update timing이 불명확함.
먼저 고쳐야 하는 이유: baseline 공정성 논란을 줄여야 한다.
권장 수정 방향: `stopgrad(δ_i)`와 no-gradient weight computation을 명시.

순위: 8
위치: PDF page 9
오류 요약: `AAAI27_REPRODUCIBILITY_APPENDIX_KR.md` 긴 파일명이 조판상 잘리고 외부 파일 참조가 남아 있음.
먼저 고쳐야 하는 이유: 제출본에서 명백한 편집·조판 흔적으로 보인다.
권장 수정 방향: Appendix A/B로 바꾸고 실제 appendix를 포함.

순위: 9
위치: PDF page 3, MetaCog-RL Lineage
오류 요약: citation 없는 internal framework 설명.
먼저 고쳐야 하는 이유: 익명성 및 related work 신뢰도 리스크.
권장 수정 방향: citation을 추가하거나 고유명 삭제.

순위: 10
위치: PDF page 9–12
오류 요약: 표 배치와 본문 흐름이 끊기며 후반부 공백이 큼.
먼저 고쳐야 하는 이유: 최종 제출 품질과 readability에 영향.
권장 수정 방향: 표 위치 재배치, 긴 줄 제거, references 전 공간을 related work/diagnostic으로 활용.

## 6. 최종 집계

전체 오류 개수: 24
치명적 오류 개수: 0
높은 심각도 오류 개수: 9
내용 오류 개수: 10
분야별 전문 오류 개수: 14
수식·숫자·단위 오류 개수: 8
코드·실습·절차 오류 개수: 8
그림·표·도식 오류 개수: 1
편집·조판·교정 오류 개수: 8
독자 관점 오류 개수: 16
의심 항목 개수: 7
반복 오류 패턴 개수: 7

## 7. 편집자용 종합 의견

이 문서는 이전 초안보다 claim을 많이 낮췄고, replay action critic residual과 current policy action support proxy의 관계를 명시적으로 설명한 점이 개선되었다. 특히 Abstract와 Conclusion에서 universal improvement와 wall-clock efficiency를 주장하지 않는다고 못 박은 점은 안전하다. 리뷰 패킷과 claim revision 문서도 현재 가장 안전한 claim을 “support-aligned sparse HER actor-critic setting에서 sample efficiency를 개선할 수 있는 critic-side Bellman error reweighting objective”로 좁히고 있다. fileciteturn5file1 fileciteturn5file2

가장 시급하게 수정해야 할 문제는 세 가지다. 첫째, support indicator 수식이 실제 의도와 맞는지 점검해야 한다. 둘째, structured support mechanism을 입증하는 sample-level diagnostic이 없다. 셋째, Related Work와 baseline이 direct HER improvement 계열에 비해 약하다. 이 세 항목은 단순 오탈자가 아니라 논문의 technical soundness, novelty, experimental validity에 직접 영향을 준다.

독자나 reviewer가 가장 혼동할 가능성이 큰 부분은 `M_Q`의 의미, `Hit90` 산출 방식, `AUC` normalization, FetchSlide/DrawerOpen context 정의, Random-support 결과 해석이다. 특히 Table 5는 support-aware predicate 자체가 유일한 원인이라는 해석을 약화하므로, main claim을 더 방어하려면 positive-rate matched random control과 support/TD-error diagnostic이 필요하다.

조판 과정에서는 page 9의 긴 appendix 파일명 잘림, 표의 후반부 몰림, references 전 여백, 한영 혼합 상태를 다시 확인해야 한다. 최종 AAAI 제출본에서는 현재 review-only Korean build를 영어 제출 양식으로 재작성해야 한다.

다음 교정 단계 체크리스트는 다음 순서가 적절하다.
1. 수식 (20)과 코드의 support predicate 완전 일치 확인.
2. `M_Q`를 unique action support 기준으로 쓸지 결정.
3. FPP/FetchSlide/DrawerOpen task별 context definition table 추가.
4. AUC/Hit90 수식 및 산출 코드 명시.
5. environment version, reward convention, hardware 추가.
6. FPP random-support/shuffled/fixed-count 1M control 추가.
7. ψ–TD-error correlation 및 support-positive/negative diagnostic 추가.
8. REDQ+HER/high replay ratio 또는 직접 관련 HER baseline 추가.
9. Related Work에 CHER, EBP, HGG, HGR, MHER 계열 추가.
10. 조판상 긴 파일명과 appendix 참조를 Appendix 번호로 교체.

---

# Stage 1: Internal Compression

## Core Claims (H_core)

논문은 sparse-reward HER actor-critic에서 HER가 relabeled Bellman target의 수는 늘리지만, 각 update가 current policy action 근방에서 동일하게 reliable한지는 보장하지 않는다고 주장한다. 이 차이를 `hindsight Bellman target reliability gap`으로 부른다. SABER는 local state-goal-horizon context에서 current policy action이 replay-action support를 갖는지 확인하고, support-positive sample의 critic Bellman residual에 더 큰 batch-normalized weight를 적용한다. Actor objective와 replay sampling distribution은 유지된다. fileciteturn5file9

가장 안전한 claim은 universal RL algorithm이 아니라 “support-aligned sparse HER actor-critic setting에서 environment-step sample efficiency를 개선할 수 있는 critic-side Bellman error reweighting objective”다. 이 claim 축소는 보조 claim revision 문서와 현재 Abstract에 일관되게 반영되어 있다. fileciteturn5file2

## Method (M)

SABER는 mini-batch sample마다 training goal `g_i^tr`을 기준으로 local context `c_i = C(o_i, g_i^tr, h_i)`를 만든다. FPP에서는 hand-object distance, object-goal distance, object height, gripper gap, horizon bin, phase 등을 사용해 `C_full` key를 구성한다. replay buffer의 reference transition도 query sample의 training goal 아래에서 다시 context화한다. fileciteturn5file15

그 다음 local replay pool `L_i`를 찾고, replay actions를 action bin set `B_i`로 변환한다. current policy action `a_i^π = π_θ(o_i, g_i^tr)`의 bin `b_i^π`가 `B_i` 안에 있으면 support-positive로 간주한다. Raw weight는 `1 + λ_S ψ_i`, batch-normalized weight는 `Ω_i`이며, critic loss에만 적용한다. Algorithm 1과 수식 (24), (25)는 critic-only update 구조를 제시한다. fileciteturn4file9

## Evidence (E)

FPP 1M steps, seeds 1–5에서 DDPG/TD3/SAC 모두 HER+SABER가 HER보다 mean final success와 success AUC가 높다. 예를 들어 TD3는 final success가 `0.840 ± 0.188`에서 `0.970 ± 0.045`로, SAC는 `0.790 ± 0.288`에서 `0.990 ± 0.022`로 보고된다. FPP DDPG PER-style comparison에서는 SABER AUC `0.609 ± 0.038`, PER-style `0.568 ± 0.069`, full PER `0.460 ± 0.238`이다. fileciteturn4file1

FetchSlide에서는 TD3/SAC가 개선되지만 DDPG는 final success와 last-quarter AUC가 낮아진다. DrawerOpen TD3 stress-test에서는 HER+SABER가 HER보다 높지만 variance가 크고, Table 5에서는 Random-support와 Actor+critic weighting이 critic-only보다 AUC 평균이 높다. fileciteturn4file8 fileciteturn5file12

---

# Stage 2: External Context Construction

## Sub-domain

이 논문은 sparse-reward goal-conditioned reinforcement learning, HER-style hindsight relabeling, off-policy actor-critic, robotic manipulation, support-aware value learning의 교차점에 있다. HER는 sparse and binary reward 환경에서 sample-efficient learning을 가능하게 하고 off-policy RL과 결합 가능하다는 점에서 이 분야의 기본 출발점이다. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html?utm_source=chatgpt.com))

## Relevant Prior Work

Seminal baseline은 HER, DDPG, TD3, SAC, PER다. TD3는 twin critics와 delayed policy update 등을 통해 function approximation error와 overestimation을 줄이는 off-policy actor-critic이고, SAC는 maximum-entropy framework 기반의 off-policy stochastic actor-critic이다. PER는 transition을 uniform하게 replay하는 대신 중요 transition을 더 자주 replay한다. ([proceedings.mlr.press](https://proceedings.mlr.press/v80/fujimoto18a.html?utm_source=chatgpt.com))

직접 관련 HER improvement 계열로는 CHER, Energy-Based Hindsight Experience Prioritization, HGG, MHER 등이 있다. CHER는 failed experience 선택을 goal proximity와 curiosity/diversity 기준으로 curriculum화하고, Energy-Based Prioritization은 robotic manipulation에서 hindsight experience를 energy 기준으로 우선순위화한다. HGG는 valuable hindsight goals를 생성하는 curriculum 계열이고, MHER는 dynamics model로 virtual achieved goals를 생성한다. ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2019/hash/83715fd4755b33f9c3958e1a9ee221e1-Abstract.html?utm_source=chatgpt.com))

Meta-World는 50개 robotic manipulation task로 구성된 meta/multi-task RL benchmark이므로 DrawerOpen stress-test의 배경 benchmark로 적절하다. ([arxiv.org](https://arxiv.org/abs/1910.10897?utm_source=chatgpt.com))

## Current SOTA / Strong Baseline Context

현 시점에서 반드시 비교 후보로 고려해야 할 강한 관련 baseline은 HER와 high replay ratio/regularization을 결합하는 REDQ-style sparse-reward GCRL 접근이다. OpenReview 2024 논문은 high replay ratio와 regularization을 sparse-reward goal-conditioned tasks로 확장하는 방향을 다룬다. ([openreview.net](https://openreview.net/forum?id=7uml1PLgAN&utm_source=chatgpt.com)) 이 논문과 SABER는 모두 sample efficiency를 목표로 하므로, SABER가 top-tier venue에서 설득력을 얻으려면 적어도 FPP에서 REDQ+HER 또는 high replay ratio HER comparison이 필요하다.

## Domain Narrative

이 분야는 HER의 “실패 trajectory를 hindsight goal로 relabel한다”는 기본 아이디어에서 출발해, 어떤 hindsight goal이나 episode를 더 가치 있게 replay할지 고르는 prioritization/curriculum 방향으로 발전했다. 동시에 offline RL에서는 dataset support 밖 action에서 critic extrapolation error가 커지는 문제가 별도 축으로 정리되었다. SABER는 이 두 흐름 사이에 있다. HER처럼 goal distribution을 바꾸지는 않고, PER처럼 sampling distribution을 바꾸지도 않으며, offline RL처럼 policy를 constraint하지도 않는다. 대신 “support가 있는 context의 Bellman residual을 critic이 더 강하게 fitting한다”는 좁은 objective를 제안한다.

기여는 incremental과 meaningful 사이에 있다. Mechanism이 입증되면 meaningful한 reweighting layer가 될 수 있지만, 현재 증거만으로는 random reweighting regularization과 structured support predicate의 효과가 완전히 분리되지 않는다.

---

# Stage 3: Adversarial Baseline & Evaluation Check

Task: FetchPickAndPlace, FetchSlide, Meta-World DrawerOpen sparse goal-conditioned manipulation.

Dataset/Environment: online replay buffer 기반 off-policy actor-critic. FPP/FetchSlide는 Fetch Robotics 계열로 보이나 환경 version과 success threshold가 없다. DrawerOpen은 Meta-World stress-test로 쓰인다.

Metrics: final success, best success, success AUC, median Hit90 step, last-quarter AUC, steps/sec. 현재 metric 정의는 이전보다 좋아졌으나 AUC 수식, Hit90 smoothing 여부, evaluation noise 처리는 부족하다. fileciteturn5file8

Missing baselines: REDQ+HER/high replay ratio, CHER, Energy-Based HER Prioritization, HGG, HGR, MHER/MHER(λ), possibly modern GCRL baselines.

Missing diagnostics: ψ와 |δ| correlation, top-k overlap, support-positive/negative별 Q-error 또는 TD-error 감소량, policy-action neighborhood Q calibration, positive-rate matched random control.

Suspicious evaluation setup: Table 5에서 Random-support AUC가 Critic-only SABER보다 높다. 이는 support predicate가 유일한 mechanism이라는 해석을 약화한다. 원고가 이를 limitation으로 인정한 점은 좋지만, main mechanism claim은 여전히 더 낮춰야 한다. fileciteturn5file12

Unfair or incomplete comparison: PER-style comparison은 FPP DDPG 하나에만 있고, TD-error direct run은 partial 633k라 1M table에서 제외된다. Result matrix도 이 partial result를 main row로 쓰면 안 된다고 정리한다. fileciteturn3file5

---

# Stage 4: Multi-Aspect Skeptical Q&A

## Question 1

Question: SABER의 novelty가 기존 HER prioritization과 충분히 다른가?
Paper-based answer: SABER는 HER relabeling distribution이나 replay sampling을 바꾸지 않고, critic Bellman residual만 support-derived weight로 reweighting한다고 주장한다.
Verification: 이 차별점은 실제로 존재한다. 다만 CHER, EBP, HGG, MHER 등 직접 관련 goal-selection/prioritization 계열을 Related Work와 baseline에서 충분히 다루지 않는다.
Verdict: Weak-to-Moderate.

## Question 2

Question: current policy action support가 replay-action critic residual reliability를 잘 대변하는가?
Paper-based answer: 논문은 direct target noise 측정이 아니라 policy-improvement-relevant reliability proxy라고 낮춰 설명한다.
Verification: proxy로 해석하는 것은 방어 가능하다. 그러나 ψ-positive sample이 실제로 더 낮은 fitting risk를 갖는지는 직접 diagnostic이 없다.
Verdict: Weak.

## Question 3

Question: support indicator 수식은 구현 가능하고 명확한가?
Paper-based answer: `m_i`, `M_B`, `M_Q`, `N_local`, `B_i`를 정의한다.
Verification: `M_Q`가 local transition count인지 unique action support count인지 모호하며, 현재 설정에서는 `N_local`과 중복된다.
Verdict: Weak.

## Question 4

Question: DDPG/TD3/SAC backbone compatibility는 충분히 제시되는가?
Paper-based answer: TD3/SAC target 수식과 twin critic loss가 추가되었고, FPP Table 1에서 세 backbone을 비교한다.
Verification: 수식 구조는 이전보다 훨씬 낫다. 다만 FetchSlide DDPG failure와 runtime slowdown 때문에 “general compatibility”보다는 “FPP에서 세 backbone에 적용 가능” 정도로 낮추는 것이 안전하다.
Verdict: Moderate.

## Question 5

Question: PER-style comparison으로 RQ3가 닫히는가?
Paper-based answer: FPP DDPG에서 SABER가 PER-style/full PER보다 AUC가 높다.
Verification: 단일 backbone/task이고 TD-error full 1M이 없다. TD3/SAC나 FetchSlide로 확장되지 않아 RQ3를 충분히 닫지 못한다.
Verdict: Weak.

## Question 6

Question: Random-support 결과는 SABER mechanism을 지지하는가?
Paper-based answer: 원고는 Random-support가 강하므로 limitation이라고 인정한다.
Verification: Table 5에서 Random-support AUC가 critic-only보다 높다. 이는 exact structured support rule의 독립적 기여를 약화한다.
Verdict: Weak.

## Question 7

Question: 실험 수치의 통계적 신뢰도는 충분한가?
Paper-based answer: seeds 1–5의 mean ± sample standard deviation을 보고한다.
Verification: n=5이며 paired test, CI, per-seed curve가 없다. 특히 DDPG FPP final success 차이는 작다.
Verdict: Weak.

## Question 8

Question: runtime claim은 공정한가?
Paper-based answer: sample efficiency와 wall-clock efficiency를 분리하고 steps/sec를 보고한다.
Verification: hardware/software/parallelism/evaluation 포함 여부가 없어 steps/sec의 재현성이 낮다.
Verdict: Weak.

## Question 9

Question: Reproducibility Snapshot은 충분한가?
Paper-based answer: HER ratio, terminal convention, key hyperparameter는 일부 제공한다.
Verification: 환경 version, reward convention, success threshold, FetchSlide/DrawerOpen context, hardware가 빠져 있다.
Verdict: Weak-to-Moderate.

## Question 10

Question: 현재 claim strength는 AAAI-style reviewer 관점에서 적절한가?
Paper-based answer: Abstract와 Conclusion에서 universal improvement와 wall-clock efficiency를 주장하지 않는다고 낮춘다.
Verification: 큰 방향은 적절하다. 그러나 “검증한다”, “보여준다”, “깨끗하게 보여준다” 등 일부 문장이 evidence보다 강하다.
Verdict: Moderate.

---

# Stage 5: Evidence-Based Review Synthesis

## Summary

이 논문은 sparse-reward HER actor-critic에서 HER가 만든 relabeled Bellman residual을 모두 동일하게 critic fitting하지 말고, local replay-action support를 기준으로 support-positive sample을 더 강하게 학습하자는 SABER objective를 제안한다. Main variant는 critic-only이며 actor objective와 replay sampling distribution을 바꾸지 않는다. FPP에서는 DDPG/TD3/SAC 모두에서 mean final success와 AUC가 개선되지만, FetchSlide DDPG failure와 DrawerOpen Random-support control은 mechanism과 generality에 중요한 한계를 남긴다.

## Strengths

문제 framing이 좁고 명확하다. HER가 target quantity를 늘리는 것과 critic update reliability를 구분하려는 관점은 흥미롭다. 최신 원고는 replay action critic residual과 current policy action support proxy의 관계를 직접 설명하며, ψ가 Bellman target noise의 직접 측정치가 아니라고 낮춰 말한다. FPP에서 세 backbone을 포함한 것은 좋은 empirical signal이다. Random-support와 FetchSlide DDPG failure를 숨기지 않고 limitation으로 제시한 점도 긍정적이다.

## Weaknesses

Major weakness 1: support predicate의 수식과 의미가 아직 완전히 정리되지 않았다. 특히 `M_Q`와 `N_local`의 역할이 중복 또는 모호하다.

Major weakness 2: structured support mechanism이 직접 검증되지 않았다. Random-support가 critic-only와 유사하거나 더 높은 AUC를 보이므로, 현재 결과만으로 exact support rule의 고유한 기여를 주장하기 어렵다.

Major weakness 3: baseline이 약하다. CHER, EBP, HGG, HGR, MHER, REDQ+HER/high replay ratio 등 직접 관련 baseline이 없다.

Major weakness 4: 실험 재현성과 통계 보고가 부족하다. 환경 version, success threshold, hardware, AUC formula, Hit90 smoothing, per-seed curve, paired test가 필요하다.

Minor weakness: page 9의 긴 appendix 파일명 조판, references 부족, Korean review-only build, table float 배치가 최종 제출 전 정리되어야 한다.

## Novelty & Significance

SABER의 novelty는 HER relabeling이나 replay sampling이 아니라 critic-side support-aware Bellman residual reweighting이라는 점에 있다. 이 novelty는 충분히 구분 가능하지만, 직접 관련 HER prioritization 계열과 비교하지 않으면 reviewer가 “another heuristic weighting/prioritization”으로 볼 수 있다. 의미 있는 기여가 되려면 support-specific diagnostic과 strong baseline comparison이 필요하다.

## Technical Soundness

기술적으로 핵심 아이디어는 plausible하지만, 현재 논문은 mechanism proof가 아니라 empirical hypothesis 수준이다. 이 점을 명확히 인정하는 문장은 좋다. 그러나 support indicator 수식, task-specific context, PER-style baseline implementation, terminal semantics가 더 정밀해야 한다. TD3/SAC target 수식 추가는 긍정적 개선이다.

## Suggestions

가장 먼저 수식 (20)을 정리하고, task별 context definition table을 추가하라. 다음으로 FPP에서 positive-rate matched Random-support, Shuffled-weight, Random-fixed-count를 1M seeds 1–5로 실행하고, ψ–TD-error correlation, top-k overlap, support-positive/negative별 TD-error 감소량을 추가하라. Related Work에는 HER improvement 계열을 별도 절로 넣고, 실험 baseline에는 최소한 REDQ+HER/high replay ratio 또는 CHER/EBP 계열 중 하나를 포함하라. 모든 수치 claim은 “mean improvement”로 낮추고, per-seed paired difference와 bootstrap CI를 제공하라. Final submission에서는 review-only Korean build를 영어 AAAI 양식으로 재조판하고, appendix 파일명 참조를 실제 Appendix 번호로 바꾸는 것이 좋다.

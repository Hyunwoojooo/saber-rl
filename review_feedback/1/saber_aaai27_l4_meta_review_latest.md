# SABER AAAI-27 L4 AC/Meta-reviewer 평가

2026-06-24 기준 최신 review packet이 지정한 `saber_aaai27_draft_kr.tex` / `saber_aaai27_draft_kr_review.pdf`를 기준으로 평가합니다. 첨부된 packet은 이 PDF가 최신 `.tex`에서 만든 review-only build라고 명시하므로, 이전 9쪽 빌드는 기준 원고로 쓰지 않았습니다.

# 총평

**현재 단계: 위험 / Reject-high.**
이전 draft 대비 **C3/C8 재현성·프로토콜은 분명히 개선**되었습니다. FPP exact context, action bin, HER ratio, optimizer, batch size, replay size, support scan mode 등 핵심 구현 정보가 본문에 들어왔습니다.

그러나 AAAI-27 top-tier L4 AC 기준에서는 아직 **핵심 mechanism closure가 닫히지 않았습니다.** 특히 DrawerOpen에서 Random-Support가 Critic-only SABER와 비슷하거나 더 높은 AUC를 보이고, artifact matrix도 structured support mechanism의 primary evidence로 쓰면 안 된다고 정리합니다.

현재 가장 안전한 claim은 이미 claim revision 문서에 적힌 대로:

> **SABER is a critic-side Bellman error reweighting objective that can improve environment-step sample efficiency in support-aligned sparse HER actor-critic settings.**

이 범위를 넘어서 “support-aware rule이 원인이다”, “HER reliability gap을 해결했다”, “TD-error/PER와 독립적이다”라고 읽히면 reject risk가 급격히 올라갑니다.

---

# (0) 읽은 범위 체크리스트

| 범위 | 판정 | 메모 |
|---|---:|---|
| Review packet | ✅ | 최신 원고와 리뷰 요청 범위 확인. packet은 최신 원고가 `saber_aaai27_draft_kr.tex`와 `saber_aaai27_draft_kr_review.pdf`라고 명시함. |
| Abstract | ✅ | HER target quantity vs reliability gap, SABER critic-only design, FPP positive, FetchSlide/DDPG failure, DrawerOpen Random-Support caveat가 모두 들어 있음. |
| Introduction / Contributions | ✅ | 문제정의와 narrowed contribution이 명확해짐. claim revision도 FPP gain과 FetchSlide/DrawerOpen limitation을 함께 보고하도록 정리함. |
| Related Work | ✅ | HER, off-policy actor-critic, PER/TD-error, offline support, MetaCog-RL lineage를 다룸. 다만 HGG/CHER/MHER/HGR/EBP 계열 비교는 약함. |
| Preliminaries / Motivation | ✅ | replay action에서 critic residual을 계산하고 current policy action으로 support predicate를 정의한다는 구분이 명시됨. |
| Method / Algorithm | ✅ | Algorithm 1, local context, local replay-action support, support indicator, weight construction, critic objective 확인. |
| Theoretical Interpretation | ✅ | reliability-weighted Bellman regression 해석은 있으나 theorem이 아니라 empirical hypothesis라고 명시함. |
| Experiments | ✅ | FPP, FetchSlide, DrawerOpen, PER-style/full PER, actor/random/shuffled controls 확인. |
| Tables / Visuals | ✅ | Table 1–5 확인. 그러나 learning curve figure, per-seed curve, diagnostic plots는 없음. |
| PDF 내 Appendix / Supplementary Material | ❌ | 최신 review PDF 12쪽 안에는 Appendix 섹션 없음. Reproducibility appendix가 별도 `.md`로 “maintained”된다고 쓰지만, PDF 내부 appendix는 아님. |
| 별도 Supplement PDF | ❌ | 별도 supplement PDF는 제공되지 않음. 따라서 “supp에 있음” 형태의 근거는 논문 평가에서는 인정하지 않음. |
| 보조 artifact/result matrix | ✅/참고 | 제출 논문 본문 근거가 아니라, missing baseline과 artifact readiness 판단용으로만 사용. |

---

# (1) Claim Ledger

| # | 핵심 claim | 근거 위치 / 근거 유형 | 닫힘 |
|---:|---|---|---:|
| 1 | HER는 relabeled Bellman target의 quantity를 늘리지만, current policy action 근방의 reliability를 보장하지 않는다. | Abstract/Introduction/Motivation. 개념 정의와 해석. claim revision도 “critic-side proxy”로 좁힘. | ⚠️ |
| 2 | SABER는 local state-goal-horizon context에서 current policy action이 replay-action support 안에 있는지 보고, support-positive sample의 critic residual만 더 강하게 fitting한다. | Method, Algorithm 1, Eq. support indicator/weight construction. 정의/알고리즘. | ✅ |
| 3 | FPP에서 SABER는 DDPG/TD3/SAC+HER 모두의 environment-step final success와 AUC를 개선한다. | Table 1. DDPG AUC 0.539→0.609, TD3 final 0.840→0.970, SAC final 0.790→0.990. 실험. | ✅/⚠️ |
| 4 | SABER의 이득은 단순 TD-error/PER-style weighting으로 설명되기 어렵다. | Table 2의 FPP DDPG PER-style/full PER 비교. 그러나 TD-error 1M은 없고 support-positive vs high-TD-error overlap/correlation도 아직 필요하다고 원고가 인정함. | ⚠️ |
| 5 | SABER는 universal improvement가 아니며 task/backbone-dependent limitation이 있다. | FetchSlide Table 3에서 DDPG는 final 0.600→0.520, last-quarter AUC 0.519→0.476으로 악화. TD3/SAC는 개선. | ✅ |
| 6 | Structured support-aware rule이 성능 개선의 핵심 원인이다. | DrawerOpen Table 5에서 Random-Support final/AUC가 Critic-only와 유사하거나 더 높음. Result matrix도 structured support mechanism의 주증거로 쓰지 말라고 정리함. | ❌ |

**핵심 문제:** 현재 논문은 claim을 많이 낮췄지만, title/method의 중심이 여전히 “Support-Aware”입니다. 따라서 #6이 닫히지 않으면 C11에서 치명 리스크가 남습니다.

---

# (2) C0–C11 체크리스트

| 항목 | 판정 | 근거 및 판단 | ✅로 올리는 최소 수정 |
|---|---:|---|---|
| C0. 문제정의·기여·필요성 | ✅ | HER target quantity와 Bellman update reliability를 구분하는 문제의식은 명확함. Contribution도 critic-side objective, backbone compatibility, limitation reporting으로 좁혀짐. | 현재 톤 유지. 단, “gap”을 empirical diagnostic으로도 보여주면 더 강해짐. |
| C1. Novelty/차별점·관련연구 | ⚠️ | HER/PER/offline support/MetaCog-RL과의 구분은 있음. 그러나 packet에서 사용자가 직접 우려한 PER/HGR/EBP/MHER/HGG/CHER 계열과의 closest-prior 비교가 충분히 닫히지 않음. | Related Work에 closest prior comparison table 추가. HGG/CHER/MHER/HER curriculum/goal-prioritized relabeling과 “relabeling distribution vs critic residual weighting” 축으로 분리. |
| C2. 공정하고 강한 baseline/비교 | ⚠️ | FPP에서 HER baseline 3 backbones와 PER-style/full PER는 좋음. 그러나 main-task Random-Support/Shuffled/Fixed-Count 1M이 없고, Result Matrix도 FPP mechanism controls가 1k smoke뿐이라 main/appendix 불가라고 적음. | FPP DDPG Random-Support, Shuffled-Weight, Random-Fixed-Count 1M seeds 1–5 추가. 가능하면 TD3 또는 SAC에도 random/shuffled control 추가. |
| C3. 실험 프로토콜·세팅 정의 | ⚠️ | 이전보다 개선됨. FPP exact context, bins, support scan, HER ratio, hyperparams가 들어감. 그러나 FetchSlide/DrawerOpen task-specific context는 paper 본문에서 FPP만큼 exact하지 않고, reproducibility appendix는 PDF 내부가 아님. | Appendix/Supp에 task별 context key, bin threshold, support scan, env version, eval interval, seeds, command/config를 모두 넣기. |
| C4. Ablation/요소 분해 | ⚠️ | PER-style/full PER, actor-only/actor+critic, random/shuffled controls가 있음. 하지만 strongest control이 DrawerOpen에 몰려 있고, FPP primary evidence에는 Random-Support/Shuffled/Fixed-Count가 없음. | FPP primary task에서 support predicate ablation, λS, K bins, Nlocal/MQ/MB, context feature removal 추가. |
| C5. 일반화/강건성 | ⚠️ | FPP는 3 backbones 모두 개선. FetchSlide는 DDPG 실패, TD3/SAC 소폭 개선. DrawerOpen은 variance가 큼. 따라서 “support-aligned sparse HER settings” 정도의 좁은 claim만 안전함. | FetchSlide DDPG failure 원인 분석. FetchPush는 appendix-only로 정확히 제한. 추가 task를 넣더라도 saturated task는 main evidence로 과장 금지. |
| C6. 효과크기·통계·민감도 | ⚠️ | seeds 1–5 mean±std는 보고됨. 그러나 DDPG FPP final gain은 0.940→0.960으로 작고, DrawerOpen은 std가 매우 큼. Random-Support도 매우 강함. | Paired-seed bootstrap CI, per-seed learning curves, final/AUC significance, 10 seeds 또는 same-horizon paired analysis 추가. |
| C7. 실용성/비용·지연 | ⚠️ | steps/sec를 정직하게 보고하고 wall-clock efficiency를 주장하지 않는 점은 좋음. 그러나 slowdown이 큼: FPP DDPG 65.03→41.72, FetchSlide DDPG 119.08→26.85. | wall-clock-to-threshold, support lookup overhead breakdown, optimized index/cache version, inference-time cost 0 여부 명시. |
| C8. 재현성/artifact | ⚠️ | 코드 공개는 요구하지 않음. 논문만으로 FPP primary는 어느 정도 재현 가능해졌지만, full task reproducibility는 아직 외부 `.md appendix`에 의존하고 PDF 내부에 없음. | Reproducibility appendix를 제출 PDF/supplement에 포함. YAML-style config와 task별 exact support predicate 제공. |
| C9. 스코프 명확성 | ✅ | replay action critic residual과 current policy action support proxy를 명시적으로 구분했고, actor objective/replay sampling unchanged도 반복해서 명시함. | 모듈별 변경표를 추가하면 더 안전하지만, 현재도 치명적 애매성은 크게 줄었음. |
| C10. 논리 일관성·과장/정직성 | ✅ | universal improvement, wall-clock efficiency, support as unique cause, theorem claim을 피하라고 claim revision에 명시되어 있고, 원고도 FetchSlide/DDPG failure와 Random-Support caveat를 숨기지 않음. | 이 톤 유지. Abstract/title에서 “support-aware solves reliability gap”처럼 읽히지 않게 계속 방어. |
| C11. Claim Closure + 구성 품질 | ❌ | 핵심 performance claim 일부는 닫혔지만, 핵심 mechanism claim은 닫히지 않음. DrawerOpen Random-Support가 Critic-only와 유사/우위이고, FPP primary에는 Random/Shuffled controls가 없음. Learning curves/diagnostic plots도 부족함. | support signal closure 실험 필수: FPP main-task random/shuffled/fixed controls, corr(ψ, \|δ\|), top-k overlap, support-positive vs negative Bellman residual reduction, target variance diagnostics. |

**요약:** ✅ 3개 / ⚠️ 8개 / ❌ 1개.
전 버전보다 C3/C8은 확실히 좋아졌지만, **C11 ❌ 하나가 여전히 stage를 “위험”으로 끌어내립니다.**

---

# (3) Scorecard: N/S/E/R/I

| 축 | 점수 | 판단 |
|---|---:|---|
| N — Novelty | 2.5 / 5 | HER target generation과 critic residual weighting을 분리하는 framing은 괜찮음. 그러나 method는 local support bin + critic loss reweighting으로 단순하고, closest prior 대비 이론적 레버리지는 제한적임. |
| S — Significance | 2.5 / 5 | sparse HER actor-critic의 실제 pain point를 건드림. 하지만 FPP 중심, FetchSlide DDPG failure, wall-clock overhead 때문에 top-tier significance는 제한됨. |
| E — Evidence | 2.0 / 5 | FPP 3-backbone result는 긍정적. 그러나 mechanism evidence가 DrawerOpen random controls 때문에 약하고, FPP primary mechanism controls가 없음. |
| R — Reproducibility/Rigor | 2.0 / 5 | FPP exact context와 hyperparams가 들어와 개선됨. 그래도 task별 full reproducibility와 diagnostics가 아직 부족함. |
| I — Impact/Interest | 2.5 / 5 | critic-side reliability proxy라는 아이디어는 흥미롭지만, 현재 evidence로는 “nice workshop/weak conference paper” 수준에 머물 가능성이 큼. |

**Unweighted total: 11.5 / 25**

---

# (4) Venue/Profile 가중 L4Total

가정: **AAAI-27 top-tier ML/RL method profile**
가중치: N 1.1 / S 1.0 / E 1.2 / R 0.9 / I 0.8

| 축 | 점수 | 가중치 | 가중 기여 |
|---|---:|---:|---:|
| N | 2.5 | 1.1 | 2.75 |
| S | 2.5 | 1.0 | 2.50 |
| E | 2.0 | 1.2 | 2.40 |
| R | 2.0 | 0.9 | 1.80 |
| I | 2.5 | 0.8 | 2.00 |

**L4Total = 11.45 / 25**

해석: 전 draft 대비 protocol/reproducibility 점수는 소폭 상승했지만, top-tier acceptance 기준에서는 여전히 낮습니다. 현재 제출하면 **Reject 또는 Weak Reject가 가장 가능성 높은 판정**입니다.

---

# (5) Stage 결정

**Stage: 위험**

강제 규칙상 C2/C8/C9/C11 중 하나라도 ❌이면 최소 “위험”인데, 현재는 **C11 ❌**입니다. C2와 C8도 ❌까지는 아니지만 강한 ⚠️입니다. 따라서 “약하게 위험”이 아니라 **위험**으로 둡니다.

메타리뷰식 판정:

> The paper has a cleaner and more honest scope than the previous draft, and the FPP evidence is nontrivial. However, the central support-aware mechanism remains under-closed because random reweighting controls are competitive in the strongest diagnostic setting and are missing on the primary FPP task. The paper is not yet ready for a top-tier main-track submission.

---

# (6) 상대 비교

외부 “과거 논문 베이스라인 테이블”은 제공되지 않았습니다. 대신 `AAAI27_RESULT_MATRIX_KR.md`를 내부 artifact/result matrix로 보고, 논문 내부 Table 1–5와 함께 상대 위치를 평가합니다.

## 내부 rank

| 비교군 | SABER 위치 |
|---|---|
| FPP DDPG HER/PER/SABER | **AUC 1위**: SABER 0.609, PER-style 0.568, HER 0.539, PER-full 0.460. 단, steps/sec는 SABER가 가장 느림. |
| FPP 3 backbones | **sample-efficiency positive**: DDPG/TD3/SAC 모두 final/AUC 개선. 특히 TD3/SAC final success 개선은 의미 있음. |
| FetchSlide | **mixed**: TD3/SAC는 개선, DDPG는 악화. 이 결과 때문에 universal claim은 불가. |
| DrawerOpen TD3 controls | **mechanism 약함**: Random-Support AUC 0.426이 Critic-only SABER AUC 0.362보다 높고, Actor+Critic도 0.432로 높음. |
| Artifact readiness | FPP main rows와 PER-style/full PER는 ready지만, FPP Random/Shuffled/Fixed-Count는 1k smoke 또는 없음. |

## 유사/위험 비교 3개

1. **DDPG+HER+PER-style**
   SABER보다 낮지만 HER보다 높은 AUC를 보입니다. 따라서 “그냥 TD-error/loss weighting 아니냐”는 반론은 일부만 막습니다. TD-error 1M rerun 또는 support-vs-TD diagnostic이 필요합니다.

2. **DrawerOpen Random-Support**
   가장 치명적인 control입니다. Structured support가 아니어도 Critic-only SABER와 비슷하거나 더 높게 나옵니다. 이 결과는 “support-aware rule이 유일한 원인”이라는 해석을 사실상 막습니다.

3. **FetchSlide DDPG HER+SABER**
   method generality를 제한하는 failure case입니다. 원고가 이를 limitation으로 둔 것은 옳지만, reviewer는 “왜 FPP만 되는가?”를 물을 가능성이 높습니다.

---

# (7) 한 단계 올리는 최소 액션

1. **FPP primary mechanism controls를 먼저 닫으세요.**
   Result matrix가 이미 정리한 대로 FPP Random-Support, Shuffled-Weight, Random-Fixed-Count는 1M seeds 1–5가 없습니다. AAAI 리스크를 닫으려면 이 셋이 최우선입니다.

2. **support가 TD-error/PER와 다른 signal이라는 sample-level diagnostic을 넣으세요.**
   최소 `corr(ψ_i, |δ_i|)`, support-positive와 high-TD-error top-k overlap, support-positive/negative residual reduction, support bin별 target variance를 넣어야 합니다. 현재 원고도 이 분석 필요성을 인정합니다.

3. **Reproducibility appendix를 실제 제출물 안으로 넣으세요.**
   지금 본문은 FPP exact context는 좋아졌지만, FetchSlide/DrawerOpen까지 완결되지는 않았습니다. 별도 `.md`가 “maintained”된다는 문장은 심사자가 볼 수 있는 supplement가 아니면 감점 방어가 안 됩니다.

4. **Related work를 closest-prior table로 재작성하세요.**
   HGG/CHER/MHER/HER curriculum/goal relabeling prioritization 계열과 “goal selection/relabeling을 바꾸는가 vs critic residual weight를 바꾸는가 vs replay sampling을 바꾸는가 vs actor objective를 바꾸는가”로 분리해야 합니다.

5. **Learning curves와 통계 검정을 추가하세요.**
   Table-only RL paper는 L4 reviewer에게 약합니다. Per-seed curves, bootstrap CI, paired-seed AUC/final comparison, failure seed visualization을 넣어야 C6/C11이 올라갑니다.

6. **claim을 더 방어적으로 유지하세요.**
   지금 claim revision 방향은 맞습니다. “Structured support is the unique cause”와 “SABER proves lower target noise”는 계속 피해야 합니다. Random-Support가 FPP에서도 강하면 논문은 “support-aware reliability”가 아니라 “critic-side batch-normalized reweighting regularization with support heuristic” 쪽으로 pivot해야 합니다.

---

# 최종 판정

**현재 제출 기준: Reject-high / Stage 위험.**
**가장 큰 reject reason:** C11, 즉 **support-aware mechanism closure 부재**입니다.
**가장 빠르게 올릴 수 있는 축:** C2/C4/C11. FPP primary task에서 Random/Shuffled/Fixed controls와 sample-level diagnostics가 나오면, Stage는 “위험”에서 “약하게 위험” 또는 “보더 하”까지는 올라갈 수 있습니다.

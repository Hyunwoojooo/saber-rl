# SABER L4 AC/Meta-review

**Paper:** `saber_aaai27_draft_kr.pdf`
**Assumed Venue/Profile:** AAAI27 draft / top-tier ML-RL applied-method profile
**Reviewer Role:** Top-tier conference L4 AC/Meta-reviewer
**Overall Stage:** **위험 / Reject 쪽**
**L4Total:** **11.0 / 25**

---

## 총평

현재 원고는 **흥미로운 critic-side HER 보강 아이디어**가 있고, 자기 한계도 비교적 정직하게 드러낸 편입니다. 그러나 top-tier L4 AC 기준에서는 **“support-aware”라는 핵심 메커니즘이 아직 닫히지 않았고**, 실험 프로토콜과 재현성 정보가 부족하며, Random-Support 결과가 주장을 상당히 약화합니다. 현 상태 판정은 **위험 / Reject 쪽**입니다. 업로드된 9쪽 PDF 전체를 기준으로 평가했습니다.

가정한 Venue/Profile은 파일명 기준 **AAAI27 draft**, 즉 AAAI/NeurIPS/ICML식 top-tier ML/RL applied-method profile입니다. 별도 베이스라인 테이블은 제공되지 않아 상대 비교는 논문 내부 Table 1–5 기준으로만 수행합니다.

---

# (0) 읽은 범위 체크리스트

| 범위 | 판정 | 메모 |
|---|---:|---|
| Abstract | ✅ | 문제, 방법, 주요 결과, limitation이 모두 요약됨. Abstract에서 FPP 개선, FetchSlide DDPG 실패, DrawerOpen random-support 이슈까지 명시함. |
| Introduction / Contributions | ✅ | “hindsight Bellman target reliability gap”와 critic-only SABER의 기여가 명확히 제시됨. |
| Related Work | ✅ | HER, DDPG/TD3/SAC, PER, offline support/OOD, MetaCog-RL lineage를 다룸. 다만 closest prior coverage는 충분하지 않음. |
| Preliminaries / Motivation | ✅ | Goal-conditioned MDP, Bellman regression, target quantity vs reliability motivation이 제시됨. |
| Method / Algorithm | ✅ | Algorithm 1, support indicator, weight construction, critic objective가 있음. |
| Theoretical Interpretation | ✅ | reliability-weighted Bellman regression 해석과 “not a theorem” caveat가 있음. |
| Experiments / Discussion / Conclusion | ✅ | RQ, tasks/metrics, main/secondary/stress-test, limitation, runtime discussion 포함. |
| Tables / Visuals | ✅ | Table 1–5 확인. Learning curve figure는 없음. |
| Appendix in same PDF | ❌ | Appendix 섹션 없음. 본문에서 FetchPush는 appendix/보조 evidence로 제한한다고 쓰지만, 제공 PDF 안에는 FetchPush appendix 결과가 없음. |
| Separate Supplement PDF | ❌ | 별도 supplement PDF가 제공되지 않음. “supp에 있음”으로 가정할 수 없음. |

---

# (1) Claim Ledger

| # | 핵심 주장 | 근거 위치 / 유형 | 닫힘 |
|---:|---|---|---:|
| 1 | HER는 target quantity를 늘리지만, relabeled Bellman target reliability는 보장하지 않는다. | Introduction/Motivation의 문제정의와 “target quantity is not target reliability” 논의. 개념 정의/해석. | ⚠️ |
| 2 | SABER는 local replay-action support를 사용해 support-positive sample의 critic TD loss를 더 크게 반영하고, actor objective/replay sampling은 유지한다. | Algorithm 1, Eq. 14–18. 정의/알고리즘. | ✅ |
| 3 | FetchPickAndPlace에서 DDPG/TD3/SAC 모두 HER 대비 final success와 AUC가 개선된다. | Table 1. 실험. DDPG AUC 0.539→0.609, TD3 final 0.840→0.970, SAC final 0.790→0.990. | ✅/⚠️ |
| 4 | SABER의 이득은 단순 TD-error/PER-style weighting으로 설명되기 어렵다. | Table 2에서 DDPG FPP 기준 PER-style/Full-PER 비교. 논문도 support–TD-error correlation 추가 필요성을 인정함. | ⚠️ |
| 5 | SABER는 여러 backbone에 호환되지만 모든 task/backbone에서 항상 개선되지는 않는다. | FetchSlide Table 3에서 TD3/SAC는 개선, DDPG는 final 및 late AUC 악화. | ✅ |
| 6 | support-aware structured rule이 핵심 원인이다. | DrawerOpen Table 5에서 Random-Support가 Critic-Only와 유사하거나 더 높은 AUC를 보임. Shuffled-Weight 실패는 alignment 가능성을 시사하지만 structured support claim 자체는 닫히지 않음. | ❌ |

핵심 AC 리스크는 #6입니다. 원고가 정직하게 limitation을 적은 점은 좋지만, **제목과 기여의 중심 단어가 “Support-Aware”인 이상, Random-Support가 강하게 나오는 현상은 단순한 부록 이슈가 아니라 메커니즘 closure 문제**입니다.

---

# (2) C0–C11 체크리스트

| 항목 | 판정 | 근거 및 감점 이유 | ✅로 올리는 최소 수정 |
|---|---:|---|---|
| C0. 문제정의·기여·필요성 | ✅ | 문제의식은 명확함. HER가 goal relabeling은 하지만 current policy action 근방의 critic reliability를 보장하지 않는다는 setting을 잘 정의함. 기여도 5개로 정리됨. | 현재 수준 유지. 단, “gap”을 empirical diagnostic으로도 보여주면 더 강함. |
| C1. Novelty/차별점·관련연구 | ⚠️ | HER/PER/offline support와의 차이는 설명하지만, support-aware weighting, HER prioritization, goal-conditioned curriculum/prioritized relabeling 계열의 closest prior가 충분히 촘촘하지 않음. 이론적 레버리지는 theorem이 아니라 hypothesis임을 인정함. | closest prior 5개 내외를 표로 비교. “local support critic loss reweighting”이 기존 PER/HER prioritization과 정확히 무엇이 다른지 formal axis로 분해. |
| C2. 강한 베이스라인/비교 | ⚠️ | HER baseline을 DDPG/TD3/SAC에 붙인 점은 좋음. PER-style/full PER 비교도 있음. 그러나 PER 비교는 FPP DDPG 중심이고, Random/Fix/Shuffled control은 DrawerOpen 중심이라 main evidence와 mechanism evidence가 분리됨. 강한 goal-conditioned RL baseline과 wall-clock budget baseline도 부족함. | FPP/FetchSlide에도 Random-Support, Fixed-Count, Shuffled, PER-style, full PER를 동일 프로토콜로 추가. 동일 env-step뿐 아니라 동일 wall-clock 비교 추가. |
| C3. 실험 프로토콜·세팅 정의 | ❌ | 1M/2M steps, seeds 1–5, 일부 threshold는 있으나, 네트워크 구조, optimizer/lr, batch size, replay buffer, HER ratio/strategy, evaluation episodes, env version, exact binning/hash, context feature discretization이 충분히 고정되어 있지 않음. “can include” 식 local context 설명은 재현 프로토콜이 아님. | Appendix에 full config table 추가. task별 context vector, bin edge, action binning, lookup implementation, HER relabeling strategy, train/eval loop를 고정된 pseudocode로 명시. |
| C4. Ablation/요소 분해 | ⚠️ | PER-style/full PER, actor-only/actor+critic, random/shuffled controls가 있어 방향은 좋음. 하지만 support predicate 요소별 ablation, λS sensitivity, batch_norm vs unnormalized, context feature ablation, lookup size ablation이 부족함. | λS, Nmin/Mmin, context-feature removal, action-bin granularity, batch normalization, lookup pool size ablation 추가. |
| C5. 일반화/강건성 | ⚠️ | FPP는 3 backbones 모두 개선. FetchSlide는 DDPG 실패, TD3/SAC 소폭 개선. DrawerOpen은 stress-test지만 variance가 큼. 따라서 “general method” 주장은 제한적이어야 함. | 적어도 하나의 추가 sparse-goal benchmark 또는 FetchPush complete rows 제공. FetchSlide DDPG failure 분석을 정량화. |
| C6. 효과크기·통계·민감도 | ⚠️ | mean±std와 5 seeds는 제시됨. 그러나 일부 개선폭은 작고, DrawerOpen은 std가 매우 큼. 예: DrawerOpen SABER final 0.590 ± 0.539, baseline 0.190 ± 0.425. Random-Support도 0.600 ± 0.548로 유사함. | 10 seeds 이상 또는 paired-seed bootstrap CI. per-seed curves, statistical test, effect size, support-positive vs support-negative variance/TD-error 감소량 제시. |
| C7. 실용성/비용·지연 | ⚠️ | steps/sec를 정직하게 보고하고, wall-clock efficient라고 주장하지 않는 점은 좋음. 그러나 slowdown이 큼. FPP DDPG 65.03→41.72, FetchSlide DDPG 119.08→26.85 steps/sec. | wall-clock-to-success, lookup overhead breakdown, optimized/cache lookup version, inference-time cost가 0인지 명시. |
| C8. 재현성/artifact | ❌ | 코드 미공개 자체는 감점하지 않지만, 논문만으로 핵심 support predicate와 실험을 재구성하기 어렵다. 특히 task-specific context와 binning/lookup이 정확히 고정되어 있지 않음. | “논문만 보고 재현 가능한” implementation appendix 추가. 최소한 YAML-style full configs와 task별 support predicate exact definition 제공. |
| C9. 스코프 명확성 | ⚠️ | critic-only, actor objective unchanged, replay sampling unchanged는 명확함. 그러나 local context/horizon/action bin scope가 애매하고, support lookup이 training-only인지 runtime metric에 어떤 비용으로 들어가는지 더 명확해야 함. | 모듈별 표: actor/critic/replay/HER/support-index/inference 각각 변경 여부, 학습/추론 사용 여부, gradient flow 여부 명시. |
| C10. 논리 일관성·과장/정직성 | ✅ | 원고가 스스로 “not a theorem”, “not wall-clock efficient”, “task/backbone-dependent limitation”, “Random-Support weakens structured support claim”을 명시함. 이 정직성은 강점. | 제목/abstract에서 “support-aware mechanism proven”처럼 읽히지 않게 더 좁은 claim으로 유지. |
| C11. Claim Closure + 글/그림/구성 | ❌ | 성능 claim 일부는 닫히지만, 핵심 mechanism claim은 닫히지 않음. Random-Support가 Critic-Only와 유사하거나 더 높고, 논문도 support–TD-error 독립성/overlap 분석이 필요하다고 인정함. Learning curves/diagnostic plots도 없음. | support indicator가 실제로 “reliability”를 잡는다는 직접 진단 필수: corr(ψ, \|δ\|), top-k overlap, Bellman residual reduction, support-positive/negative target variance, random controls를 main tasks에도 적용. |

요약하면, 현재 체크리스트는 **✅ 2개 / ⚠️ 7개 / ❌ 3개** 수준입니다. 특히 **C3, C8, C11**이 top-tier 제출에서 치명적입니다.

---

# (3) Scorecard: N/S/E/R/I, 0–5

| 축 | 점수 | 이유 |
|---|---:|---|
| N — Novelty | 2.5 | HER target quantity vs reliability framing은 괜찮지만, support/OOD/PER/reweighting 계열과의 거리가 충분히 크다고 보기 어렵다. 방법도 local support bin + critic loss reweighting으로 비교적 단순함. |
| S — Significance | 2.5 | sparse goal-conditioned RL의 실제 문제를 다루지만, 실험이 오래된/small-scale robotics benchmark 중심이고, wall-clock overhead가 크며, task/backbone dependency가 뚜렷함. |
| E — Evidence | 2.0 | FPP 3-backbone 개선은 긍정적. 그러나 FetchSlide DDPG 실패, DrawerOpen variance, Random-Support 결과, main-task mechanism controls 부족 때문에 evidence가 불안정함. |
| R — Reproducibility/Rigor | 1.5 | algorithm/equation은 있으나 실험 재현에 필요한 세팅이 크게 부족함. 특히 support predicate의 exact definition이 부족함. |
| I — Impact/Interest | 2.5 | critic-only reliability weighting이라는 좁고 해석 가능한 아이디어는 관심을 끌 수 있음. 다만 현 evidence로는 top-tier main-track impact로 밀기 어렵다. |

**Raw total: 11.0 / 25**

---

# (4) Venue/Profile 가중 L4Total 계산

가정한 AAAI/top-tier applied ML/RL method profile 가중치:

| 축 | 점수 | 가중치 | 가중 기여 |
|---|---:|---:|---:|
| N | 2.5 | 1.1 | 2.75 |
| S | 2.5 | 1.0 | 2.50 |
| E | 2.0 | 1.2 | 2.40 |
| R | 1.5 | 0.9 | 1.35 |
| I | 2.5 | 0.8 | 2.00 |

**L4Total = 11.0 / 25**

해석: **workshop/early-stage로는 가능성이 있으나, top-tier main-track 기준으로는 현재 reject-risk가 높음.**

---

# (5) Stage 결정

**Stage: 위험**

강제 규칙상 **C8 또는 C11 중 하나라도 ❌이면 최소 “위험”**인데, 현재는 C8과 C11이 모두 ❌입니다. 여기에 C3도 ❌이고, C2/C4/C5/C6/C7/C9가 ⚠️이므로 “약하게 위험”이 아니라 **명확히 위험**으로 둡니다.

메타리뷰식 한 줄 판정:

> The paper has a plausible and cleanly scoped idea, but the central support-aware mechanism is not yet empirically closed, and the experimental protocol is insufficiently specified for a top-tier acceptance decision.

---

# (6) 상대 비교

외부 베이스라인 테이블은 제공되지 않았으므로, 내부 Table 1–5 기준으로만 비교합니다.

## 내부 상대 위치

| 비교 축 | SABER 위치 |
|---|---|
| FPP env-step sample efficiency | 강함. DDPG/TD3/SAC 모두 final/AUC 개선. 특히 TD3 final 0.840→0.970, SAC final 0.790→0.990은 눈에 띔. |
| FPP wall-clock throughput | 약함. 모든 backbone에서 steps/sec 감소. DDPG 65.03→41.72, TD3 65.33→41.25, SAC 38.93→29.10. |
| FetchSlide generalization | 혼합. TD3/SAC는 개선, DDPG는 final 0.600→0.520 및 last-quarter AUC 0.519→0.476으로 악화. |
| DrawerOpen stress-test | 결과 자체는 좋아 보이나 variance가 너무 큼. TD3+HER final 0.190±0.425, SABER 0.590±0.539. Hit90은 1/5→3/5. |
| Mechanism controls | 취약. Random-Support AUC 0.426이 Critic-Only AUC 0.361보다 높고 Actor+Critic AUC 0.432도 높음. support-specific mechanism claim을 막지 못함. |

유사한 내부 비교 3개를 꼽으면:

1. **DDPG+HER+PER-style**: TD-error weighting baseline으로서 가장 직접적인 대안. SABER가 AUC는 높지만, PER-style도 HER보다 AUC가 높아 “그냥 weighting 효과” 가능성을 완전히 닫지 못함.
2. **B3-Random-Support**: 가장 위험한 control. structured support가 아니어도 비슷하거나 더 높은 AUC가 나옴.
3. **FetchSlide DDPG HER+SABER**: generality claim을 제한하는 failure case. 이 결과를 숨기지 않은 것은 좋지만, top-tier에서는 원인 분석이 필요함.

---

# (7) 한 단계 올리는 최소 액션

1. **C3/C8/C9부터 복구해야 함.**
   Appendix에 full reproducibility block을 넣으세요: env version, network architecture, optimizer/lr, batch size, replay buffer size, HER relabeling ratio/strategy, eval episodes, seed list, logging interval, success metric definition, all hyperparameters. 특히 task별 local context, bin edge, action binning, support lookup, Nmin/Mmin/Ngap/MB의 exact definition을 표와 pseudocode로 고정해야 합니다.

2. **Random-Support 문제를 main tasks에서 정면으로 닫아야 함.**
   FPP와 FetchSlide에도 Critic-Only, Random-Support, Random-Fixed-Count, Shuffled-Weight, PER-style, full PER를 동일 seeds로 돌리세요. DrawerOpen에서만 control을 제시하면 “main result도 random reweighting일 수 있다”는 반론을 막지 못합니다.

3. **support가 reliability signal이라는 직접 진단을 추가해야 함.**
   최소한 corr(ψᵢ, |δᵢ|), top-k overlap between support-positive and high-TD-error samples, support-positive vs support-negative Bellman residual 감소량, support bin별 target variance, actor action의 local support coverage over training을 보여줘야 합니다.

4. **통계와 learning curves를 보강해야 함.**
   현재 표만으로는 high variance RL에서 AC가 설득되기 어렵습니다. Per-seed learning curves, bootstrap CI, paired-seed comparison, final/AUC significance, drop best-final curves를 넣으세요. 5 seeds를 유지하더라도 paired analysis는 필요하고, 가능하면 10 seeds로 늘리는 것이 좋습니다.

5. **wall-clock claim을 별도 축으로 분리하세요.**
   “sample-efficient but not always wall-clock efficient”는 이미 좋은 톤입니다. 여기에 wall-clock-to-90%, support lookup overhead breakdown, optimized indexing/cache version을 추가하면 C7 리스크가 크게 줄어듭니다.

6. **논문 claim을 더 좁히세요.**
   현재 가장 안전한 claim은 “SABER is a critic-side reweighting objective that can improve env-step sample efficiency in support-aligned sparse HER settings”입니다. “structured support is the cause”는 아직 닫히지 않았으므로, closure 실험 전까지는 abstract/title/conclusion에서 과도하게 밀지 않는 편이 안전합니다.

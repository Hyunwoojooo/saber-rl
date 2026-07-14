# SABER AAAI 논문 작성 및 실험 설계 가이드

> 기준일: 2026-07-14  
> 대상 초안: `SABER_draft_0708.pdf`  
> 목표 학회: AAAI-27 Main Technical Track

## 핵심 방향

SABER는 **새로운 범용 강화학습 백본**보다 **HER 이후 critic loss allocation이라는 별도의 설계 축을 규명한 controlled empirical method**로 제시하는 것이 현재 증거와 가장 잘 맞는다.

AAAI-27은 초록 2026년 7월 21일, 본문 7월 28일, 보충자료와 코드 7월 31일 마감이다. 본문은 최대 7페이지이며, 8~9페이지에는 참고문헌과 윤리·재현성 체크리스트 등 허용된 항목만 배치할 수 있다. Supplement는 심사자가 반드시 읽어야 하는 자료가 아니므로 핵심 claim과 근거는 본문에 있어야 한다.

- [AAAI-27 Main Technical Track Call](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/)
- [AAAI-27 Conference](https://aaai.org/conference/aaai/aaai-27/)

## 유사 논문의 공통 서술 흐름

### 1. 기존 방법의 역할을 먼저 인정한다

HER는 sparse reward를 완화하고 실패한 trajectory를 유효한 goal-conditioned 학습 신호로 전환한다. 기존 방법의 기여를 축소하지 않은 상태에서, 아직 다루지 않은 의사결정을 분리해야 한다.

### 2. 기존 방법이 결정하지 않는 정확한 지점을 찾는다

HER는 hindsight goal과 Bellman target을 구성하지만, 생성된 residual마다 critic fitting pressure를 얼마나 배분할지는 결정하지 않는다. 따라서 문제를 단순한 "HER의 비효율"이 아니라 **post-relabeling critic-loss allocation 문제**로 정의한다.

### 3. 인접 문제와 intervention point를 분리한다

Goal construction, replay sampling, critic-loss allocation, update scaling은 서로 다른 개입 지점이다. 이 구분이 SABER의 novelty와 실험 공정성을 동시에 설명한다.

### 4. 최소 변경 방법을 제시한다

HER relabeling, replay distribution, actor objective, Bellman target, backbone architecture는 유지한다. SABER는 local state-goal-horizon context에서 current-policy action support를 확인하고, support-positive sample의 critic Bellman residual에 더 큰 weight를 부여한다.

### 5. 대안 설명을 직접 통제한다

성능 향상이 structured support signal 때문인지, 단순한 random reweighting인지, TD-error prioritization인지, 추가 계산량인지 각각 대응하는 control을 둔다.

### 6. 성공 조건과 실패 조건을 함께 보여준다

성공한 환경만 나열하지 않고 task와 backbone에 따른 효과 차이, negative case, runtime overhead를 함께 보고한다. 이는 claim의 범위를 제한하는 동시에 결과의 신뢰도를 높인다.

### 7. 증거보다 작은 claim을 사용한다

"SABER가 항상 개선한다"가 아니라 "matched sparse-HER 조건에서 개선할 수 있으며, 특정 task/backbone 조건에서 효과가 강하다"고 주장한다.

## 참고할 논문과 구성상의 교훈

| 논문 | 핵심 구성 | SABER에 주는 교훈 |
|---|---|---|
| [ReenGAGE, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26024) | DDPG+HER의 critic loss만 변경하고 theory, toy problem, robotics, 실패 조건을 연결 | 백본과 HER를 고정한 critic-only intervention을 명확히 기술한다 |
| [Attentive Experience Replay, AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6049) | Current-policy relevance를 제기하고 DDPG/TD3/SAC, PER, runtime, confounder ablation을 비교 | 정확한 confounder마다 대응 control을 둔다 |
| [Off-Policy Deep RL by Bootstrapping Covariate Shift, AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4246) | 기존 방법의 구체적 결함을 지적하고 네 가지 대안 설명을 분리하며 mixed result도 공개 | "왜 좋아졌는가"에 대한 reviewer의 대안 가설을 선제적으로 검증한다 |
| [A Deeper Understanding of State-Based Critics, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/21171) | 환경 속성별 가설, 이론, toy example, 실험을 연결 | 단순 leaderboard보다 작동 조건을 중심으로 결과를 구성한다 |
| [Hindsight Goal Ranking](https://arxiv.org/abs/2110.15043) | Episode와 future goal을 two-step TD-error ranking하며 component ablation 수행 | HGR은 HER goal-selection baseline으로 공정하게 분리한다 |
| [Curriculum-Guided HER, NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/83715fd4755b33f9c3958e1a9ee221e1-Abstract.html) | 목표 선택 규칙의 component ablation과 선택 goal 시각화 | 알고리즘 수식뿐 아니라 실제 선택 결과를 진단한다 |
| [DisCor, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/d7f426ccbc6db7e235c57958c21c5dfa-Paper.pdf) | 추정된 target error에 따라 Bellman update를 reweight | Critic-side reweighting 계열에서 SABER와 가장 가까운 차이를 명시한다 |

## SABER의 권장 포지셔닝

논문의 핵심 문제 제기는 다음과 같이 쓸 수 있다.

> Existing HER variants primarily intervene at goal construction or replay selection. We study an orthogonal post-relabeling question: conditional on the same sampled HER minibatch, should every critic residual receive the same fitting pressure?

이어 다음 문장으로 HER 자체를 부정하는 오해를 차단한다.

> This distinction does not imply that HER targets are invalid; it separates target construction from critic-loss allocation.

### 인접 방법과의 경계

| 방법 | 변경하는 항목 | SABER와의 차이 |
|---|---|---|
| HER, HGR, CHER | Hindsight goal의 생성 또는 선택 | SABER는 relabeling 이후의 loss allocation을 변경한다 |
| PER | Replay sampling distribution | SABER는 sampled minibatch와 sampling distribution을 유지한다 |
| ReenGAGE | Goal-gradient 기반 critic objective | SABER는 local current-policy action support 기반 weight를 사용한다 |
| DisCor | 추정된 propagated target error 기반 Bellman reweighting | SABER는 sparse HER 문맥에서 local replay support를 사용하며 별도 error model을 학습하지 않는다 |
| High replay ratio, REDQ | Update 수 또는 critic 수 | SABER는 matched one-update budget에서 fitting pressure를 재배분한다 |
| Offline support methods | 고정 dataset에서의 policy extrapolation 제어 | SABER는 support intuition을 사용하지만 offline-RL guarantee를 주장하지 않는다 |

## 권장 Claim 구조

### 1. 방법론적 claim

SABER는 동일한 HER minibatch에서 current-policy action이 local replay action support에 존재하는 sample의 critic residual을 더 강하게 fitting하는 critic-only objective다.

### 2. 제한된 실증적 claim

Matched one-update-per-step 조건에서 SABER는 FPP TD3/SAC와 DrawerOpen DDPG의 success AUC 및 학습 안정성을 개선한다. FPP DDPG는 평균 개선이 있으나 paired confidence interval이 0을 포함하므로 통계적으로 확정적인 개선처럼 표현하지 않는다.

### 3. Mechanism-consistent claim

Support signal은 TD error와 동일하지 않다. TD3/SAC에서는 support-positive subset이 high-TD-error sample을 enrich하고 더 큰 immediate residual correction을 보인다. 이는 제안한 mechanism과 일치하는 관찰이지만 인과적 증명은 아니다.

### 4. 명시적 non-claim

다음 사항은 주장하지 않는다.

- Support-positive sample이 항상 저분산이거나 본질적으로 더 좋은 target이라는 주장
- 모든 task와 backbone에서 개선한다는 주장
- Wall-clock efficiency가 항상 우월하다는 주장
- Random controls만으로 structured support signal의 인과성이 증명되었다는 주장
- Offline RL의 extrapolation error를 해결한다는 주장

## 권장 실험 설계

현재 6개 RQ는 다음 세 가설로 압축하는 것이 좋다.

### H1. SABER가 matched HER baseline을 개선하는가?

- FPP에서 DDPG, TD3, SAC 비교
- DrawerOpen에서 DDPG 비교
- Architecture, HER ratio, exploration, Bellman target, terminal convention, update budget 고정
- Primary metric은 normalized success AUC로 사전 고정
- Final success와 Hit90은 secondary metric으로 사용
- 같은 seed를 사용한 paired comparison과 paired bootstrap 95% confidence interval 보고

### H2. 개선이 어떤 변경에서 발생하는가?

- Random-support: 동일하거나 유사한 positive 비율을 갖는 random selector
- Shuffled weights: 동일한 weight multiset을 유지하되 sample 대응 관계 제거
- Random-fixed: support lookup 없이 고정 확률 reweighting
- PER-style weighting: uniform replay를 유지한 TD-error loss weighting
- Full PER: sampling distribution까지 변경하는 baseline
- HGR: two-step episode/future-goal prioritization baseline
- High-RR와 REDQ: 추가 compute가 제공하는 성능을 보여주는 별도 reference
- Critic-only와 actor+critic scope ablation

### H3. 언제, 왜 작동하거나 실패하는가?

- Support-positive base rate
- Top-k TD-error enrichment와 support/TD-error correlation
- Update 전후 absolute TD residual correction
- Weight-normalized residual correction
- Support-positive/negative target variance
- Support lookup과 전체 update의 wall-clock timing
- FetchSlide, DrawerClose 등 negative 또는 mixed task/backbone 결과
- `lambda`, local-neighbor 수, action-bin/context 정의에 대한 sensitivity

DrawerOpen의 diagnostic 10-seed artifacts가 완성되면 aggregate statistics와 confidence interval을 사용한다. Per-seed learning curve와 전체 수치는 supplement에 공개한다.

## 기존 방법을 비판하는 문장 구성

기존 연구를 비판할 때는 그 방법이 해결한 목적을 먼저 인정하고, 다루지 않는 intervention point를 정확히 지적한다.

권장 표현:

> HGR prioritizes which hindsight goals enter training; it does not determine their relative fitting pressure after a minibatch has been formed.

> Increasing replay ratio tests the benefit of additional optimization, whereas SABER asks how a fixed update budget should be allocated.

> PER alters replay frequency according to a global transition priority. SABER retains uniform replay and asks whether local support for the current policy action should affect critic fitting pressure after hindsight relabeling.

피해야 할 표현:

- "HER trains on useless samples."
- "SABER solves extrapolation error."
- "Random controls prove SABER's causal mechanism."
- "SABER is universally compute efficient."
- "SABER is state of the art"라는 포괄적 표현
- 충분한 문헌 검토 없이 사용하는 "the first" 표현

## 현재 초안에서 우선 수정할 부분

### 1. 23페이지 초안을 7페이지 본문으로 재구성한다

현재 초안의 6개 RQ와 다수의 반복 설명, 24개 표를 그대로 유지할 수 없다. 핵심 claim과 가장 직접적인 근거만 본문에 남기고 나머지는 supplement로 이동한다.

### 2. 잠재 변수 표현을 완화한다

현재의 `U_i^fit`과 "hindsight Bellman critic-fit usefulness gap"은 직접 관측되거나 정리로 증명된 양이 아니다. 이를 theorem처럼 보이게 하기보다 **post-relabeling critic-loss allocation question**과 **design rationale**로 표현하는 것이 안전하다.

### 3. 가장 가까운 related work를 보강한다

ReenGAGE, Attentive Experience Replay, covariate-shift correction, DisCor를 직접 비교해야 한다. 특히 DisCor를 누락한 채 critic-side Bellman reweighting을 새로운 축이라고 강하게 주장하면 novelty reviewer에게 취약하다.

### 4. Random-Support 결과를 정직하게 해석한다

FPP DDPG와 DrawerOpen에서 Random-Support가 강하므로, 현재 데이터는 exact structured support rule만이 유일한 원인임을 입증하지 않는다. 논문의 중심을 **loss allocation이라는 문제 설정**에 두고 SABER를 그에 대한 support-conditioned 구현으로 제시한다.

### 5. Hand-crafted context의 일반화 위험을 다룬다

Task-specific phase, state context, action bin은 가장 큰 reviewer 위험이다. Context recipe를 실험 전에 고정하고 task adapter를 명시하며, 주요 설정에 대한 sensitivity와 실패 조건을 공개한다. 현재 증거로는 "task agnostic"이라고 표현하지 않는다.

### 6. Runtime claim을 분리한다

SABER의 support lookup은 측정 가능한 overhead를 발생시킨다. 다른 하드웨어의 AUC/hour를 직접 비교해 compute efficiency를 주장하지 말고, 동일 하드웨어에서 update throughput과 상대 overhead를 보고한다.

## 권장 7페이지 본문 구성

| 절 | 권장 분량 | 핵심 내용 |
|---|---:|---|
| 1. Introduction | 0.9p | HER의 역할, post-relabeling gap, SABER, 세 가지 기여 |
| 2. Related Work and Positioning | 0.6p | HER prioritization, replay prioritization, critic reweighting과의 경계 |
| 3. Problem and Design Rationale | 0.7p | 고정된 HER minibatch에서의 loss-allocation 문제 |
| 4. SABER | 1.2p | Context, support lookup, normalized critic weight, unchanged components |
| 5. Hypotheses and Protocol | 0.7p | H1-H3, matched settings, metric, seed, 통계 절차 |
| 6. Results | 2.4p | Main effect, causal controls, diagnostics, negative cases, runtime |
| 7. Limitations and Conclusion | 0.4p | 적용 범위, hand-crafted context, mixed cases, 결론 |

본문 시각 자료는 다음 정도로 제한한다.

- Figure 1: HER target construction과 SABER critic-only intervention을 분리한 개념도
- Figure 2: FPP 세 backbone과 DrawerOpen의 주요 learning curves
- Figure 3: Control 또는 task-by-backbone effect와 diagnostic을 결합한 panel
- Table 1: Main aggregate result와 HGR 비교
- Table 2: Control closure, mechanism diagnostic, runtime overhead 요약

Per-seed 결과, 전체 stress test, 세부 hyperparameter, 환경별 context 정의, 현재의 상세 표는 supplement에 둔다.

## Introduction 권장 흐름

1. Sparse goal-conditioned RL에서 HER가 실패 trajectory를 유효한 학습 신호로 바꾸는 이유를 설명한다.
2. HER의 target construction과 critic-loss allocation이 서로 다른 결정임을 제기한다.
3. SABER의 local current-policy action-support rule과 변경하지 않은 항목을 설명한다.
4. Main effect, control 결과, mechanism evidence, negative case를 한 문단에서 제한적으로 요약한다.
5. Contribution은 세 항목만 둔다: 문제 설정, 방법, controlled evidence.

## 권장 Abstract 골격

1. Sparse goal-conditioned actor-critic과 HER의 역할
2. HER가 target을 구성하지만 critic fitting pressure는 대체로 uniform이라는 문제
3. HER, replay, actor, target을 유지하는 SABER critic-only objective
4. Matched-budget main result와 가장 중요한 control/diagnostic 결과
5. 적용 범위와 제한을 포함한 결론

## 최종 논문 메시지

SABER의 가장 방어 가능한 메시지는 다음과 같다.

> Hindsight relabeling determines what Bellman targets are constructed, but not how a fixed critic-update budget should be allocated among them. SABER exposes this post-relabeling allocation axis and provides a support-conditioned critic objective that can improve sparse goal-conditioned learning under matched HER sampling, while exhibiting task- and backbone-dependent limitations.

핵심은 SABER를 모든 환경에서 우월한 heuristic으로 포장하는 것이 아니다. **Target construction 이후 critic fitting pressure를 어떻게 배분할 것인가가 독립적으로 연구할 가치가 있는 문제이며, SABER가 그 문제를 검증 가능한 방식으로 드러낸다**는 흐름이 현재 결과와 가장 잘 부합한다.

# SABER AAAI-27 제출 전략 메모

작성일: 2026-06-23

관련 문서:

- `AAAI27_CORE_ARTIFACTS.md`
- `SABER_AAAI27_theory_outline.md`
- `SABER_THEORY_LINEAGE_KR.md`
- `/Users/joo/Downloads/AuthorKit27/AAAI27_SUBMISSION_RULES.md`

## 1. 핵심 판단

현재 제안된 방향성은 매우 합당하며, 현재 결과와 아티팩트 상태에서 AAAI-27 제출용으로 가장 방어 가능한 방향이다. 다만 "accept 확률이 높다"는 말은 절대 보장이 아니라, 넓은 범용 RL 주장이나 MetaCog-RL 전체 이론 주장보다 리뷰어 공격면이 훨씬 좁고 방어하기 쉽다는 의미다.

AAAI급 리뷰에서는 단순히 "결과가 좋다"보다 문제 정의, 기여 경계, ablation 설계, negative case 해석이 명확해야 한다. AAAI-26 관련 규모와 리뷰 프로세스 참고 자료로는 다음 문서를 확인할 수 있다.

- AI-Assisted Peer Review at Scale: The AAAI-26 AI Review Pilot: https://arxiv.org/abs/2604.13940

논문 전체를 관통하는 thesis는 다음 문장으로 잡는 것이 좋다.

> HER creates targets; SABER decides which Bellman errors the critic should trust more.

SABER의 가장 방어 가능한 정의는 다음이다.

> SABER는 sparse-reward goal-conditioned HER setting에서 HER가 만들어낸 Bellman target들을 모두 동일하게 신뢰하지 않고, local replay-action support가 있는 target의 critic Bellman error를 더 강하게 학습시키는 방법이다.

## 2. 왜 이 방향이 좋은가

### SABER를 HER 변형으로 만들지 않는다

HER는 goal relabeling이고, SABER는 relabeled Bellman error의 critic-side reliability weighting이다. 이 구분이 명확해야 related work에서 HER, HGG, HGR, MHER류와 불필요하게 겹치지 않는다.

### PER/TD-error와 분리된다

SABER는 큰 TD-error를 더 학습하는 방법이 아니다. Replay sampling distribution도 기본적으로 바꾸지 않는다. 핵심 차이는 다음과 같다.

- PER: TD-error priority를 사용해 샘플링 분포를 바꾼다.
- TD-error weighting: Bellman error magnitude 자체를 기준으로 loss를 키운다.
- SABER: local replay support를 Bellman supervision reliability 신호로 사용해 critic TD loss만 재가중한다.

따라서 논문에서 다음 문장을 강하게 밀 수 있다.

> SABER does not prioritize surprise; it prioritizes supported Bellman supervision.

### MetaCog-RL 전체 주장을 피한다

현재 SABER는 world model ensemble, ignorance/aleatoric decomposition, CVaR-style robust control을 포함하지 않는다. 따라서 "MetaCog-RL full theory 구현"으로 주장하면 구현과 claim이 어긋난다.

가장 안전한 해석은 다음이다.

> SABER는 MetaCog-RL의 PRACTICE/incompetence 아이디어를 sparse HER actor-critic에 맞게 최소화한 critic-side instantiation이다.

### Critic-only 설계가 깨끗하다

Actor까지 weight를 걸면 support signal이 critic reliability인지 actor state-goal distribution bias인지 섞인다. SABER를 critic-only로 고정하면 support signal이 critic Bellman fitting reliability만 바꾼다는 해석이 깔끔해진다.

## 3. Accept 가능성을 높이는 요소

### 문제 정의가 명확하다

핵심 문제는 다음이다.

> Hindsight relabeling은 training target을 늘리지만, 모든 relabeled Bellman update가 current policy action 근방에서 equally reliable한 것은 아니다.

이는 HER의 빈틈을 정확히 찌르는 문제 정의다.

### 방법이 단순하다

SABER는 local state-goal-horizon context를 만들고, current policy action이 local replay action support 안에 있으면 support-positive로 판단한 뒤, batch-normalized weight를 critic TD loss에 곱한다.

단순한 objective는 reviewer가 이해하기 쉽고, ablation으로 mechanism을 검증하기 좋다.

### 실험 claim을 좁히면 결과와 잘 맞는다

현재 결과 기준으로 가장 강한 증거는 FetchPickAndPlace다. FPP에서는 DDPG, TD3, SAC 계열 모두에서 HER baseline 대비 개선이 명확하다. FetchSlide에서는 SAC/TD3는 개선되지만 DDPG는 약하다.

따라서 논문 구조는 다음이 가장 안전하다.

- FetchPickAndPlace: primary evidence
- FetchSlide: generalization + task/backbone-dependent limitation
- FetchPush: 완전한 row만 보조 evidence
- DrawerOpen: stress-test / ablation / limitation discussion

## 4. 가장 큰 리스크

### 리스크 1: Novelty가 "그냥 loss weighting"으로 보일 수 있음

SABER는 구조적으로 간단한 loss weighting이다. 리뷰어가 "PER/TD-error weighting과 무엇이 다른가?"라고 물을 가능성이 높다.

이를 방어하려면 support와 TD-error가 다른 신호라는 분석이 필요하다.

추천 추가 분석:

$$
\operatorname{corr}(\psi_i, |\delta_i|)
$$

$$
\text{Top-}k\text{ overlap}(\psi_i, |\delta_i|)
$$

$$
\Delta |\delta|_{\text{after update}}
\text{ by support bin}
$$

이 분석이 있으면 "SABER does not prioritize surprise; it prioritizes supported Bellman supervision"이라는 주장이 강해진다.

### 리스크 2: Backbone-agnostic claim이 과해질 수 있음

FPP에서 DDPG/TD3/SAC 모두 좋아지는 것은 강한 결과다. 하지만 FetchSlide DDPG가 반례라면 다음 표현은 위험하다.

> SABER is backbone-agnostic and generally improves all actor-critic backbones.

더 안전한 표현은 다음이다.

> SABER is compatible with multiple off-policy actor-critic backbones and shows consistent gains on FetchPickAndPlace across DDPG, TD3, and SAC, while revealing task/backbone-dependent limitations on FetchSlide.

즉, "compatible/tested across backbones"는 가능하지만 "universally beneficial across backbones"는 피해야 한다.

### 리스크 3: DrawerOpen Random-Support가 structured-support claim을 약화시킴

DrawerOpen에서 Random-Support가 강하다면 "정확한 structured support alignment만이 성능 향상의 원인"이라는 주장은 위험하다.

피해야 할 claim:

> SABER works because structured local support is always the unique useful weighting signal.

권장 claim:

> In the main sparse HER tasks, support-aware critic weighting improves Bellman fitting, and shuffled controls show that sample-weight alignment can matter. DrawerOpen suggests that in some tasks, support-conditioned or even randomized reweighting may also act as a stabilizing regularizer.

이렇게 쓰면 Random-Support 결과는 약점이 아니라 정직한 limitation/discussion이 된다.

### 리스크 4: Runtime overhead

SABER는 local support lookup 때문에 wall-clock cost가 있다. 이를 숨기면 안 된다. Sample efficiency와 wall-clock efficiency를 분리해서 보고해야 한다.

권장 지표:

- Eval AUC
- Final success
- Best success
- Drop from best to final
- Steps-to-threshold
- Wall-clock time
- Steps/sec
- AUC/hour

핵심 해석:

> SABER can be sample-efficient but is not always wall-clock-efficient.

## 5. 실험 테이블 배치

### Main Table 1: FetchPickAndPlace primary result

가장 중요한 테이블이다.

| Task | Backbone | HER baseline | SABER | Delta AUC | Delta Final | Delta Best | Drop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FPP | DDPG | | | | | | |
| FPP | TD3 | | | | | | |
| FPP | SAC | | | | | | |

이 테이블이 논문의 중심이어야 한다. `AAAI27_CORE_ARTIFACTS.md` 기준으로 FPP는 B0, b3_critic_only_runtime_opt, SAC HER/SABER, TD3 HER/SABER의 결과 경로가 정리되어 있어 main table source로 쓰기 좋다.

### Main Table 2: Mechanism ablation

FPP 중심으로 넣는 것이 가장 깔끔하다.

| Method | AUC | Final | Best | Drop | Weight ESS | Support align |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HER baseline | | | | | | |
| SABER | | | | | | |
| SABER-AC | | | | | | |
| SABER-Random | | | | | | |
| SABER-Shuffled | | | | | | |
| TD-error weighting | | | | | | |
| PER-style weighting | | | | | | |

이 테이블이 있으면 "명확한 problem + simple objective + mechanism ablation" 구조가 완성된다.

### Main Figure 1: conceptual diagram

반드시 넣는 것이 좋다. HER가 goal을 relabel하고, SABER가 local support를 확인해서 critic loss weight를 정하는 그림이다. 방법이 단순하므로 그림이 reviewer 이해도를 크게 올릴 수 있다.

### Secondary main: FetchSlide

FetchSlide는 "일부 backbones에서는 잘 작동하지만 DDPG에서는 실패한다"는 정직한 보조 결과로 넣는 것이 좋다.

권장 문장:

> FetchSlide reveals a backbone-dependent failure mode: SABER improves SAC and TD3 but not DDPG, suggesting that support-aware critic weighting interacts with the stability of the underlying actor-critic learner.

이런 negative case 해석은 논문 신뢰도를 올릴 수 있다.

### FetchPush

FetchPush는 main table에 너무 크게 넣지 않는 것이 좋다. 현재 core artifacts 기준으로 DDPG result와 TD3 SABER result는 있지만, SAC 결과 폴더는 GitHub checkout에 없고 TD3-HER-B0도 결과 폴더가 없다.

권장 처리:

- 결과가 완전한 DDPG/TD3 subset만 appendix
- main text에서는 additional FetchPush evidence로 짧게 언급
- full cross-backbone table에는 넣지 않음

### DrawerOpen

DrawerOpen은 main result가 아니라 external generalization, stress-test, failure analysis로 쓰는 것이 맞다. Random-Support가 강하므로 structured-support mechanism의 주증거로 쓰면 위험하다.

## 6. 논문 claim 수정안

### Abstract/Intro용 핵심 claim

> Hindsight relabeling increases the number of goal-conditioned Bellman targets, but it does not ensure that each target is equally reliable near the current policy action. SABER addresses this reliability gap by using local replay-action support as a critic-side Bellman error weighting signal.

### Method contribution

> SABER computes a binary support predicate from a local state-goal-horizon replay context and applies the resulting batch-normalized weight only to the critic Bellman error, leaving both replay sampling and actor updates unchanged.

### Experiment claim

> On FetchPickAndPlace, SABER improves HER-based DDPG, TD3, and SAC, supporting compatibility across common off-policy actor-critic backbones. Additional FetchSlide and DrawerOpen results reveal both benefits and limitations, including task/backbone-dependent failures and cases where randomized support-style reweighting is competitive.

## 7. 반드시 피해야 할 문장

다음 표현은 피해야 한다.

> SABER is a general-purpose RL algorithm.

범위가 너무 넓다.

> SABER solves extrapolation error.

너무 강하다. "mitigates support-related critic fitting instability" 정도가 더 안전하다.

> SABER is a new HER method.

부정확하다. SABER는 HER와 결합되는 critic objective다.

> SABER proves support-positive samples have lower Bellman target noise.

증명하지 못했다면 위험하다. "We interpret", "We hypothesize", "We empirically examine"으로 써야 한다.

> SABER always improves over HER.

FetchSlide DDPG 때문에 바로 반박된다.

## 8. 보완 우선순위

### 1순위: main/appendix result matrix 확정

Main 후보:

- FPP DDPG HER vs SABER
- FPP TD3 HER vs SABER
- FPP SAC HER vs SABER
- FPP SABER vs SABER-AC
- FPP SABER vs Random/Shuffled
- FPP SABER vs TD-error/PER-style

Secondary main 또는 appendix 후보:

- FetchSlide SAC/TD3 positive
- FetchSlide DDPG negative
- FetchPush only complete rows
- DrawerOpen stress/generalization

### 2순위: support vs TD-error 분리 분석

이 분석은 accept 가능성에 직접적으로 중요하다. SABER가 PER 변형이 아니라는 핵심 방어선이다.

### 3순위: random/shuffled 해석 정교화

SABER-Shuffled가 실패하면 "alignment matters"를 말할 수 있다. 하지만 Random-Support가 강한 task가 있으면 "structured support만 중요하다"가 아니라 다음처럼 조정해야 한다.

> Support-conditioned critic reweighting and reweighting-induced regularization may both contribute, depending on the task.

### 4순위: runtime plot

AUC improvement vs runtime overhead plot이 있으면 reviewer가 "비싸기만 한 방법 아닌가?"라고 묻는 것을 미리 막을 수 있다.

### 5순위: acronym 공식화

논문에서는 다음 이름으로 고정하는 것을 추천한다.

> SABER: Support-Aware Bellman Error Reweighting

현재 코드상 공식 acronym 확장이 명확하지 않으므로, 논문에서 새로 정의하는 것이 좋다.

## 9. 최종 논문 방향

이 방향으로 논문 작성을 진행하는 것이 맞다. 단, main claim은 다음처럼 제한해야 한다.

> SABER is a support-aware critic-side Bellman error reweighting method for sparse goal-conditioned HER. It improves critic fitting by upweighting locally supported, policy-relevant Bellman updates, while leaving actor updates and replay sampling unchanged.

실험 claim은 다음처럼 제한한다.

> The strongest evidence is FetchPickAndPlace across DDPG, TD3, and SAC. FetchSlide and DrawerOpen are used to show generality, stress-test behavior, and limitations rather than to claim universal improvement.

이 framing이면 논문은 "새로운 HER/SOTA"보다 명확한 reliability gap, 간단한 objective, 충분한 ablation, 정직한 limitation으로 읽힌다. AAAI급 리뷰에서는 이쪽이 훨씬 방어력이 높다.

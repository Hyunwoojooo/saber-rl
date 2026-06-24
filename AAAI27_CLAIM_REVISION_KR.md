# SABER AAAI-27 Claim Revision Notes

작성일: 2026-06-24

목적: 현재 evidence 범위 안에서 AAAI-27 원고의 claim을 좁히고, Abstract/Introduction/Contribution/Conclusion에 반영한 기준을 기록한다.

## 1. 핵심 판단

현재 원고의 가장 안전한 claim은 다음이다.

> SABER is a critic-side Bellman error reweighting objective that can improve environment-step sample efficiency in support-aligned sparse HER actor-critic settings.

한국어로는 다음과 같다.

> SABER는 support-aligned sparse HER actor-critic setting에서 environment-step sample efficiency를 개선할 수 있는 critic-side Bellman error reweighting objective다.

이 범위를 넘는 claim은 아직 위험하다.

## 2. 이번 수정에서 낮춘 claim

| 기존 방향 | 수정 방향 |
| --- | --- |
| Sparse goal-conditioned RL 전반에 대한 방법처럼 보임 | Sparse HER actor-critic으로 범위 축소 |
| HER의 reliability gap을 해결한다고 보일 수 있음 | reliability proxy를 사용하는 critic-side 접근으로 축소 |
| support가 유일한 원인처럼 읽힐 수 있음 | Random-Support가 reweighting regularization 가능성을 남긴다고 명시 |
| 여러 backbone/task에 일반적으로 좋다고 읽힘 | FPP에서는 consistent gain, FetchSlide DDPG는 failure case로 명시 |
| wall-clock efficiency도 기대하게 할 수 있음 | sample efficiency와 wall-clock efficiency를 분리 |
| PER/TD-error와 다름을 확정한 듯 보임 | 현재는 limited evidence이며 sample-level diagnostic 필요성을 명시 |

## 3. 유지해야 할 중심 문장

논문 전체 thesis:

> HER creates training goals; SABER decides how strongly the critic should fit each resulting Bellman residual.

조금 더 정확한 버전:

> HER increases the number of goal-conditioned Bellman targets, while SABER uses local replay-action support as a critic-side proxy for weighting Bellman residuals.

## 4. Abstract 반영 내용

수정된 Abstract는 다음 네 가지를 명시한다.

- HER는 target quantity를 늘리지만 update reliability를 보장하지 않는다.
- SABER는 local state-goal-horizon context에서 current policy action support를 확인한다.
- SABER는 critic Bellman residual에만 weight를 적용하고 actor/replay sampling은 유지한다.
- FPP positive result와 FetchSlide/DrawerOpen limitation을 함께 말한다.

중요한 caveat:

> SABER는 universal improvement나 wall-clock efficiency를 주장하지 않는다.

## 5. Introduction 반영 내용

Introduction에서 다음을 명시했다.

- Critic residual은 replay action에서 계산되고, actor는 current policy action을 exploit한다.
- SABER support predicate는 Bellman target noise의 직접 측정치가 아니다.
- 더 정확히는 policy-improvement-relevant reliability proxy다.
- Critic-only 설계는 HER relabeling, prioritized replay, actor objective shaping과 구분하기 위한 선택이다.

## 6. Contribution 반영 내용

Contribution은 다음 범위로 낮췄다.

1. hindsight Bellman target reliability gap 문제 제기
2. local replay-action support를 사용하는 critic residual reweighting objective 제안
3. DDPG/TD3/SAC에 붙일 수 있는 backbone-compatible critic reweighting layer 정의
4. FPP gain과 FetchSlide/DrawerOpen limitation을 함께 보고
5. PER/random-control과의 차이를 분석하되, 아직 닫히지 않은 mechanism diagnostic을 명시

## 7. Conclusion 반영 내용

Conclusion은 다음 메시지로 닫는다.

- SABER는 HER replacement가 아니다.
- SABER는 prioritized replay algorithm도 아니다.
- SABER는 critic loss reweighting objective다.
- FPP에서는 sample-efficiency 개선을 보였지만, FetchSlide DDPG와 DrawerOpen Random-Support는 한계를 보여준다.
- 최종 제출 전 보강해야 할 것은 task/backbone dependence, runtime overhead, sample-level mechanism diagnostic이다.

## 8. 피해야 할 문장

최종 원고에서 피해야 할 문장:

- SABER is a general-purpose RL algorithm.
- SABER solves extrapolation error.
- SABER is a new HER method.
- SABER always improves over HER.
- Structured support is the unique cause of improvement.
- SABER is wall-clock efficient.
- SABER proves support-positive samples have lower target noise.

대신 쓸 문장:

- SABER is a critic-side loss reweighting objective for sparse HER actor-critic.
- SABER uses local replay-action support as a policy-improvement-relevant reliability proxy.
- SABER can improve environment-step sample efficiency in support-aligned sparse HER settings.
- Random-support controls suggest that reweighting-induced regularization may also contribute in some tasks.

## 9. 수정 파일

- `paper/saber_aaai27_draft_kr.tex`
  - title
  - abstract
  - introduction
  - contribution list
  - limitations
  - conclusion

## 10. 남은 작업

- #7 related work framing을 원고 Related Work에 압축 반영한다.
- sample-level diagnostic 결과가 나오면 Abstract/Conclusion의 mechanism caveat를 업데이트한다.
- FPP Random-Support/Shuffled 결과가 나오면 limitation과 mechanism table을 갱신한다.

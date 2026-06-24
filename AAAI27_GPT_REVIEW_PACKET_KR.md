# SABER AAAI-27 GPT Review Packet

작성일: 2026-06-24

## 1. 리뷰 요청 목적

이 패킷은 SABER 논문을 AAAI-27 top-tier ML/RL reviewer 관점에서 비판적으로 점검하기 위한 것이다. 첨부된 최신 기준 원고는 `saber_aaai27_draft_kr.tex`와 이를 바탕으로 만든 `saber_aaai27_draft_kr_review.pdf`이다. PDF는 GPT 리뷰용 build이며, 최종 AAAI 제출 형식은 아니다.

원고는 아직 한국어 draft다. 기술 용어와 수식은 영어/LaTeX가 섞여 있다. 최종 제출 전 영어로 번역할 예정이다.

## 2. 논문 목표

SABER는 sparse-reward goal-conditioned HER actor-critic에서 HER가 만든 relabeled Bellman updates를 모두 동일하게 critic fitting하지 말고, local replay-action support가 있는 update를 critic loss에서 더 크게 반영하자는 방법이다.

핵심 thesis:

> HER creates training goals; SABER decides how strongly the critic should fit each resulting Bellman residual.

현재 가장 안전한 claim:

> SABER is a critic-side Bellman error reweighting objective that can improve environment-step sample efficiency in support-aligned sparse HER actor-critic settings.

## 3. 방법 요약

SABER는 각 training sample에 대해 다음을 수행한다.

1. HER 또는 baseline replay sampling으로 mini-batch를 가져온다.
2. Training goal `g_i^tr` 기준 local state-goal-horizon context를 만든다.
3. Replay buffer에서 같은 local context의 transition set을 찾는다.
4. Current policy action이 해당 local replay-action support 안에 있는지 확인한다.
5. Support-positive sample에는 raw weight `1 + lambda_S`를 준다.
6. Batch-normalized weight를 critic Bellman residual에만 곱한다.
7. Actor objective와 replay sampling distribution은 baseline과 동일하게 유지한다.

중요한 caveat:

- SABER는 HER replacement가 아니다.
- SABER는 prioritized replay가 아니다.
- SABER는 offline RL policy constraint가 아니다.
- Support predicate는 Bellman target noise의 직접 측정치가 아니라 `policy-improvement-relevant reliability proxy`다.

## 4. 현재 실험 요약

Primary evidence:

- FetchPickAndPlace, 1M steps, seeds 1--5
- DDPG+HER, TD3+HER, SAC+HER 모두에서 SABER가 final success와 success AUC를 개선

Secondary evidence:

- FetchSlide
- TD3/SAC에서는 개선
- DDPG에서는 실패 또는 악화

Stress-test:

- Meta-World DrawerOpen TD3
- Critic-only SABER가 HER baseline보다 좋아 보이지만 variance가 큼
- Random-Support control도 강하게 나와 structured support가 유일한 원인이라는 claim은 위험

Runtime:

- SABER는 support lookup overhead가 있다.
- Sample efficiency와 wall-clock efficiency를 분리해서 해석해야 한다.
- 현재 원고는 wall-clock efficiency를 주장하지 않는다.

## 5. 진행 중이거나 아직 부족한 부분

현재 진행 중:

- FPP DDPG Shuffled-Weight 1M seeds 1--5 실험 실행 중

아직 필요한 실험:

- FPP DDPG Random-Support 1M seeds 1--5
- FPP DDPG sample-level diagnostic run
- 가능하면 TD-error weighting full 1M rerun
- 가능하면 Random-Fixed-Count FPP port

이미 준비한 코드/계획:

- Sample-level diagnostic logger 구현
- `diagnostic_samples.csv` 분석 스크립트 구현
- Random/Shuffled control 실행 계획 정리

## 6. 리뷰에서 특히 봐줬으면 하는 질문

1. 현재 claim이 AAAI-27 top-tier reviewer 기준으로 과한가?
2. `hindsight Bellman target reliability gap`이라는 문제 정의가 충분히 설득력 있는가?
3. SABER가 PER/HGR/EBP/MHER/HGG/CHER와 다른 점이 명확한가?
4. Random-Support 결과가 논문 전체 claim을 얼마나 약화하는가?
5. FetchSlide DDPG failure를 limitation으로 처리하는 현재 방식이 충분한가?
6. Main result가 FPP 중심인 것이 약점으로 보이는가?
7. 어떤 missing baseline이 가장 치명적인가?
8. Method 수식에서 replay action critic residual과 current policy action support proxy의 관계가 논리적으로 방어 가능한가?
9. 현재 원고에서 reject risk가 가장 큰 부분은 무엇인가?
10. Accept 가능성을 올리기 위한 수정 우선순위는 무엇인가?

## 7. 리뷰어에게 요청하는 출력 형식

가능하면 다음 형식으로 리뷰해달라.

1. Overall verdict: accept 가능성 관점의 현재 단계
2. Top 5 rejection risks
3. Claim ledger: 각 핵심 claim이 evidence로 닫혔는지
4. Missing baselines / missing diagnostics
5. Related work novelty risk
6. Method definition risk
7. Experiment design risk
8. Concrete revision plan

아직 진행 중인 실험이 있다는 점을 감안하되, 현재 제출한다고 가정했을 때의 위험도와, 실험 결과가 추가됐을 때 어떻게 고쳐야 하는지를 구분해서 평가해달라.

## 8. 첨부 파일 안내

최신 원고:

- `saber_aaai27_draft_kr.tex`
- `saber_aaai27_draft_kr_review.pdf`

주의:

- `saber_aaai27_draft_kr_review.pdf`는 최신 `.tex`에서 만든 review-only PDF다.
- `saber_aaai27_draft_kr_previous_build.pdf`가 포함되어 있다면, 이는 최신 claim revision 전 빌드일 수 있으므로 첨부하지 않는 것을 권장한다.

보조 문서:

- `AAAI27_CLAIM_REVISION_KR.md`: 현재 claim 축소 기준
- `AAAI27_RESULT_MATRIX_KR.md`: 현재 artifact/result matrix
- `AAAI27_RELATED_WORK_NOTES_KR.md`: related work 비교 축
- `AAAI27_REPRODUCIBILITY_APPENDIX_KR.md`: 재현성 appendix 초안
- `AAAI27_SUPPORT_DIAGNOSTICS_KR.md`: 기존 artifact 기반 diagnostic 한계
- `AAAI27_RANDOM_CONTROL_PLAN_KR.md`: random/shuffled control 실험 계획

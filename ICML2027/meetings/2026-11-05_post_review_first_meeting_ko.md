# SABER 리뷰 이후 첫 미팅 자료

- 작성일: **2026-11-05**. GitHub 공개 자료와 이 PC의 완료 결과를 함께 확인했다.
- 목적: AAAI 리뷰 이후 **무엇을 했고, 무엇을 알게 됐으며, 다음에 무엇을 할지** 공동으로 결정한다.
- 대상: 2026-09-25 결정 이후 추가 실험·진단. 원 제출 실험은 비교에 필요한 부분만 포함한다.
- 읽는 순서: 시간이 짧으면 **1절 → 3절 → 4절 → 14절 → 15절**만 먼저 읽는다. 세부 수치와 출처는 뒤에서 확인할 수 있다.

## 1. 미팅에서 먼저 공유할 핵심 다섯 가지

1. **리뷰어들은 방법의 개입 위치와 실험 통제는 인정했다.** 다만 “가중치를 다르게 주는 것은 기존에도 있는데, 왜 SABER의 support 기준이어야 하는가?”와 “설명이 너무 복잡하다”는 문제가 남았다.
2. **리뷰 이후 실험은 상당히 보강됐다.** 공정한 가중치 배정 대조군, PAL, 검색 방식 비교, seed 확장, DrawerOpen/Close SAC, 정책 가치 예측 진단, 가중치 강도 민감도, 실패 분석까지 진행했다.
3. **추가 결과가 모두 SABER에 유리한 것은 아니다.** 처음 10개 seed에서 보인 배정 비교의 평균 우위는 11–20에서 반전됐다. SAC에서도 폭넓은 개선을 확인하지 못했고, PER가 더 좋은 결과를 보였다.
4. **이제 중요한 질문은 “support가 어떤 조건에서 유용한가?”다.** Support를 더 많이 찾는 것이 좋은 학습으로 이어지지는 않았고, 학습 배치의 오차를 줄여도 정책이 사용하는 가치 예측은 나빠지는 사례가 있었다.
5. **다음 작업은 이미 모은 실패 경로를 분석하고, 같은 저장 상태에서 배정 방식만 바꾸는 직접 비교를 보강하는 것이다.** 공통 context 전이와 남은 민감도도 필요하다. 이미 완료한 SAC·PAL·λ 실험을 처음부터 다시 계획할 단계는 아니다.

오늘 합의할 핵심은 **논문에서 유지할 주장, 우선 검증할 가설, 서로 겹치지 않는 다음 실험 담당**이다. SABER에 유리한 결과만 모으기보다, 현재 결과에 맞게 방법과 주장을 발전시키는 방향으로 논의한다.

## 2. 먼저 용어를 쉽게 맞추기

SABER는 로봇이 과거에 모은 경험을 공부할 때 **어떤 경험의 예측 오차를 더 비중 있게 학습할지** 정하는 방법이다. 비슷한 상황의 replay에 현재 정책이 고를 행동과 맞는 행동이 있으면 더 큰 가중치를 준다.

여기에는 서로 다른 두 행동이 등장한다. **Support는 지금 정책이 고를 행동으로 조회하지만, critic이 실제로 공부하는 정답은 과거 replay 행동의 transition으로 만든다.** 이 둘 사이에서 학습의 이득이 전달되는지가 중요한 질문이다.

| 용어 | 쉬운 설명 |
| --- | --- |
| Actor / 정책 | 지금 어떤 행동을 할지 정하는 모델 |
| Critic / Q | 이 상태에서 이 행동을 하면 앞으로 보상을 얼마나 얻을지 예측하는 모델 |
| Replay | 과거 경험을 저장한 학습 자료 |
| HER | 실패한 경험도 실제 도달한 목표 기준으로 다시 학습에 쓰는 방법 |
| Context | 비슷한 상태·목표·남은 시간 등을 묶는 기준 |
| Support | 그 문맥의 검사한 replay 후보 안에 현재 정책 행동을 뒷받침하는 기록이 있는지 나타내는 신호. 성공 가능성 자체는 아님 |
| Seed | 초기 모델·탐색 등의 난수를 바꿔 처음부터 학습한 한 번의 반복 실험 |
| Checkpoint | 특정 시점의 모델·optimizer·replay 등을 저장한 파일. 파일마다 환경 상태까지 포함하는지는 다름 |
| AUC | 학습 과정 전체의 성공률을 요약한 점수. 여기서는 보통 0–1이며 클수록 좋음 |
| Final | 마지막 학습 시점에 평가한 성공률. AUC와는 다른 지표 |
| 2M / 500k | 환경과 상호작용한 횟수 200만 / 50만 |

**AUC 0.65는 마지막 성공률 65%라는 뜻이 아니다.** 학습 도중 얼마나 빨리, 얼마나 꾸준히 성공했는지를 함께 반영한다. 2M 실험은 보통 10k–2M 구간이며, 뒤의 A1은 10k–500k 구간이다. 두 점수를 같은 길이의 실험처럼 비교하지 않는다.

### 신뢰구간에 0이 들어간다는 뜻

비교를 `SABER AUC − 비교 방법 AUC`로 정하면 양수는 SABER에 유리하고, 음수는 비교 방법에 유리하다.

예를 들어 처음 배정 비교의 `SABER − Random`은 평균 **+0.137**, 95% 신뢰구간은 **[−0.127, +0.384]**다. 관측 평균은 SABER가 높지만, 반복 학습의 변동을 고려하면 평균 효과의 방향을 충분히 좁히지 못했다는 뜻이다. **어떤 seed의 차이가 정확히 0이어서 문제가 생긴다는 뜻이 아니다.**

구간이 0을 포함해도 결과를 보고할 수 있다. 다만 그것만으로 우월성이나 동등성을 확정하지 않는다. 반대로 구간이 0을 벗어나도 모든 환경에서 우월하다는 뜻은 아니다. 이 자료의 bootstrap은 seed 행을 재표본하는 근사적 절차이며, 반복 횟수를 늘린다고 독립 학습 seed가 늘어나지는 않는다. 95%를 99%로 바꾸는 것이 해결책도 아니다.

## 3. AAAI 리뷰는 무엇을 말했나?

최종 결정은 **2026-09-25 Reject, Phase 2로 진출하지 못함**이다. 아래는 사용자가 전달한 OpenReview 리뷰에 근거한다. 원 제출 PDF와 전체 부록을 이번에 다시 감사한 문서는 아니다.

| 리뷰 | 인정한 점 | 핵심 문제를 쉽게 풀면 |
| --- | --- | --- |
| 사람 리뷰어 wpgC: **4점**, confidence 3 | Critic 가중치만 바꿔 효과를 분리한 점, 경쟁력 있는 일부 결과 | “가중치 조정은 흔하다. 왜 이 support 규칙이 좋은지 근거가 부족하다. 간단한 방법을 너무 어렵게 설명했다.” |
| 사람 리뷰어 mK8N: **6점**, confidence 3 | 방법은 타당하고 큰 실험 결함을 찾지 못함. 대조군·재현 자료도 긍정적 | “다른 환경에서도 쉽게 쓸 수 있나? 수작업 context와 설정값에 얼마나 민감한가? 왜 Tail로 조회하나? 표와 설명을 더 쉽게 만들자.” |
| AI 리뷰: 별도 참고 | 통제된 개입, 평균 가중치 정규화, 여러 진단 | “적은 seed와 넓은 구간, 정확한 support 위치의 기여, 가까운 선행방법, 제한된 환경 범위, 조회 근사와 시간 비용을 더 확인하자.” |

**두 사람 리뷰어의 공통 지적은 가독성**이다. 한 리뷰어가 이론적·원리적 근거를 요구했다고 해서 반드시 거대한 수렴 정리가 필요하다는 뜻은 아니다. 현재 방법이 왜·언제 유효한지 더 직접적인 실험과 명확한 설명이 필요하다는 것으로 받아들일 수 있다. 어느 리뷰가 최종 탈락을 결정했는지는 이 기록만으로 알 수 없다.

## 4. 리뷰 이후 무엇을 했나: 전체 지도

“원격 공유”는 GitHub main에 공개된 자료, “이 PC”는 로컬 및 work 브랜치 자료를 뜻한다. 실제 담당자 이름은 문서 출처만으로 추정하지 않았다.

| 작업 | 왜 했나? | 최신 확인 상태 | 핵심 결과 |
| --- | --- | --- | --- |
| Support / Random / TD 배정 비교 | 높은 가중치를 **어디에** 주는지가 중요한가? | 원격 1–10 완료, 이 PC 11–20 완료 | 처음에는 SABER 평균 우위, 추가 seed에서는 반전 |
| PAL 비교 | 가까운 기존 loss 변경 방법보다 좋은가? | 원격 10 seeds 완료 | SABER 평균은 높지만 차이의 CI는 0 포함 |
| Tail / Random-512 조회 | 후보를 고르는 방식이 support와 성능을 바꾸나? | 원격 온라인 10 seeds 및 양쪽 checkpoint 분석 완료 | Random은 support를 덜 놓치지만 평균 학습 성능은 Tail보다 낮음 |
| 원 논문 DrawerOpen–DDPG seed 확장 | 기존 5-seed 결과가 추가 반복에서도 유지되나? | 원격 8방법 × 새 5 seeds, **40회 완료** | 합친 평균은 SABER가 HER보다 높지만 새 6–10만 보면 방향이 다름 |
| DrawerOpen / DrawerClose SAC | 다른 알고리즘에서도 유효한가? | 원격 4방법 × 5 seeds × 2 tasks, **40회 완료** | SABER의 폭넓은 개선 없음. PER가 더 좋은 결과 |
| Policy-query 가치 진단 | 정책이 사용하는 Q 예측이 실제로 정확해지나? | 이 PC 초기·250k·성숙 정책 진단 완료 | 시점별 결과가 다르고 일관된 SABER 우위 없음 |
| Optimizer / Q 오차 점검 | 가치 진단의 나쁜 결과가 단순 구현·Adam 문제인가? | 이 PC 사후 진단 완료 | 모든 결과를 하나의 optimizer 문제로 설명할 수 없음 |
| 가중 강도 λ 민감도 | 특정 강도에서만 잘 되는가? | 이 PC 3값 × 3 seeds 완료 | λ=1 평균이 높음. 작은 탐색 실험 |
| 소스 재감사·실패 분석 | 새 seed의 나쁜 결과가 코드 변경 때문인가? 어떤 실패인가? | 이 PC 감사 완료 | 핵심 구현 동일. 실패 정책의 행동 포화 관측, 원인 미확정 |
| A1 초기 학습 경로 관찰 | 행동 포화가 언제 나타나나? | 이 PC 4방법 × 3 seeds × 500k, **12회 완료** | SABER 한 실행에서 일시적 전체 포화 관측. 영구 정지로 해석 불가 |
| DrawerOpen–TD3 seed 확장 | TD3의 기존 5-seed 결과도 확장 | **11/4 공개 기록상 진행·대기**, 오늘 상태 미확인 | 최종 결과 아직 확인할 수 없음 |
| 공통 context 전이 / 추가 설정 민감도 | 환경별 수작업과 설정 의존성을 줄일 수 있나? | 설계 존재, 완료 결과 미확인 | 다음 연구 과제 |

10월 6일 보고서의 “Random-512 진행 중”과 과거 대화의 “SAC 코드 설계 중”은 이제 지난 상태다. **같은 SABER 기준군을 여러 비교에 재사용했으므로 이 표의 숫자를 모두 독립 실험으로 더하지 않는다.**

## 5. 정확한 support 위치가 중요한가: 배정 비교와 seed 확장

### 무엇을 바꿨나?

각 실행의 미니배치 안에서 후보 조건과 가중치 분포를 맞추고, 높은 비중을 줄 위치를 바꿨다.

| 방법 | 높은 가중치를 주는 위치 |
| --- | --- |
| SABER | 현재 정책 행동의 support가 확인되는 위치 |
| Random assignment | 주변 데이터가 충분한 위치들 중 무작위로 선택 |
| TD-error Top-k | 같은 후보들 중 현재 예측 오차가 큰 위치부터 선택 |

각 업데이트에서 SABER가 정한 개수만큼 선택하고 큰/작은 가중치 비율은 2:1, 평균 가중치는 1로 맞춘다. 다만 각 온라인 실행은 자기 정책과 replay로 개수를 계산하므로 **서로 다른 실행의 자료와 선택 개수가 끝까지 같은 실험은 아니다.** Random/TD도 후보 조건과 개수를 통해 support 정보를 일부 사용한다. TD Top-k는 TD3 알고리즘과 다르며, 여기서는 모두 DDPG다.

### 결과: 모두 DrawerOpen–DDPG, 실행당 2M

| Seed 집합 | SABER AUC | Random AUC | TD Top-k AUC | Hit90 도달: SABER / Random / TD |
| --- | ---: | ---: | ---: | --- |
| 처음 1–10 | 0.6526 | 0.5157 | 0.5346 | 9/10 · 7/10 · 7/10 |
| 추가 11–20 | 0.3088 | 0.5374 | 0.6093 | 4/10 · 8/10 · 8/10 |
| 1–20 합산 | 0.4807 | 0.5265 | 0.5720 | 13/20 · 15/20 · 15/20 |

Hit90은 평가 성공률이 한 번이라도 90%에 도달한 실행 수다. 이후 성능 유지까지 보장하지는 않는다.

| 비교 | 처음 1–10: 차이 [95% CI] | 추가 11–20 | 1–20 합산 |
| --- | --- | --- | --- |
| SABER − Random | +0.1369 [−0.1275, +0.3843] | −0.2286 [−0.5469, +0.1046] | −0.0458 [−0.2503, +0.1625] |
| SABER − TD | +0.1180 [−0.0549, +0.3249] | **−0.3006 [−0.5302, −0.0838]** | −0.0913 [−0.2374, +0.0513] |

**무엇을 알았나?** 처음의 평균 우위는 추가 seed에서 유지되지 않았다. 추가 11–20의 TD 대비 차이는 구간 전체가 음수이며, 해당 분석의 두 비교 보정 구간에서도 음수다. “seed를 늘려 SABER의 우위를 확인했다”라고 쓸 수 없다. 합산 결과 역시 정확한 support 위치의 우위를 확립하지 못한다.

추가 SABER의 최종 실패가 6/10으로 많았다는 점이 평균 하락에 크게 반영된다. 실패를 뺀 평균으로 방법 순위를 바꾸면 안 된다. 과거 실제 실행 환경 기록이 완전하지 않아 두 집합의 차이를 **seed 효과만**으로 단정할 수도 없다. 이에 대한 소스 감사는 11절에 정리했다.

근거: [처음 배정 비교][R3], [추가 11–20 및 합산 보고서][L1]. 추가 30회의 원시 평가 CSV에서 200개 평가 지점과 AUC를 재확인했으며 저장 결과와 일치했다.

## 6. 기존 방법 PAL과 비교하면 어떤가?

**실험 이유:** “Replay는 균일하게 뽑되 critic loss를 바꾼다”는 가까운 선행방법과 직접 비교하기 위해서다. PAL은 replay sampling을 바꾸는 PER와 다르다.

DrawerOpen–DDPG, PAL 10개 seed를 새로 2M까지 학습했다. SABER는 5절의 처음 10회 결과를 재사용했다.

| 방법 | AUC 평균 | Final 평균 |
| --- | ---: | ---: |
| SABER | 0.6526 | 0.890 |
| HER + PAL | 0.5454 | 0.800 |

`SABER − PAL = +0.1073`, 95% CI **[−0.1531, +0.3577]**다.

**무엇을 알았나?** 가까운 선행방법을 비교하지 않았다는 공백은 보완됐다. 평균은 SABER가 높지만 확정적인 우위는 아니다. 이 실험은 PAL을 DDPG+HER에 적용한 비교이며, 원 PAL 논문의 TD3 benchmark를 그대로 재현한 것은 아니다. 성능표 하나가 신규성 자체를 입증하지도 않는다.

근거: [PAL 결과·구현 범위][R4].

## 7. 조회를 더 정확하게 하면 학습도 좋아지는가?

**실험 이유:** 리뷰어가 “왜 index 끝부분의 후보를 쓰는가?”라고 물었다. 제한된 조회가 전체 replay의 support를 얼마나 놓치는지, 후보 선택을 바꾸면 학습도 달라지는지 확인했다.

| 방식 | 후보 선택 |
| --- | --- |
| Tail-512 | 같은 coarse bucket의 유지된 index 순서 끝부분에서 최대 512개 |
| Random-512 | 같은 bucket에서 최대 512개를 무작위 비복원 선택 |
| Full | 같은 bucket의 모든 유효 후보를 조사 |

Tail 순서는 시간적으로 가장 최근이라는 뜻이 아니다. Full도 **현재 이산 context 정의의 전수 조회 기준**이며 연속 공간의 참 support를 아는 기준은 아니다. 여기서 Random은 조회 후보 선택이고, 5절 Random assignment는 큰 가중치를 받을 위치 선택이다.

### 온라인 10-seed 학습 결과: 이제 완료됨

| 방식 | AUC 평균 | Final 평균 |
| --- | ---: | ---: |
| Tail-512 | 0.6526 | 0.890 |
| Random-512 | 0.4302 | 0.590 |

사전에 정한 방향인 `Random-512 − Tail = −0.2224`, 95% CI **[−0.5184, +0.0809]**다. Tail 기준은 처음 SABER 10회를 재사용했다. 평균은 Tail이 높지만 Tail의 보편적 우위나 최적성까지 입증하지는 않는다.

### 양쪽 학습에서 나온 최종 checkpoint를 모두 조사한 결과

아래는 **Full이 positive로 판정한 query 중 놓친 비율**이다. 최종 보고서의 seed별 비율을 seed 동일 가중치로 평균했다. 낮을수록 이 기준의 support를 덜 놓쳤다.

| Checkpoint를 만든 학습 | Tail 조회 | Random 조회 |
| --- | ---: | ---: |
| Tail로 학습한 10개 | 41.87% | 31.48% |
| Random-512로 학습한 10개 | 25.35% | 19.22% |

**무엇을 알았나?** 두 replay 분포 모두에서 Random이 support를 덜 놓친다. 그러나 online 평균 성능이 좋아진 것은 아니다. **“Support를 더 많이 확인한다”와 “더 유용하게 학습 비중을 나눈다”는 다른 질문**이다. 이 후속 조회는 초기 10월 진단과 query 표본이 달라 수치가 조금 다르다.

추가로 처음 배정 비교 30회에서 평가 성공률이 계속 0이었던 7회는 모두 나중에 support 배정이 포화됐다. 나머지 23회에는 없었다. 다만 처음 500k에는 모두 포화가 없어, 초기 실패 예측이나 인과관계의 증거로 쓸 수는 없다.

근거: [조회·실패 진단][R5], [완료된 온라인 비교][R6], [양쪽 조회 원시 집계 보고서][R11].

## 8. 원 논문의 DrawerOpen–DDPG 결과를 더 반복하면?

**실험 이유:** 원 제출의 5개 seed에 새로운 5개를 추가해 일반 성능과 query 대조군을 확인했다. 8방법 × 새 seeds 6–10 × 2M, **새 40회가 모두 완료**됐다. 이 부분은 5절의 별도 assignment cohort와 구분한다.

| 방법 | 원 제출 1–5 AUC | 새 6–10 AUC | 합산 1–10 AUC | 합산 Final |
| --- | ---: | ---: | ---: | ---: |
| HER | 0.2073 | 0.5948 | 0.4011 | 0.595 |
| SABER | 0.7537 | 0.5501 | 0.6519 | 0.895 |
| HGR | 0.6882 | 0.4416 | 0.5649 | 0.750 |
| PER | 0.8251 | 0.4758 | 0.6504 | 0.795 |
| Replay-Action Query | 0.3178 | 0.5480 | 0.4329 | 0.600 |
| Shuffled-Policy Query | 0.6483 | 0.5973 | 0.6228 | 0.895 |
| Random-Support | 0.7065 | 0.4257 | 0.5661 | 0.800 |
| Context-Eligible Count-Matched | 0.6051 | 0.4263 | 0.5157 | 0.700 |

Replay-Action Query는 현재 정책 행동 대신 저장 행동으로 조회한다. Shuffled는 문맥과 정책 행동의 연결을 섞는다. Random-Support는 전체 배치 안에서, Context-Eligible는 주변 데이터가 충분한 후보 안에서 높은 가중치 위치를 무작위화한다. HGR/PER는 기존 replay 관련 비교 방법이다.

현재 확장 분석에서 산출한 대표 차이와 구간을 보면:

| 비교 | AUC 차이 [95% CI] |
| --- | --- |
| 원 제출 1–5의 SABER − HER | +0.5464 [+0.3399, +0.7410] |
| 새 6–10의 SABER − HER | −0.0447 [−0.4781, +0.4561] |
| 합산 1–10의 SABER − HER | +0.2508 [−0.0676, +0.5439] |
| 합산 1–10의 SABER − PER | +0.0014 [−0.1050, +0.1487] |

합산 기준 SABER와 나머지 7방법 차이의 95% 구간은 모두 0을 포함한다. 이 구간들은 비교별 구간이며 전체 비교를 한꺼번에 보장하는 구간은 아니다.

**무엇을 알았나?** 기존 HER 비교는 이미 있고 추가 반복도 확보했다. 다만 새 seed에서 원래의 큰 개선량이 유지되지는 않았다. 따라서 “HER 기준군이 없으니 무조건 새로 10회”라는 제안은 중복일 수 있다. 새로운 protocol에 재사용하려면 설정·실행 환경·평가 규칙을 먼저 맞춰 확인한다.

주의: 여기의 SABER AUC **0.6519**와 5절의 **0.6526**은 다른 실행 집합이다. 숫자가 비슷하다고 한 기준군으로 바꾸어 쓰지 않는다. 일부 대조군 수치가 다시 등장해도 독립 재현 횟수로 더하지 않는다. 과거 실행 환경 전체가 복구된 것은 아니므로 역사적 실행과 새 실행을 구분해 보고한다.

근거: [원 논문 DDPG 8방법 seed 확장][R7].

## 9. 다른 backbone 확장: DrawerOpen / DrawerClose SAC

**실험 이유:** 기존 최종 논문에 비어 있던 Drawer SAC를 채우고, 다른 backbone에서도 SABER가 도움이 되는지 확인했다. 현재는 두 task 모두 HER/SABER/HGR/PER, 방법당 5 seeds, 2M 결과가 있다. **총 40개 고유 실행**이며 HER/SABER 결과를 HGR/PER 비교에 재사용했다.

HGR/PER는 HER/SABER 결과를 본 뒤 추가한 탐색적 비교다. 네 방법을 모두 선행 결과 없이 동시에 확정한 실험으로 설명하지 않는다.

| Task | 방법 | AUC 평균 | Final 평균 |
| --- | --- | ---: | ---: |
| DrawerOpen | HER | 0.4470 | 0.600 |
| DrawerOpen | SABER | 0.4485 | 0.600 |
| DrawerOpen | HGR | 0.7066 | 1.000 |
| DrawerOpen | PER | 0.7976 | 1.000 |
| DrawerClose | HER | 0.2606 | 0.200 |
| DrawerClose | SABER | 0.1033 | 0.000 |
| DrawerClose | HGR | 0.2118 | 0.030 |
| DrawerClose | PER | 0.4873 | 0.550 |

| 비교 | DrawerOpen: 차이 [95% CI] | DrawerClose: 차이 [95% CI] |
| --- | --- | --- |
| SABER − HER | +0.0015 [−0.0466, +0.0413] | −0.1573 [−0.4191, +0.0250] |
| SABER − PER | **−0.3491 [−0.6747, −0.0235]** | **−0.3840 [−0.6239, −0.1440]** |

**무엇을 알았나?** Open은 HER와 SABER 평균이 거의 같다. Close는 SABER의 최종 평가가 모든 seed에서 0이고 마지막 1/4 구간 AUC도 0.0176으로 낮다. 두 task에서 PER 대비 차이는 음수이며, 원 분석의 다중 비교 보정 구간에서도 음수다. 특히 Open의 보정 구간 상한은 약 −0.0001로 0에 매우 가깝다. 작은 5-seed의 근사 구간이라는 한계는 유지하되 불리한 결과를 숨기지 않아야 한다.

DrawerClose는 네 방법 모두 각 seed에서 한 번은 Hit90에 도달했다. 따라서 **“한 번 성공했다”와 “끝까지 잘 유지했다”를 나누어 봐야 한다.** 여기의 성공률은 평가 episode 중 한 번이라도 성공한 비율이지, episode 마지막 상태에서 목표를 유지한 비율은 아니다.

이 구현은 FPP 기준의 자동 entropy-temperature SAC다. 예전 고정 temperature prototype와 섞지 않는다. Support query에는 squashed policy mean을 쓴다. 이 사실과 DDPG 실패 관측만으로 SAC 실패 원인을 같은 것으로 단정하지 않는다.

근거: [DrawerOpen SAC][R8], [DrawerClose SAC][R9].

## 10. 이 PC에서 한 작은 실험: 정책에 필요한 Q 예측이 좋아지는가?

### 실험의 목적과 방법

**“공부한 문제의 오차가 줄었다”와 “정책이 실제로 묻는 질문에 더 정확히 답한다”를 구분하려고 했다.** 같은 checkpoint의 critic과 optimizer를 복사하고 정책·target·학습 자료를 고정한 뒤, HER/SABER/Random/TD로 critic만 짧게 학습했다. 별도 상태에서 정책을 실제로 실행해 얻은 할인 누적보상을 Q 예측의 기준으로 썼다.

| 단계 | 무엇을 했나? | 결과와 역할 |
| --- | --- | --- |
| 저장·복원 예비 검증 | 시뮬레이터 복원, 모델·Adam 적재, 원 업데이트·조회 일치 확인 | 비교 실험이 같은 조건에서 동작하는지 확인한 준비 작업 |
| HER seed 1, 20k | 새로 학습한 초기 저장본에서 네 방식 비교 | SABER 1회 업데이트 후 Q MSE +17.2792. 평가 return이 모두 같은 실패 상황이라 정보가 제한적 |
| HER seed 1, 250k | 250k까지 별도로 새 학습 후 진단 | SABER 후 MSE +1.5291. 역시 같은 return에 치우친 평가 |
| 성숙 정책 진단 | HER/SABER 각각 seed 1을 2M까지 새 학습. 500k/1M/1.5M/2M, 총 8개 checkpoint | 시점별 효과가 달랐으며 일관된 SABER 우위 없음 |
| Adam 감사 | 초기 두 저장본의 optimizer 상태·이동 크기를 바꾸어 복제 진단 | 일부 초기 현상에 update 크기가 관여하지만 모든 Q 악화를 설명하지 못함 |
| 2M Q 오차 상세 분석 | 큰 오차가 나는 상태와 실제 rollout 확인 | 배치 오차 감소와 정책 Q 오차 증가가 동시에 가능함을 확인 |

성숙 정책 진단은 checkpoint마다 2개 episode의 4개 시점, 총 8개 query를 사용했다. Query당 1,024-step rollout으로 학습의 timeout bootstrap 해석에 맞는 continuing return을 근사했다. **학습 seed는 원 방법당 하나**이므로 checkpoint나 query 개수를 독립 학습 반복으로 세지 않는다. 주 진단은 독립적으로 복원한 1-step batch 두 개이며, 추가 순차 진단은 2-step이다.

SABER로 학습한 저장본 네 시점에서 **한 번 업데이트 후 Q MSE 변화**는 다음과 같다. 각 값은 독립적으로 복원해 한 번씩 업데이트한 두 배치의 평균이다. 음수는 개선, 양수는 악화다. 표의 HER 열은 저장본을 새로 HER로 학습했다는 뜻이 아니라, 같은 critic 복사본에 균일 가중치를 적용했다는 뜻이다.

| 원 저장 시점 | HER 배정 | SABER 배정 | Random 배정 | TD 배정 |
| --- | ---: | ---: | ---: | ---: |
| 500k | +0.0313 | +0.0325 | +0.0283 | +0.0314 |
| 1M | −0.3441 | −0.3507 | −0.3143 | −0.3152 |
| 1.5M | −0.1009 | −0.0590 | −0.1116 | −0.1022 |
| 2M | +3.2592 | +3.5726 | +2.7895 | +3.3497 |

1M에서는 SABER의 개선이 조금 더 크지만 다른 시점에서는 그렇지 않다. HER로 학습한 저장본 네 개도 진단했으며, 일부에는 배정 방식이 실제로 달라지는 batch가 없었다.

### 가장 직관적인 사례

SABER로 학습한 2M 저장본에서 SABER 가중치로 critic을 한 번 더 학습했을 때, 위와 같은 두 독립 배치 평균을 보면:

| 측정 | 전 → 후 | 뜻 |
| --- | --- | --- |
| 학습 배치의 Bellman residual MSE | **1.2103 → 1.1056** | 방금 공부한 자료의 오차는 감소 |
| 별도 policy-action의 Q MSE | **183.2572 → 186.8298** | 정책이 사용할 예측은 이 진단에서 악화 |

8개 query 중 오차가 큰 두 개가 시작 MSE의 81.22%를 차지했다. 이들을 빼고 결론을 바꾸지 않았다. 한 episode에서 한 번 성공했더라도 나중에 목표에서 이탈하면 continuing return은 낮아질 수 있으므로, 높은 성공률과 큰 Q 오차가 함께 나타날 수 있다.

**무엇을 알았나?** “SABER가 residual을 줄이므로 정책 Q도 정확해진다”는 연결을 자동으로 주장할 수 없다. 그렇다고 이 작은 단기 진단 하나로 장기 online 성능 전체를 부정할 수도 없다. 다음 보강은 초기·여러 seed·다양한 평가 목표에서 Q와 정책 업데이트 방향을 함께 확인하는 것이다.

근거: [20k][L2], [250k][L3], [성숙 정책 8-checkpoint 진단][L4], [20k optimizer 감사][L5], [250k optimizer 감사][L6], [2M Q 오차 분석][L7].

## 11. 왜 추가 SABER seed가 나빠졌나: 코드와 실패 경로 점검

### 코드가 달라졌는지 확인했다

기존 1–10과 추가 11–20의 **핵심 `code/` 26개와 `upstream/` 25개 파일은 byte-identical**이다. Loss·target·HER가 다른 구현으로 바뀌거나 방법 이름이 뒤바뀐 증거는 찾지 못했다. 새 실행 config도 원본 설정 함수로 재구성한 값과 일치했다.

단, 모든 것이 같다는 뜻은 아니다. Launcher·seed 허용 범위·검증 wrapper는 바뀌었다. 새 실행은 CPU이며, 과거 실제 device·패키지 기록은 공개 자료만으로 확정하지 못했다. 과거 기본값이 CUDA였다는 이유만으로 실제 GPU 실행이었다고 단정할 수 없다. 병렬 실행 수와 방법별 실행 순서도 달랐다.

**현재 판단:** “코드가 달라서 그렇다”는 설명을 뒷받침할 핵심 구현 변경은 발견되지 않았다. 반대로 “seed 외에는 완전히 같다”거나 “SABER와 무관한 불안정성이 확실하다”는 증명도 아니다.

### 최종 실패 정책에서 확인한 것

추가 SABER의 실패 6개 최종 모델에서는 검사한 상태의 행동이 한계값에 붙고 actor gradient가 0인 현상을 관측했다. 쉽게 말하면 **행동 출력을 더 바꾸기 어려운 상태**였다. Random과 TD 실패 일부에서도 같은 형태가 나왔다.

여기서 두 종류의 ‘포화’를 구분한다.

| 구분 | 뜻 | 이것만으로 알 수 없는 것 |
| --- | --- | --- |
| Support 배정 포화 | 주변 데이터가 충분한 후보를 모두 큰 가중치로 선택해 배정 방법 간 차이가 사라짐 | 이것이 실패의 원인인지, 실패한 replay의 결과인지 |
| Actor 행동 포화 | 출력이 행동 한계값에 붙어 tanh의 미분과 정책 gradient가 0 또는 매우 작아짐 | 언제 시작됐는지, 모든 상태에서 영구적인지, SABER가 위험을 바꿨는지 |

최종 실패 모습이 다른 방법에도 있다는 사실은 **SABER만의 전용 오류가 아닐 가능성**을 보여준다. 그러나 SABER가 그 경로에 더 자주 빠지게 했는지는 별도 문제다. “SABER의 책임이 없다”로 결론 내리지 않는다.

조회 구현 대조 18,432건에서는 불일치가 없었다. 기존 CSV의 일부 거리·phase 열이 마지막 HER minibatch 값을 담는 계측 문제는 확인했으나, 평가 성공률과 AUC가 바뀐 것은 아니다. 이후 실제 평가 경로는 전용 evaluation JSONL로 읽는다.

근거: [소스 재감사][L8], [실패 심층 감사][L9].

## 12. 가장 최근 완료한 A1: 실패가 만들어지는 과정을 관찰

**실험 이유:** 최종 저장본만 보면 언제 행동이 굳었는지 알 수 없었다. 초기부터 같은 일정으로 정책 출력·critic gradient·실제 optimizer 이동·support를 기록했다.

DrawerOpen–DDPG, HER/SABER/Random/TD × 새 seeds 101–103 × 500k, **12회 모두 완료**다. 학습 전 4,800-step 상태를 포함해 실행당 13개, 총 156개 checkpoint를 남겼다. 초기 경로를 보는 3-seed 예비 실험이며 2M 성능 확증 실험이 아니다.

| 방법 | 10k–500k AUC 평균 | 500k Final 평균 | 평가 내내 성공 0인 실행 |
| --- | ---: | ---: | ---: |
| HER | 0.1316 | 0.6667 | 1/3 |
| SABER | 0.0117 | 0.1167 | 1/3 |
| Random | 0.0723 | 0.2833 | 0/3 |
| TD Top-k | 0.1327 | 0.3333 | 1/3 |

SABER seed 102에서 고정 HER query 전체의 tanh 미분이 처음 0이 된 시점은 399k, 고정 원래 목표 query에서는 407k였다. 다른 11개 실행에서는 500k까지 이 **고정 query 전체 포화**가 관측되지 않았다. 12개 모두 같은 기간에 support 배정 포화는 없었다. 즉, 행동 포화가 반드시 support 배정 포화 뒤에만 나타나는 것은 아니다.

**이번 미팅 준비 중 원시 시계열에서 추가 확인한 중요한 점:** 399k/407k는 최초 관측 시점이다. 이후 다시 0이 아닌 gradient가 나타났다. 예를 들어 HER query는 406k에 이미 일부 미분과 gradient가 돌아왔고, 500k에는 두 query 집합 모두 actor gradient가 0이 아니었다. **“399k부터 영구적으로 학습이 멈췄다”라고 말하면 안 된다.** Gradient가 다시 생겼다는 사실이 성능 회복을 뜻하는 것도 아니다.

또한 A1의 고정 probe 기록에는 native 관측의 마지막 목표 좌표와 별도 HER 목표가 달라지는 현상이 담겨 있다. 네 방법이 공유하는 입력 표현의 점검 근거지만, 이 차이가 성능을 해쳤다는 인과 증거는 아니다.

**무엇을 알았나?** 실패 경로를 추적할 자료는 확보했다. 다음에는 순간적인 포화와 오래 유지되는 포화를 나누고, 그 직전 배정 방식이 critic과 actor의 업데이트를 어떻게 바꾸는지 직접 비교할 수 있다. A1 자체가 policy Q 정확도나 실패 원인을 입증한 실험은 아니다.

근거: [A1 공개 결과][L10], [seed 102 원시 probe 기록][L11], [목표 입력 점검 기록][L12].

## 13. 설정 민감도와 아직 진행 중인 일

### 가중치 강도 λ: 이미 작은 실험은 끝났다

λ는 support-positive 자료에 얼마나 더 큰 학습 비중을 줄지 정한다. DrawerOpen–DDPG, λ=0.5/1/2, 각 seeds 1–3, 2M 결과다. 9개 결과 칸 중 λ=1 seed 1은 10절의 성숙 SABER 학습을 재사용했으며 **새 학습은 8회**다.

| λ | AUC 평균 | Final 평균 | Hit90 |
| --- | ---: | ---: | ---: |
| 0.5 | 0.4780 | 0.6667 | 2/3 |
| 1: 기존 기본값 | 0.7452 | 1.0000 | 3/3 |
| 2 | 0.5380 | 0.6667 | 2/3 |

**현재 해석:** 이 작은 실험에서는 기존 기본값 1의 평균이 높다. 3개 seed, CI 없는 탐색 결과이므로 최적값이나 넓은 강건성을 확정할 수 없다. λ=0 HER arm은 이 실험에 없다. N_local, N_scan, context 구간까지 검증한 것은 아니다. [λ 결과][L13]

### DrawerOpen–TD3 확장: 오늘 상태를 동료에게 확인

11월 4일 GitHub 공개 기록에서는 HER/SABER/HGR/PER × seeds 6–10 × 2M, **20회 계획 중 HER 5회 실행, 나머지 15회 대기, 완료 0회**였다. **이것은 11월 4일의 기록이며 11월 5일 현재 프로세스 상태를 확인한 것은 아니다.** 현재 README의 기존 TD3 성능표를 새 10-seed 결과로 읽으면 안 된다. [최신 공개 상태][R1]

### 과거에 이미 있었던 근거도 재사용

K=3/5/7, FPP의 추가 seed·TD weighting·HER convention, 높은 update ratio 등의 제출 자료를 새 리뷰 대응 학습으로 다시 세지 않는다. Runtime 정보도 전혀 없는 것은 아니다. 예를 들어 기존 FPP–TD3 100k seed 1 profile에서 **20k 이후–100k 구간의 평가 시간을 제외한 training 시간**은 HER **645초**, SABER **1,931초**였다. 특정 구현·조건의 비용 측정이며, 모든 task의 비용이나 성능–시간 곡선을 대표하지 않는다. [기존 runtime profile][R10]

## 14. 그래서 리뷰에 대해 어디까지 답했나?

| 리뷰의 질문 | 보완된 사항 | 아직 필요한 답 |
| --- | --- | --- |
| 기존 reweighting과 무엇이 다른가? | PAL 직접 비교, 배정 강도·개수를 맞춘 대조군 | Support 신호의 독자적 유용성이 어떤 조건에서 나타나는지. 성능표 추가만으로 신규성 해결은 아님 |
| 정확한 policy-action support가 필요한가? | 배정 20-seed 자료와 query 대조군 확장 | 일관된 추가 가치 미확립. 같은 상태에서 직접 효과와 실패 경로 구분 필요 |
| 정책이 쓰는 critic을 더 정확하게 만드나? | 실제 rollout 기준 policy-query 진단 | 작은 범위에서 일관된 개선 없음. 초기·다중 seed·HER 평가 목표·정책 업데이트 연결 보강 |
| Tail 조회는 왜 쓰나? | 정적 누락률, Random 온라인 비교, 양쪽 replay 감사 | 더 많이 찾는 support가 왜 더 유용하지 않은지. Tail 최적성은 주장 불가 |
| seed·알고리즘 범위가 충분한가? | DDPG 추가 반복, Drawer SAC 완료 | 확장에서 약해진 우위·불리한 SAC 결과를 설명하고 주장 범위 조정 |
| 설정을 조금 바꾸면 무너지나? | λ 3값 pilot 완료 | 주요 population·scan·context 설정의 실제 민감도 |
| 환경마다 수작업이 필요한가? | 전이 설계 문서 존재 | 고정한 공통 context 규칙의 전이 결과 미확인 |
| 비용이 싼가? | 기존 runtime 및 새 실행 시간 기록 | 같은 계측·부하의 성능–wall-clock 비교 |
| 읽기 어려운가? | 표로 옮길 근거가 늘어남 | Method·그림·용어·주장 구조를 실제로 다시 써야 함 |

**보강의 성과는 대조군과 실패 근거를 더 정직하게 분리할 수 있게 됐다는 점이다.** 현재 자료 전체로 “support 기반 배정이 항상 더 좋다”거나 “여러 backbone에서 일관되게 우월하다”는 주장을 강화하기는 어렵다. 효과가 나는 조건과 나지 않는 조건을 밝히는 방향이 필요하다.

## 15. 다음에 할 일: 중복 실험을 피한 우선순위 제안

아래는 미팅에서 합의할 **새 제안**이다. 이 문서 작성으로 학습을 시작하거나 예약하지 않았다. 각 신규 실험의 설정·seed·종료점·주 비교는 실행 전에 고정한다.

### 우선 1: 이미 끝난 A1의 전체 시계열을 분석하고 직접 분기 비교 준비

먼저 추가 학습 없이, 포화가 잠깐 나타났다 사라지는지 오래 지속되는지 분석한다. 고정 query와 실제 평가 경로를 함께 보고, critic gradient·actor gradient·실제 Adam 이동이 어떤 순서로 달라지는지 정리한다. 최초 포화 시점 하나를 실패 시점으로 삼지 않는다.

다음으로 **같은 저장 상태를 복제해 가중치 배정만 바꾸는 A2 진단**을 한다. 쉽게 말하면, 같은 학생에게 같은 문제를 주고 공부 비중만 바꾼 뒤 예측과 행동 수정 방향이 어떻게 달라지는지 보는 것이다.

- Critic·optimizer 복사, actor·target·batch·HER 목표 고정, 균일/SABER/Random/TD 비교.
- 배정별 gradient 방향과 실제 parameter 이동량, 별도 policy-query Q 오차를 함께 측정.
- 기존 10절 진단의 반복 범위를 초기 상태와 여러 seed로 넓혀, 어느 상황에서 차이가 생기는지 확인.
- 모든 방법에서 얻은 미리 정한 저장 시점을 포함하고 유리한 checkpoint만 선택하지 않음.

A1 checkpoint에는 학습 상태가 있지만 **시뮬레이터 내부 상태까지 저장한 것은 아니다.** 관측 벡터만으로 실제 rollout을 정확히 재개할 수 있다고 가정하지 않는다. Q 정답 평가에는 기존 검증된 상태 복원 방식이나 새 독립 rollout bank를 사용한다. Replay 행동의 reward·next state를 다른 정책 행동의 정답으로 그대로 쓰지 않는다. 단기 직접 효과를 보여도 장기 실패 확률의 원인까지 바로 증명하지는 않는다.

### 우선 2: 공유하는 목표 입력 표현을 점검하고, 필요하면 통제 실험

MetaWorld 원 관측에 원래 목표가 남아 있는데 별도 HER 목표만 바뀌는 입력을 확인할 필요가 있다. A1 probe에는 두 목표가 달라지는 기록이 이미 있다. **아직 확인할 것은 그것이 학습에 어떤 영향을 주는지**다.

입력 의미를 먼저 확인한 뒤 `기존 표현 / 원 목표 성분을 일관되게 제거한 표현` × `HER / SABER`처럼 비교할 수 있다. Actor·critic·target·support·평가 모두 같은 표현 규칙을 쓰도록 한다. 이것은 새로운 변형의 실험이며 기존 결과를 지우는 수정으로 취급하지 않는다. SAC의 나쁜 결과가 이 표현 때문이라고 미리 결론 내리지 않는다.

### 우선 3: 공통 context 규칙의 전이

“새 환경마다 사람이 support 기준을 다시 만들어야 하는가?”에 답하는 실험이다. HER / 기존 SABER / 단순 공통 context SABER를 비교하고, 개발 task에서 정한 규칙을 다른 task 결과를 보고 다시 바꾸지 않는다. FPP와 Drawer처럼 구조가 다른 task를 활용할 수 있다.

이것은 **context 설계 규칙의 전이**다. 새 환경에서 정책을 다시 학습하므로 정책의 zero-shot generalization이라고 부르지 않는다. 필요한 관측 매핑과 수작업도 함께 기록한다. [기존 전이 설계][L14]

### 그다음: 남은 민감도·다른 backbone의 배정 검증·시간 비용

| 후보 | 이번에 새로 답할 질문 | 먼저 할 최소 작업 |
| --- | --- | --- |
| N_local / context 구간 | 주변 자료 수·수동 경계가 결과를 좌우하나? | 요청한 설정이 실제 판정 조건을 바꾸는지 먼저 확인 |
| N_scan | 검색량·누락률·성능·비용 관계는? | 같은 checkpoint에서 후보 수를 바꾸어 진단 후 필요한 online 조건만 선정 |
| TD3 mechanism control | DDPG 외에서도 정확한 배정 위치가 중요한가? | 진행 중 일반 TD3 확장 상태 확인 후 대표 조건에 엄격한 대조군 추가 검토 |
| 성능–wall-clock | 실제 사용 시간까지 고려해 이득이 있나? | 같은 하드웨어·부하·계측의 HER/SABER/PAL 비교. Index 유지 비용도 포함 |
| 실행 환경 대조 | 과거와 새 cohort의 차이에 환경이 관여했나? | 과거 실제 config·package·device 기록 확보. 확보되면 같은 seed 대조 설계 |

N_local은 다른 최소 조건과 함께 작동한다. 현재 M_Q_min=4이면 N_local=1과 4가 같은 판정 조건이 될 수 있다. 또한 현재 DrawerOpen의 frozen config에는 N_local_min 요청값 8을 4로 변환하는 경로가 있다. 새 민감도 코드는 요청값과 실제 적용값을 확인해야 한다. **CLI 숫자만 바꾼 것을 민감도 실험으로 세지 않는다.** [설정 점검 근거와 기존 후속 설계][L15]

지금 바로 21–30 seed를 더 돌리는 것을 최우선으로 권하지 않는다. 더 정밀한 성능 추정이 필요하면 의미 있는 효과 크기·원하는 구간 폭·비용을 정해 계획한다. 유리한 구간이 나올 때까지 반복하는 방식은 피한다. 이미 완료한 PAL·λ pilot·Random-512·SAC를 미실행으로 착각해 중복 예약하지 않는다.

## 16. 학습과 별개로 바로 할 원고 작업

1. **기여 문장 수정:** 일반적인 loss reweighting 자체보다 구체적인 support 신호와 그 적용 조건에 초점을 둔다. 아직 성립하지 않은 설명을 확정적으로 쓰지 않는다.
2. **Method 단순화:** 작은 예시 → support 조건 → 평균 1 가중치 목적함수 → 짧은 pseudocode → task별 context 표 순으로 설명한다.
3. **두 행동 구분:** 현재 정책 행동은 조회용, replay 행동은 Bellman 회귀용이라는 것을 한 그림에 분명히 표시한다.
4. **결과 배치:** 가장 가까운 PAL 비교, 엄격한 배정 대조군, SAC의 한계, 전이 결과가 있다면 그 핵심을 본문에서 읽을 수 있게 한다. 진단 수치의 긴 문장은 표로 바꾼다.
5. **통계 표현 통일:** 평균·seed 분포·CI·실패 실행을 함께 보고한다. 기존/추가/합산 cohort, 500k/2M, 같은 기준군 재사용을 명시한다.
6. **선행연구 위치 정리:** PAL과 actor-aware replay의 관계를 정확히 설명한다. 관련 연구 비교와 메커니즘 대조군의 역할을 구분한다.

## 17. 오늘 미팅 진행안과 결정 기록

| 시간 예시 | 논의 | 회의가 끝날 때 남길 것 |
| --- | --- | --- |
| 5분 | 리뷰의 공통 문제와 현재 핵심 결론 | 다음 원고의 중심 질문 한 문장 |
| 10분 | 배정 seed 반전, DDPG 확장, SAC 결과 | 유지할 주장 / 줄일 주장 |
| 10분 | Q 진단과 실패·A1 기록 | 다음 직접 검증할 가설 1–2개 |
| 10분 | A2·입력 표현·공통 context·진행 중 TD3 | 담당, protocol 작성일, 예산과 종료 조건 |

회의에서 채울 항목:

- [ ] 11/4에 진행 중이던 TD3 20회는 오늘 어디까지 완료됐는가?
- [ ] 다른 PC에서 끝났지만 아직 push하지 않은 실험이 있는가?
- [ ] SABER가 유효하다고 주장할 task/backbone/조건의 범위를 어떻게 정할 것인가?
- [ ] A1 추가 분석과 A2의 담당은 누구이며, 어떤 checkpoint·측정을 우선할 것인가?
- [ ] 목표 입력 표현 변경을 별도 변형으로 실험할 것인가?
- [ ] 공통 context 개발 task와 고정 후 평가 task는 무엇인가?
- [ ] main / work 브랜치와 로컬 전용 결과 중 무엇을 공동 archive로 묶을 것인가?
- [ ] Method 재작성·결과표·관련 연구 정리 담당과 다음 미팅 날짜는 언제인가?

## 18. 자료 범위와 출처

### 확인 범위

- GitHub 확인 기준: `main` **332c2246d05ac05548335169b3f482175af8e4ce**, `work/icml2027` **c1b94362dde8228cbf7ee70442ec93c109c95112**. 원격을 갱신해 읽었으며 두 브랜치를 임의로 합치지 않았다.
- 다른 공개 원격 브랜치도 확인했으나 이번에 확인한 리뷰 이후 신규 결과는 main과 work의 자료에 해당했다. 아직 push되지 않은 동료 PC의 결과는 포함하지 못한다.
- 원격 archive는 자체 범위에서 새 실행 130회를 집계한다. 로컬 추가 실험과 같은 기준군 재사용이 있어 이 문서에서는 전체 독립 반복 수로 합산하지 않는다.
- 공개 보고서·분석 CSV·protocol·로컬 결과를 교차 확인했다. 이 PC의 추가 assignment 30회는 원시 평가 CSV에서 AUC를 재계산했고, λ·A1 요약도 seed별 표와 대조했다. 모든 원격 원시 실행과 manifest를 이번에 다시 감사한 것은 아니다.
- 표는 읽기 쉽게 반올림했다. CI는 각 고정 분석의 결과를 사용했으며 새로운 유리한 통계 절차로 교체하지 않았다. 실행 시기·환경이 다른 집합의 합산은 그 차이를 제거하지 않는다.
- 기술 검증·smoke test·optimizer 복제 진단·checkpoint query를 독립적인 장기 학습 seed로 세지 않는다. 성능 실패인 유한값 0-success 실행도 결과에 포함한다.

### 원격 공유 자료

| 자료 | 링크 |
| --- | --- |
| 최신 공개 전체 상태 | [저장소 README][R1], [리뷰 이후 archive][R2] |
| 처음 배정 1–10 / PAL | [배정 비교][R3], [PAL][R4] |
| 조회·실패 / Random-512 최종 | [정적 진단][R5], [온라인·양쪽 조회][R6] |
| 원 논문 DDPG 8방법 확장 | [완료 artifact][R7] |
| SAC | [DrawerOpen][R8], [DrawerClose][R9] |
| 과거 제출용 시간 비용 | [FPP–TD3 100k profile][R10] |

### 이 PC 및 work 브랜치 자료

| 자료 | 링크 |
| --- | --- |
| 배정 11–20 및 합산 | [최종 보고서 — 로컬][L1] |
| Policy-query | [20k — 로컬][L2], [250k — 로컬][L3], [성숙 정책 — 로컬][L4] |
| Optimizer / Q 오차 | [20k — 로컬][L5], [250k — 로컬][L6], [2M Q — 로컬][L7] |
| 소스·실패 감사 | [소스 재감사][L8], [실패 심층 감사][L9] |
| A1 | [공개 artifact][L10], [원시 probe][L11], [입력 표현 기록][L12] |
| λ / 전이 계획 | [λ 결과 — 로컬][L13], [공통 context 설계][L14] |

“로컬” 링크는 이 PC의 절대 경로다. 다른 PC나 GitHub 웹에서는 직접 열리지 않을 수 있다. 핵심 결과는 이 문서에 포함했으며, 공동 재분석에 필요한 원자료 공개 범위는 미팅에서 정한다.

### 작성 방법

근거·해석·추측을 구분하고 표본 재사용과 적용 범위를 점검하는 데 `scientific-critical-thinking` 스킬을 사용했다. 도구 방법론 참고: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065). 이는 SABER 성능이나 메커니즘의 근거가 아니며, 문서 작성 지원 방법의 고지다.

[R1]: https://github.com/Hyunwoojooo/saber-rl/blob/332c2246d05ac05548335169b3f482175af8e4ce/README.md
[R2]: https://github.com/Hyunwoojooo/saber-rl/blob/332c2246d05ac05548335169b3f482175af8e4ce/ICML_27/README.md
[R3]: https://github.com/Hyunwoojooo/saber-rl/blob/332c2246d05ac05548335169b3f482175af8e4ce/ICML_27/experiments/draweropen_ddpg_saber_vs_random_vs_td_topk/README.md
[R4]: https://github.com/Hyunwoojooo/saber-rl/blob/332c2246d05ac05548335169b3f482175af8e4ce/ICML_27/experiments/draweropen_ddpg_saber_vs_pal/README.md
[R5]: https://github.com/Hyunwoojooo/saber-rl/blob/332c2246d05ac05548335169b3f482175af8e4ce/ICML_27/experiments/draweropen_ddpg_retrieval_and_failure_diagnostics/results/outputs/final/REPORT.md
[R6]: https://github.com/Hyunwoojooo/saber-rl/blob/332c2246d05ac05548335169b3f482175af8e4ce/ICML_27/experiments/draweropen_ddpg_saber_tail_vs_random512/README.md
[R7]: https://github.com/Hyunwoojooo/saber-rl/blob/332c2246d05ac05548335169b3f482175af8e4ce/metaworld_Drawer_Open_DDPG/paper_artifacts/drawer_open_ddpg_main_seed_extension_2m/README.md
[R8]: https://github.com/Hyunwoojooo/saber-rl/blob/332c2246d05ac05548335169b3f482175af8e4ce/metaworld_Drawer_Open_SAC/paper_artifacts/drawer_open_sac_her_saber_hgr_per_2m/README.md
[R9]: https://github.com/Hyunwoojooo/saber-rl/blob/332c2246d05ac05548335169b3f482175af8e4ce/metaworld_Drawer_Close_SAC/paper_artifacts/drawer_close_sac_her_saber_hgr_per_2m/README.md
[R10]: https://github.com/Hyunwoojooo/saber-rl/blob/332c2246d05ac05548335169b3f482175af8e4ce/experiment_FPP/paper_artifacts/fpp_td3_her_vs_saber_runtime_profile_100k/README.md
[R11]: https://github.com/Hyunwoojooo/saber-rl/blob/332c2246d05ac05548335169b3f482175af8e4ce/ICML_27/experiments/draweropen_ddpg_saber_tail_vs_random512/results/outputs/final_retrieval/report.json
[L1]: /home/mimiclab-gigabyte-gen5/ICML2027/draweropen_ddpg_assignment_extension_20261020/outputs/final/REPORT.md
[L2]: /home/mimiclab-gigabyte-gen5/ICML2027/draweropen_ddpg_her_source_pilot/outputs/her_s01_20k_pilot_01/diagnostic_analysis/REPORT.md
[L3]: /home/mimiclab-gigabyte-gen5/ICML2027/draweropen_ddpg_her_250k_pilot/outputs/her_s01_250k_pilot_01/diagnostic_analysis/REPORT.md
[L4]: /home/mimiclab-gigabyte-gen5/ICML2027/draweropen_ddpg_mature_policy_pilot_20261014/outputs/final_analysis/REPORT.md
[L5]: /home/mimiclab-gigabyte-gen5/ICML2027/draweropen_ddpg_optimizer_audit/outputs/audit_20k_01/REPORT.md
[L6]: /home/mimiclab-gigabyte-gen5/ICML2027/draweropen_ddpg_her_250k_pilot/optimizer_audit_recovery_20261014/outputs/audit_250k_recovery_01/REPORT.md
[L7]: /home/mimiclab-gigabyte-gen5/ICML2027/draweropen_ddpg_q_error_audit_20261015/outputs/attempt_01/REPORT.md
[L8]: ../diagnostics/assignment_source_reaudit_20261030/REPORT_ko.md
[L9]: ../diagnostics/assignment_extension_deep_audit_20261030/outputs/REPORT_ko.md
[L10]: https://github.com/Hyunwoojooo/saber-rl/blob/c1b94362dde8228cbf7ee70442ec93c109c95112/ICML2027/paper_artifacts/a1_actor_dynamics_20261031/README.md
[L11]: ../paper_artifacts/a1_actor_dynamics_20261031/records/saber/seed_102/diagnostics/probes.jsonl.gz
[L12]: ../paper_artifacts/a1_actor_dynamics_20261031/records/saber/seed_102/diagnostics/probe_bank.json
[L13]: /home/mimiclab-gigabyte-gen5/ICML2027/draweropen_ddpg_lambda_sensitivity_20261015/outputs/analysis/REPORT.md
[L14]: ../experment_design/03_common_context_transfer.md
[L15]: ../experment_design/2026-10-30_additional_experiments/README_ko.md

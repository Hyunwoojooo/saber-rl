# FetchPickAndPlace TD3 측정 protocol (2026-11-10 고정)

- 상태: **측정 전에 고정.** 검증(`outputs/validation_20261110_fpp`, 43개 통과)만 실행했다.
- 전체 계획에서의 위치: [논리 공백 로드맵](../../experment_design/2026-11-10_logic_gap_roadmap_ko.md) 2번
- DrawerOpen 쪽 설계와 결과: [초반부 설계](../../experment_design/04_first_order_usefulness/README_ko.md), [후반부 protocol](PROTOCOL_late_ko.md)

## 1. 질문

DrawerOpen–DDPG에서는 10k부터 2M까지 SABER 가중이 정책 조회 critic의 값 오차를 더 줄인다는 증거가 없었다. 제출 논문에서 효과가 가장 반복적으로 관측된 과제는 FetchPickAndPlace다. 같은 측정을 FPP–TD3에 적용한다.

> FPP–TD3의 checkpoint에서, SABER 가중으로 critic 두 개를 갱신하면 uniform(HER) 가중보다 actor가 읽는 critic1의 정책 조회 오차가 더 줄어드는가? 그 답이 SABER의 성능 이득이 있던 조건과 없던 조건에서 다른가?

## 2. 대상

| set | 내용 | HER AUC | SABER AUC | 차이 | 저장본 |
|---|---|---|---|---|---|
| `fpp_incl` (A) | inclusive-future HER 규칙, seed 6–10. 기존 실행 | 0.327 | 0.339 | +0.011 | 있음 |
| `fpp_primary` (C) | 주 비교 규칙(exclusive future), seed 1–5. 저장본을 남기려고 2026-11-10에 재학습 | 원 실행 0.372 | 원 실행 0.442 | 원 실행 +0.070 | 재학습이 끝나면 생김 |

- 두 set의 학습 코드는 HER relabel 조건 한 줄만 다르다(`> step` 대 `>= step`). inclusive 규칙에서는 relabel goal이 그 transition 자신의 다음 achieved goal일 수 있다.
- A의 AUC는 측정 전에 이미 알고 있던 값이다(제출 논문 supplement에 보고됨). A는 **성능 이득이 거의 없는 조건**이다.
- C는 원 실행과 seed가 같지만 다른 기계에서 다시 학습한 것이다. 원 실행과 수치가 같다고 가정하지 않는다. 재학습의 AUC를 따로 보고한다. 재학습 설정은 [재학습 protocol](../fpp_td3_primary_checkpoint_rerun_20261110/PROTOCOL_ko.md)에 있다.
- 시점: 실행마다 9개(50k, 100k, 150k, 200k, 300k, 400k, 500k, 750k, 1M). 저장본은 50k마다 20개가 있지만 초반을 촘촘히, 전 구간을 덮도록 9개를 미리 골랐다. set마다 10개 실행 × 9개 시점 = 90개.

## 3. 측정 설정

DrawerOpen과 같은 코드 경로를 쓴다(`fou/protocol.py`). TD3에 맞춰 달라지는 점만 적는다.

- critic 갱신: 학습 코드와 같다. critic 두 개의 가중 MSE 합, 공동 clip 10, 공용 Adam(저장된 상태에서 시작). 검증에서 `TD3Agent.update`와 비트 단위로 같음을 확인했다.
- 정책 조회 오차 R과 κ: actor가 읽는 critic1 기준.
- target smoothing 잡음: 학습 배치는 배치 seed로 고정한 한 번의 draw를 모든 arm이 공유한다. 정책 조회 target은 잡음 16회 평균이다.
- TD top-k의 순위: 학습 코드의 `td_error_abs`처럼 min(|td1|, |td2|).
- 배치 64개, K = 1, 10, 100. actor와 target은 고정한다. TD3의 지연된 actor 갱신은 하지 않는다.
- R1: 노이즈 없는 정책 80 episodes(4,000 queries), 원래 goal.
- R2: behavior 32 episodes에서 1,024 queries. relabel은 그 set의 학습 코드와 같은 규칙(A는 inclusive, C는 exclusive)을 쓴다.

## 4. 판정 규칙

초반부(A1)와 같은 형태다.

- 단위는 학습 seed. 학습 arm(SABER, HER)별로 따로 본다.
- seed마다 9개 시점 중 **7개 이상**이 같은 부호이고, 이것이 **5개 seed 모두**에서 성립하면 consistent(+ 또는 −). 아니면 mixed.
- tie(세지 않음): 정책 조회 RMS residual √R0 < 1e-3인 checkpoint, 모든 배치에서 p = 0 또는 1인 checkpoint.

| 구분 | 검사 | 대상 |
|---|---|---|
| **주 판정** | K = 100에서 SABER > uniform, R1과 R2 각각 | SABER로 학습한 checkpoint |
| 보조 | 같은 검사 | HER로 학습한 checkpoint(반사실 ψ) |
| 보조 | 1차 예측(G3), 실제 1-step에서 SABER > uniform | 두 arm |
| 보조 | K = 100에서 SABER > eligible Random | 두 arm |
| 보조 | H3: κ(M) > κ(E\M), 행동 거리 맞춤 | 두 arm |
| 보조 | 1차 예측 부호 일치율 ≥ 0.8 | 두 arm |
| 서술 | 역방향 arm, TD top-k, 시점별 추세, 평가 성공률과의 관계 | – |

## 5. 결과를 어떻게 읽을 것인가

| A (이득 없는 조건) | C (이득 있던 조건) | 의미 |
|---|---|---|
| mixed | consistent + | 값 정확도 경로가 성능 이득과 함께 나타난다. SABER의 메커니즘으로 주장할 근거. 두 조건의 차이(HER 규칙)가 "언제 유용한가"의 답이 됨 |
| mixed | mixed | 이득이 있던 조건에서도 값 정확도 경로의 증거가 없다. 이 경로는 DrawerOpen과 FPP 모두에서 기각. 남는 후보는 ∇_a Q 경로, actor 동역학, 또는 효과 자체의 불안정 |
| consistent + | consistent + | SABER 가중이 FPP에서는 정책 조회 critic을 돕지만 그것이 성능으로 이어지는 조건은 따로 있다 |
| consistent + | mixed | 값 경로와 성능 이득이 서로 반대로 나타난다. 메커니즘 주장과 맞지 않음 |
| 어느 쪽이든 consistent − | | SABER 가중이 정책 조회 오차를 악화시킨다 |

C의 재학습에서 SABER − HER AUC가 원 실행의 +0.070과 크게 다르면, C의 측정은 "이득이 있던 조건"이 아니라 "그 조건을 다시 돌린 결과"로 읽는다.

A만으로는 결론을 내리지 않는다. A가 mixed여도 +0.070의 원인에 대해서는 말해 주지 못한다.

## 6. 한계

- 고정 target 1-step 기준이고 Q^π 정확도가 아니다. ∇_a Q는 재지 않는다.
- actor를 고정한 critic 갱신이다.
- seed 5개. 판정 규칙은 서술용이고 확증 검정이 아니다.
- A와 C는 seed 집합이 다르다(6–10 대 1–5). 두 조건의 차이에는 seed 차이가 섞여 있다.

## 7. 실행

저장소 root에서 실행한다. checkpoint당 약 115초(4 threads)다.

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/run.py measure --set fpp_incl --out outputs/fpp_incl --arms saber
```

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/run.py measure --set fpp_incl --out outputs/fpp_incl --arms her
```

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/aggregate.py --out outputs/fpp_incl
```

C는 재학습이 끝난 뒤 `--set fpp_primary --out outputs/fpp_primary`로 같은 명령을 쓴다.

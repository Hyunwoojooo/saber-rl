# FPP TD3 20쌍(seed 11–30) 측정 protocol (2026-11-16 고정)

- 상태: **측정 전에 고정.** 이 문서를 쓴 시점에 `fpp_n20` 저장본의 측정값은 하나도 보지 않았다. 알고 있던 것은 각 실행의 평가 곡선과 AUC뿐이다.
- 전체 계획에서의 위치: [논리 공백 로드맵](../../experment_design/2026-11-10_logic_gap_roadmap_ko.md)의 다음 단계 1번
- 측정 코드와 설정: [PROTOCOL_fpp_ko.md](PROTOCOL_fpp_ko.md)와 같다. 바뀌는 것은 대상 set과 판정 규칙뿐이다.

## 1. 왜 하나

이전 FPP 측정 두 번(`fpp_incl` +0.011, `fpp_primary` 재학습 +0.004)은 모두 **성능 이득이 없던 cohort**였다. 그래서 "SABER 가중이 정책 조회 critic을 더 잘 맞추는가"에 대한 mixed 결과가 이득이 없어서인지, 이 경로가 원래 없어서인지 가를 수 없었다.

`fpp_n20`은 사전 등록한 20쌍 분석에서 **이득이 확인된** cohort이고(AUC +0.045 [+0.018, +0.073], 14/6), 50k마다 저장본이 있다. 이득은 후반에 몰려 있다(final-quarter AUC +0.137, 최종 성공률 1.0인 seed가 SABER 11/20, HER 2/20).

> 이득이 있는 FPP–TD3 cohort에서, SABER 가중은 uniform 가중보다 actor가 읽는 critic1의 정책 조회 오차를 더 줄이는가? 특히 이득이 몰린 후반부에서는 어떤가?

## 2. 대상

- set `fpp_n20`: HER·SABER × seed 11–30 × 9개 시점(50k, 100k, 150k, 200k, 300k, 400k, 500k, 750k, 1M) = 360개.
- 시점은 기존 FPP set과 같다(`FPP_STAGES`). 측정 설정(배치 64, K = 1·10·100, R1 80 episodes, R2 1,024 queries, 난수 seed)도 같다.
- 측정 순서: SABER로 학습한 checkpoint를 먼저(주 판정), 그 다음 HER로 학습한 checkpoint.

## 3. 판정 규칙

기존 FPP 규칙("seed마다 9개 중 7개 이상 같은 부호, 모든 seed에서")은 seed 5개용이다. seed 20개에서는 한 seed만 어긋나도 mixed가 되므로, 단위를 seed로 두고 seed 사이의 부호 검정을 쓴다.

- **seed 점수:** 그 seed의 센 시점들에서 `gain / R0`의 중앙값. gain = R(uniform 갱신 후) − R(arm 갱신 후), 양수면 SABER가 낫다.
- tie(세지 않는 시점): 기존과 같다. √R0 < 1e-3, 또는 모든 배치에서 p = 0이나 1인 checkpoint.
- **검정:** seed 점수의 부호에 대한 정확한 양측 부호 검정. 점수가 0인 seed(센 시점이 없는 경우 포함)는 제외한다.
- **판정:** p ≤ 0.05이고 양수가 많으면 consistent +, p ≤ 0.05이고 음수가 많으면 consistent −, 아니면 mixed.

| 구분 | 검사 | 대상 시점 | checkpoint |
|---|---|---|---|
| **주 판정** | K = 100에서 SABER > uniform, R1 | 9개 전체 | SABER로 학습 |
| 공동 보고(판정은 따로) | 같은 검사 | **후반 3개 (500k, 750k, 1M)** | SABER로 학습 |
| 보조 | 같은 검사, R2 | 전체, 후반 | SABER로 학습 |
| 보조 | 같은 검사 | 전체, 후반 | HER로 학습(반사실 ψ) |
| 보조 | 실제 1-step, 1차 예측(G3)에서 SABER > uniform | 전체 | 두 arm |
| 보조 | K = 100에서 SABER > eligible Random | 전체, 후반 | 두 arm |
| 보조 | κ(M) > κ(E\M), 행동 거리 맞춤 | 전체 | 두 arm |
| 서술 | 기존 규칙(9개 중 7개, 모든 seed)의 결과 | 전체 | 두 arm |

후반 3개 시점 판정을 주 판정과 나란히 두는 이유는, 측정 전에 이미 알고 있던 사실(이득이 후반에 몰림) 때문이다. 이 선택은 측정값을 보기 전에 했다. 두 판정이 다르면 둘 다 보고한다.

## 4. 성능과의 연결 (서술, 판정 없음)

- seed마다 짝 AUC 차이(SABER − HER), final-quarter AUC 차이와, 그 seed의 SABER 측정 점수(전체, 후반)의 Spearman 상관.
- 1M에서 평가 성공률이 0.9 이상인 SABER seed와 0.1 이하인 seed의 측정 점수 비교.
- 양성 비율 p의 시점별 추이. 정책이 무너진 실행에서 p → 1이 되어 SABER가 HER로 퇴화하는지(DrawerOpen에서 관찰).

## 5. ψ가 무엇을 대신 재는지 (로드맵 3번의 오프라인 사전 확인, 서술)

`per_sample.npz`의 샘플별 값(ψ, eligible 여부, ‖a^r − a^π‖, 나이)으로:

- eligible 안에서 ψ = 1과 ψ = 0 샘플의 행동 거리와 나이 비교.
- 행동 거리와 나이만으로 ψ를 얼마나 맞힐 수 있는지(ROC AUC). 1에 가까우면 ψ는 "최근·근접 데이터"의 대리 지표에 가깝다.

이 결과로 대조군 학습(최근 순, 거리 순 배정)이 필요한지를 정한다.

## 6. 결과를 어떻게 읽을 것인가

| 주 판정 또는 후반 판정 | 의미 |
|---|---|
| consistent + | 이득이 있는 조건에서는 값 정확도 경로가 나타난다. 이득 없는 cohort(fpp_incl, fpp_primary)의 mixed와 대비되어 메커니즘 주장의 근거가 된다 |
| mixed | 이득이 있는 조건에서도 값 정확도 경로의 증거가 없다. 이 경로는 DrawerOpen과 FPP 모두에서 기각. 남는 후보는 ∇_a Q 경로, actor 동역학(포화), 데이터 분포 효과 |
| consistent − | SABER 가중이 정책 조회 오차를 악화시키는데도 성능은 좋다. 이득은 값 정확도와 무관한 경로에서 온다 |

## 7. 한계

- 기존 FPP 측정과 같다. 고정 target 1-step 기준, actor 고정, ∇_a Q는 재지 않음.
- seed 20개의 부호 검정은 약 15/20 이상의 일관성만 잡는다.

## 8. 실행

저장소 root에서 실행한다.

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/run.py measure --set fpp_n20 --out outputs/fpp_n20 --arms saber --threads 4
```

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/run.py measure --set fpp_n20 --out outputs/fpp_n20 --arms her --threads 4
```

같은 `--out`에 여러 프로세스를 seed로 나눠 돌려도 된다. checkpoint 폴더는 측정이 끝난 뒤에만 만들어진다.

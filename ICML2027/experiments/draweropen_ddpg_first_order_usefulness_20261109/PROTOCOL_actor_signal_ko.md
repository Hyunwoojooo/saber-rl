# actor 신호 측정 protocol, FPP TD3 20쌍 (2026-11-17 고정)

- 상태: **측정 전에 고정.** 이 문서를 쓴 시점에 `fpp_n20` 저장본에서 이 측정의 값은 하나도 보지 않았다. 시간과 지표 크기는 측정 대상이 아닌 `fpp_primary`(seed 1, 500k) 저장본 하나로만 확인했다.
- 전체 계획에서의 위치: [논리 공백 로드맵](../../experment_design/2026-11-10_logic_gap_roadmap_ko.md)
- 코드: `fou/actor_signal.py`, `run_actor_signal.py`, `validate_actor_signal.py`(검사 39개, 2026-11-17 통과), `analyze_actor_signal.py`

## 1. 왜 하나

지금까지 확인된 것:
- `fpp_n20`(seed 11–30): SABER > HER, AUC +0.045 [+0.018, +0.073].
- 같은 seed의 count-matched 대조군: SABER > count-matched, +0.051 [+0.015, +0.086], count-matched ≈ HER. 따라서 이득은 **support가 고른 위치**에서 나온다.
- 그러나 같은 저장본에서 SABER 가중은 정책 조회 Q 오차(R1)를 uniform보다 **더 키웠다**(consistent −, 5/15). 따라서 그 위치가 "Q(s, a^π) 값을 더 정확히 맞추는 것"으로 작용하지는 않는다.

actor는 Q의 값이 아니라 **∇_a Q(s, a)|a=π(s)** 를 읽는다. 값 오차가 커지더라도 행동 방향 정보가 좋아질 수 있다.

> SABER 가중으로 갱신한 critic은 uniform으로 갱신한 critic보다 a^π 근처에서 "어느 행동이 실제로 더 나은지"를 더 잘 맞추는가? 그리고 그 기울기를 따라가면 실제 return이 더 오르는가?

## 2. 정답(ground truth)

FetchPickAndPlace(MuJoCo)와 저장본의 actor는 결정적이다. 그래서 저장한 시뮬레이터 상태 s에서 행동 a를 한 번 실행한 뒤 π를 따라가면 현재 정책의 행동가치 참값을 rollout 한 번으로 얻는다.

G(s, a) = Σ_{t<50} γ^t r_t, 원래 goal, 학습과 같은 sparse 보상(거리 ≤ 0.05면 0, 아니면 −1), γ = 0.98.

검사로 확인한 것: 같은 (s, a)에서 G가 매번 같다. rollout 보상이 reset부터 다시 돌린 원래 에피소드의 나머지 보상과 같다.

## 3. 대상과 설정

| 항목 | 값 |
|---|---|
| 저장본 | `fpp_n20`: SABER·HER × seed 11–30 × 9개 시점(50k–1M) = 360개. SABER로 학습한 것을 먼저 잰다 |
| query 상태 | 노이즈 없는 정책 에피소드 16개(reset seed 70,000 + e), 에피소드마다 시점 4개를 무작위로(난수 seed 970,000) → 64개 |
| 방향 | query마다 4차원 단위 방향 16개(난수 seed 971,000), ±σ 대칭 쌍. σ = 0.3(주), 0.1(보조). 행동은 [−1, 1]로 자르고, G와 Q에 같은 행동을 쓴다 |
| critic | K0(저장본 그대로)와 K = 100 step 갱신 후의 critic: uniform, saber, inverse, td_topk, random(4개 수열 평균). 갱신 loop는 기존 `run_branches`와 같다(검사로 정확히 일치 확인). actor·target 고정, head 0(actor가 읽는 critic1) |

## 4. 지표

query i, 방향 j에서 d*_ij = G(a⁺) − G(a⁻), d_ij = Q1(s, a⁺) − Q1(s, a⁻).

| 지표 | 정의 | 뜻 |
|---|---|---|
| **PA** (pair accuracy) | 정보 있는 쌍(\|d*\| > 1e-9)에서 sign(d) = sign(d*)인 비율 | 실제로 결과가 갈리는 두 행동 중 어느 쪽이 나은지 critic이 맞추는 비율 |
| GC | cos(∇_a Q1(s, a^π), ĝ*), ĝ* = Σ_j d*_ij z_j (σ = 0.3) | actor가 받는 기울기 방향과 실제 return의 평활 기울기 방향의 일치도 |
| AI | G(clip(a^π + 0.3 · ∇_aQ/‖∇_aQ‖)) − G(a^π), query 평균 | critic 기울기를 따라 한 걸음 갔을 때 실제 return 변화 |

checkpoint가 PA 검정에 들어가려면 K0 기준 정보 있는 쌍이 32개 이상, GC 검정에 들어가려면 ĝ* ≠ 0인 query가 8개 이상이어야 한다. 모자라면 그 시점은 세지 않는다(정책이 거의 항상 실패하는 초기 저장본).

## 5. 판정 규칙

기존 `fpp_n20` 측정과 같은 구조다.

- seed 점수: 센 시점들에서 (arm 값 − 비교 critic 값)의 중앙값.
- 검정: seed 점수 부호에 대한 정확한 양측 부호 검정. 점수가 없는 seed는 제외한다.
- 판정: p ≤ 0.05이고 양수가 많으면 consistent +, 음수가 많으면 consistent −, 아니면 mixed.

| 구분 | 검사 | 시점 | 저장본 |
|---|---|---|---|
| **주 판정** | PA, σ = 0.3: SABER > uniform (K = 100) | 9개 전체 | SABER로 학습 |
| 공동 보고 | 같은 검사 | 후반 3개(500k, 750k, 1M) | SABER로 학습 |
| 보조 | PA: SABER > eligible Random | 전체, 후반 | 두 arm |
| 보조 | PA σ = 0.1, inverse·TD top-k > uniform | 전체, 후반 | 두 arm |
| 보조 | GC, AI: SABER > uniform, AI: SABER > Random | 전체, 후반 | 두 arm |
| 보조 | 같은 검사들 | 전체, 후반 | HER로 학습(반사실 ψ) |
| **보조(장기 효과)** | 실행 간 비교: SABER 실행의 K0 critic − 같은 seed·시점 HER 실행의 K0 critic, PA·GC·AI | 전체, 후반 | 짝 |
| 서술 | 시점별 수준, 주 점수 및 실행 간 PA 점수와 AUC 이득의 Spearman | — | — |

실행 간 비교는 100 step 국소 효과가 아니라 **학습 전체에 걸친 누적 효과**를 본다. 다만 두 실행은 정책이 달라서 query 상태와 정답도 다르다. 그래서 인과 판정이 아니라 보조로만 둔다.

## 6. 해석 표

| 결과 | 해석 |
|---|---|
| 주 판정 consistent + | support 위치는 Q 값 정확도가 아니라 actor가 읽는 행동 순위·기울기를 개선한다. 메커니즘 서술을 이 경로로 바꾼다 |
| 주 판정 mixed, 실행 간 비교 + | 100 step 국소 효과는 없지만 학습 전체로 누적된 critic은 행동 순위를 더 잘 맞춘다. 장기 형성 경로 |
| 둘 다 mixed 또는 − | 행동 신호 경로도 지지되지 않는다. 남는 후보는 상태 분포(탐험) 변화 같은 간접 경로다 |
| SABER > uniform이지만 SABER ≈ Random | 재가중 일반의 국소 효과다. count-matched 성능 결과(SABER > count-matched)와 어긋나므로 그 차이를 따로 설명해야 한다 |

## 7. 실행

저장소 root에서 실행한다. 작업자 5개, 작업자당 1 thread.

```bash
for w in 0 1 2 3 4; do
  .venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/run_actor_signal.py measure \
      --set fpp_n20 --out outputs/actor_signal_n20 --worker $w --n-workers 5 &
done
.venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/analyze_actor_signal.py
```

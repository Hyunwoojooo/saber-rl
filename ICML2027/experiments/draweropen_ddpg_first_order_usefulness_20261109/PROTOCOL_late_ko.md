# 후반부 측정 protocol (2026-11-10 고정)

- 상태: **측정 전에 고정.** 아래 본문은 측정 전에 쓴 그대로다. 측정은 2026-11-10 09:05–09:27에 38개 모두 끝났고, 결과는 [late_ext2m](../../paper_artifacts/first_order_usefulness_20261109/late_ext2m/REPORT_ko.md)과 [late_mature](../../paper_artifacts/first_order_usefulness_20261109/late_mature/REPORT_ko.md)에 있다.
- 초반부 설계: [04_first_order_usefulness/README_ko.md](../../experment_design/04_first_order_usefulness/README_ko.md)
- 전체 계획에서의 위치: [논리 공백 로드맵](../../experment_design/2026-11-10_logic_gap_roadmap_ko.md) 1번

## 1. 질문

초반부(A1, 10k–500k)에서는 SABER 가중이 정책 조회 오차를 실제 갱신에서 더 줄인다는 증거가 없었다. 원 논문의 이득은 2M 학습 전체에서 나왔으므로 같은 측정을 후반부에 적용한다.

> 학습 후반부(500k–2M)의 checkpoint에서, SABER 가중으로 critic을 갱신하면 uniform(HER) 가중보다 정책 조회 오차가 더 줄어드는가?

새 학습은 하지 않는다. 이미 있는 2M 실행의 저장본만 읽는다.

## 2. 대상

| set | 출처 | 학습 arm | seed | 시점 | 개수 |
|---|---|---|---|---|---|
| `ext2m` | `~/ICML2027/draweropen_ddpg_assignment_extension_20261020/outputs/production` | SABER, Random, TD top-k | 11–20 | 2M 최종 `checkpoint.pt` | 30 |
| `mature` | `~/ICML2027/draweropen_ddpg_mature_policy_pilot_20261014/outputs/production` | SABER, HER | 1 | 500k, 1M, 1.5M, 2M | 8 |

- 모두 actor·critic·target·Adam·replay·index를 포함한다(2026-11-10 확인). `best_checkpoint.pt`는 쓰지 않는다. 로더가 거부한다.
- `ext2m`에는 HER로 학습한 arm이 없다. Random·TD로 학습한 checkpoint의 ψ는 "SABER였다면 무엇을 키웠을까"라는 반사실 값이다.
- 학습 코드 호환성: `task_drawer_open.py`, `replay.py`, `networks.py`, `numba_support_scan_drawer_open.py`는 A1·ext·mature에서 sha256이 같다. `weights.py`의 차이는 출력 필드 추가와 eligibility 계산 플래그뿐이고 support 판정은 같다. critic 갱신식(가중 MSE, clip 10, Adam)도 같다.

**측정 전에 이미 알고 있던 실행 결과** (`summary.json`의 `final_eval_success`):

| 학습 arm | 성공한 seed | 실패한 seed |
|---|---|---|
| SABER | 11, 12, 17, 19 | 13, 14, 15, 16, 18, 20 |
| Random | 11, 12, 13, 15, 16, 17, 18, 20 | 14, 19 |
| TD top-k | 11, 12, 14, 15, 17, 18, 19, 20 | 13, 16 |

`mature`는 SABER seed 1이 성공, HER seed 1이 실패다(2M 평가 성공률 0).

## 3. 측정 설정

초반부와 완전히 같다(`fou/protocol.py`). 비교 가능성을 위해 바꾸지 않는다.

- 배치 64개(seed `900_000 + b`), R1 20 episodes, R2 20 episodes에서 1,024 queries
- 비교 arm: uniform, SABER, 역방향(ψ=0을 키움), eligible Random, eligible TD top-k
- 실제 갱신: 제출 코드와 같은 critic step(가중 MSE, clip 10, 저장된 Adam 상태), K = 1, 10, 100. actor와 target은 고정
- gain = R(uniform 갱신 후) − R(arm 갱신 후). 양수면 그 arm이 더 낫다

## 4. 판정 규칙

**단위:** 실행 (arm, seed) 하나. 배치와 query는 실행 안의 반복이다.

**층:** checkpoint 자신의 마지막 평가 성공률로 나눈다. ≥ 0.9는 success, ≤ 0.1은 failed.

**tie(세지 않음):**
- 모든 배치에서 p = 0 또는 1이라 SABER 가중이 uniform과 같은 실행
- 정책 조회 RMS residual √R0이 1e-3 미만인 실행. float32가 Q ≈ −50을 약 4e-6까지만 구분하므로 그 아래의 차이는 반올림 잡음이다
- gain이 정확히 0인 실행

**판정:** 센 실행들에 정확 단측 부호 검정을 적용한다. p ≤ 0.05면 consistent(+ 또는 −), 아니면 mixed. 예: 20개 중 15개, 10개 중 9개.

| 구분 | 검사 | 모집단 |
|---|---|---|
| **주 판정** | K = 100에서 SABER > uniform, R1과 R2 각각 | success 실행 전체(세 학습 arm 합쳐 20개 예상) |
| 보조 | 같은 검사를 학습 arm별로 | SABER 학습 success는 4개뿐이라 서술만 |
| 보조 | K = 100에서 SABER > Random, SABER > TD top-k | success 실행 |
| 보조 | 역방향 > uniform | success 실행. SABER와 부호가 같으면 "ψ 방향"이 아니라 "재가중 자체"의 효과 |
| 보조 | H3: κ(M) > κ(E\M), 행동 거리 맞춤 | success 실행 |
| 보조 | 1차 예측(G3)의 부호 일치율 | 전체. 0.8 미만이면 1차 분석을 근거로 쓰지 않음 |
| 서술 | failed 층 전부, `mature` 전부 | 판정하지 않음 |

## 5. 검증에서 미리 확인된 구조 (결과가 아님)

검증은 fixture 5개에 아주 작은 설정(배치 2–8개, R1 2 episodes)으로 돌렸다.

- **실패한 SABER 실행(seed 13)은 8개 배치 모두 p = 1.000이다.** 모든 샘플이 support-positive라서 Ω가 전부 1이고, SABER 가중이 uniform과 같다. 즉 정책이 무너진 뒤에는 SABER가 HER로 퇴화한다. failed 층은 대부분 tie가 될 것으로 예상한다.
- 실패한 HER seed 1의 2M은 √R0 = 3.3e-4로 수치적 null이다.
- 성공한 실행의 R1 query는 네 phase에 걸쳐 있다(예: approach 35 / contact 19 / opening 16 / open 330). 초반부는 거의 approach뿐이었다.
- 후반부는 residual이 커서 SGD 유한차분의 곡률 항이 η = 1e-8에서도 9% 남는다. 검증은 Richardson 외삽과 float64 기준값으로 바꿨다.

## 6. 결과를 어떻게 읽을 것인가

| 주 판정 결과 | 의미 | 다음 |
|---|---|---|
| R1·R2 모두 consistent + | 정책이 과제를 푸는 구간에서는 SABER 가중이 정책 조회 critic을 돕는다 | 시점 추세를 보기 위한 A1-2M 학습을 검토. 논문은 "support가 유용한 조건"으로 |
| 한쪽만 + 또는 mixed | 후반부에도 값 정확도 경로의 증거가 없다 | DrawerOpen에서는 이 경로를 접고 FPP 측정(로드맵 2번)과 ψ 대리 신호 검사(3번)로 |
| consistent − | SABER 가중이 정책 조회 오차를 악화시킨다 | 메커니즘 주장 철회. 실패 조건 분석 |

역방향 arm이 SABER와 같은 부호로 consistent하면 어느 쪽이든 "ψ가 고른 위치"의 효과가 아니다.

## 7. 한계

- 2M 한 시점(ext2m)과 seed 1개의 네 시점(mature)이다. 시점 추세는 약하다.
- 같은 seed의 세 arm은 초기화와 첫 10k 무작위 행동을 공유한다. 완전히 독립은 아니다.
- SABER로 학습한 success 실행은 4개뿐이다.
- 고정 target 1-step 기준이고 Q^π 정확도가 아니다. ∇_a Q는 재지 않는다. actor를 고정한 critic 갱신이다.

## 8. 실행

저장소 root에서 실행한다. checkpoint당 약 100초, 4 threads 기준이다. 네 프로세스를 동시에 돌리면 20분 안팎이다.

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/run.py measure --set mature --out outputs/late_mature
```

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/run.py measure --set ext2m --out outputs/late_ext2m --arms saber
```

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/run.py measure --set ext2m --out outputs/late_ext2m --arms random
```

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/run.py measure --set ext2m --out outputs/late_ext2m --arms td_topk
```

집계:

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/aggregate.py --out outputs/late_ext2m
```

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/aggregate.py --out outputs/late_mature
```

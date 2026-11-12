# SABER 논리 공백 로드맵

- 작성일: 2026-11-10
- 목적: 논문 논리에서 비어 있는 고리, 지금까지 확인한 것, 남은 일을 한곳에 적어 둔다.
- 상태 표기: **완료 / 준비 완료 / 미착수 / 보류**. 진행하면서 이 문서의 상태 열을 고친다.
- 관련 문서: [리뷰 이후 첫 미팅 자료](../meetings/2026-11-05_post_review_first_meeting_ko.md), [실험 4 설계](04_first_order_usefulness/README_ko.md), [후반부 protocol](../experiments/draweropen_ddpg_first_order_usefulness_20261109/PROTOCOL_late_ko.md)

## 1. 비어 있는 고리

SABER의 주장을 다섯 고리로 나누면 다음과 같다.

| | 주장 | 상태 |
|---|---|---|
| ① | critic은 replay 행동 a^r에서 배우고, actor는 정책 행동 a^π에서 Q를 읽는다 | 사실 |
| ② | ψ = 1은 "같은 상황의 replay에 a^π의 bin이 있다"는 뜻이다 | 정의상 참. 무엇을 대신 재는 신호인지는 정해지지 않음 |
| ③ | ψ = 1 샘플에 학습을 더 몰아주는 것이 actor에게 이롭다 | 근거 없음 |
| ④ | 그 샘플의 a^r 오차를 더 맞추면 a^π에서의 Q(또는 ∇_a Q)가 좋아진다 | **근거 없음. 직접 측정도 지지하지 않음** |
| ⑤ | 그래서 성공률 AUC가 오른다 | 제출 당시 양성, 리뷰 이후 확장에서 흔들림 |

논문은 ①에서 ⑤로 바로 넘어간다. 리뷰어 wpgC의 "heuristic" 지적과 AI 리뷰의 "함수 근사 일반화에 의존한다"는 지적이 ③과 ④다. 핵심은 ④ 한 문장이다.

> support-positive 샘플의 replay-action residual을 더 맞추면, actor가 쓰는 a^π에서의 critic이 좋아진다.

판정 위치(a^π의 bin)와 가중 위치(샘플 자신의 a^r)가 다르다는 점이 어려움의 근원이다. ψ = 1 샘플 가운데 자기 행동이 a^π와 같은 bin인 비율은 SABER 학습 checkpoint에서 평균 18%였다.

## 2. 지금까지 확인한 것

### 2.1 초반부 직접 측정 (2026-11-09, 완료)

A1 checkpoint 48개(SABER·HER 학습, seed 101–103, 10k–500k)에서 critic만 복제해 가중치만 바꿔 갱신하고 정책 조회 오차 R의 변화를 쟀다. 보고서와 표는 [paper_artifacts/first_order_usefulness_20261109/early_phase_a1](../paper_artifacts/first_order_usefulness_20261109/early_phase_a1/REPORT_ko.md)에 있다.

| 검사 (SABER 학습 checkpoint, R1) | 결과 |
|---|---|
| 1차 예측(G3)에서 SABER > uniform | consistent + (seed별 7/1, 7/1, 8/0) |
| 1차 예측에서 SABER > eligible Random | consistent + |
| 실제 1-step에서 SABER > uniform | mixed (5/3, 5/3, 5/3). 중앙값 +0.26% |
| 실제 100-step에서 SABER > uniform | mixed, 음수 쪽 (4/4, 3/5, 3/5). 중앙값 −2.8% |
| 100-step에서 Random, 역방향 | uniform과 구분되지 않음 (+0.4%, −0.1%) |
| 1차 예측에서 SABER > TD top-k | mixed |
| κ(M) > κ(E\M) (전달 경로) | consistent +. 0.784 대 0.718, 행동 거리를 맞추면 +0.010 |
| 1차 예측 부호가 실제 1-step과 일치 | 0.70. 사전 기준 0.80 불통과 |

- HER 학습 checkpoint에서는 모든 항목이 mixed였다.
- 해석: ψ = 1 샘플의 gradient는 정책 조회 오차를 줄이는 쪽으로 정렬되어 있다. 그러나 1-step에서 SABER와 uniform의 차이는 두 arm에 공통인 Adam momentum 이동의 6–7%에 불과하고, 100 step 동안 누적되는 이점이 없다.
- **한계: 500k까지만 본다.** 대부분의 정책이 아직 학습되지 않은 구간이고, 정책 조회 query가 거의 approach phase에 몰려 있다.

### 2.1b 후반부 직접 측정 (2026-11-10, 완료)

기존 2M 실행의 저장본 38개에 같은 측정을 적용했다. 판정 규칙은 측정 전에 [PROTOCOL_late_ko.md](../experiments/draweropen_ddpg_first_order_usefulness_20261109/PROTOCOL_late_ko.md)에 고정했다. 보고서: [2M, seed 11–20](../paper_artifacts/first_order_usefulness_20261109/late_ext2m/REPORT_ko.md), [seed 1 시점별](../paper_artifacts/first_order_usefulness_20261109/late_mature/REPORT_ko.md).

**주 판정: mixed.** 성공한 실행 20개(SABER 학습 4, Random 학습 8, TD 학습 8)에서 100-step SABER gain의 부호는 다음과 같다.

| 정책 조회 오차 | SABER가 나음 / 못함 | 부호 검정 p | gain / R0 중앙값 (IQR) |
|---|---|---|---|
| R1 | 8 / 12 | 0.25 | −1.35% (−14.8% ~ +7.8%) |
| R2 | 10 / 10 | 0.59 | +0.82% (−7.8% ~ +6.9%) |

| 보조 검사 (성공 실행 20개) | R1 | R2 |
|---|---|---|
| 100-step에서 SABER > eligible Random | 11 / 9 | 11 / 9 |
| 100-step에서 SABER > TD top-k | 9 / 11 | 13 / 7 |
| 100-step에서 역방향 > uniform | 10 / 10 | 11 / 9 |
| 실제 1-step에서 SABER > uniform | 8 / 12 | 13 / 7 |
| 1차 예측(G3)에서 SABER > uniform | 9 / 11 | 13 / 7 |
| κ(M) > κ(E\M), 행동 거리 맞춤 | **20 / 0** | 같음 |
| 1차 예측 부호가 실제 1-step과 일치 | 0.88 | 0.89 |

- 어느 검사도 consistent 기준(20개 중 15개)에 못 미쳤다. κ만 예외다(0.797 대 0.692, 거리를 맞추면 +0.016).
- 후반부 성공 실행에서는 clipping이 항상 걸리고 1차 예측이 실제 부호를 88–89% 맞힌다. 그런데 그 1차 예측에서도 SABER의 일관된 우위가 없다. 초반부에 보였던 1차 수준의 우위는 2M에서 사라진다.
- SABER로 학습한 성공 실행 4개만 보면 100-step에서 R1 2 / 2, R2 3 / 1이다. 수가 적어 판정하지 않는다.
- **실패한 실행 10개 중 9개는 평균 양성 비율이 0.965 이상이고, 4개는 모든 배치에서 1.0이다.** 정책이 무너지면 거의 모든 샘플이 support-positive가 되어 SABER가 HER와 같아진다. 7개가 tie였다.
- seed 1의 시점별 기록(성공 구간 1M, 1.5M, 2M)에서 100-step SABER gain은 R1 −3.2%, −7.0%, +8.7%로 추세가 없다.
- 덧붙인 관찰(성공 실행 20개): ψ = 1 샘플은 ‖a^r − a^π‖가 작다(1.50 대 2.06, 20개 모두). 나이는 약간 어리다(양성 비율이 배치의 최신 1/4에서 0.578, 가장 오래된 1/4에서 0.502, 19개에서 같은 방향).

**결론:** DrawerOpen–DDPG에서는 10k부터 2M까지 어느 구간에서도 "SABER 가중이 정책 조회 critic의 값 오차를 더 줄인다"는 증거가 없다. a^r에서 a^π로 전달되는 경로(κ)는 일관되게 관측되지만, 그것이 더 나은 적합으로 이어지지 않는다.

**이 결과의 한계:** 20개 실행의 부호 검정은 강한 효과만 잡는다. 100-step gain의 실행 간 폭이 R0의 ±10% 정도라 작은 효과는 배제하지 못한다. 2M 한 시점이고, 고정 target 값 오차만 봤으며, actor를 고정한 critic 갱신이다.

### 2.2 코드를 읽고 돌리며 확인한 사실

- **Tail-512는 정확한 "최근 512개"가 아니다.** ring buffer가 slot을 같은 coarse 키로 덮어쓰면 `candidates()`의 중복 제거가 예전 위치를 남긴다. 논문의 "temporal recency로 해석하지 않는다"는 서술도, "최근"이라는 설명도 정확하지 않다.
- **로그의 `len_a_q_local`은 정의대로의 n_i가 아니다.** ψ = 1이 확정되면 스캔을 멈추고, coarse bucket에 a^π bin이 없으면 스캔을 건너뛴다. ψ 값은 정의와 같다.
- **gradient clipping이 초반에 거의 항상 걸린다.** 100k 이하에서 clip 비율 1.00, 200k 이후 0.33(SABER 학습 checkpoint). 걸리는 구간에서는 "평균 1 정규화"와 무관하게 SABER가 step 크기가 아니라 방향만 바꾼다.
- **정책이 무너진 실행에서는 SABER가 HER로 퇴화할 수 있다.** 실패한 SABER seed 13의 2M 저장본은 검사한 8개 배치 모두 p = 1.000이다. 모든 샘플이 support-positive라 Ω가 전부 1이다. 나머지 실패 실행도 같은지는 1번 측정에서 확인된다.
- 행동 bin 경계가 딱딱하다. float32에서 −0.6은 손 계산과 다른 bin에 떨어진다.

### 2.3 리뷰 이후 성능 결과 (미팅 자료 요약)

- 배정 대조군: seed 1–10에서 SABER 우위, 11–20에서 반전. 합산 SABER − TD = −0.091.
- DrawerOpen DDPG seed 6–10: SABER − HER = −0.045. 합산 +0.25, CI가 0 포함.
- Drawer SAC: 개선 없음. PER가 더 좋음.
- Random-512는 support를 덜 놓치지만 Tail-512보다 학습이 나쁨.
- 2M 진단: 배치 residual은 줄지만 policy-action Q MSE는 늘어남(183.26 → 186.83).

## 3. 할 일

| 순서 | 일 | 상태 | 비용 |
|---|---|---|---|
| 1 | DrawerOpen 후반부 측정 | **완료 (2026-11-10). 주 판정 mixed** | 2.1b절 |
| 0 | **SABER 이득의 재현성 확인** (2026-11-10 추가) | **진행 중. GPU 재실행을 2026-11-12 02:30에 시작** | 아래 0번 절 |
| 2 | FPP에서 같은 측정 | **완료 (2026-11-10). A·C 모두 mixed. C 재학습에서 원 실행의 이득이 재현되지 않음(+0.071 → +0.004)** | 아래 2번 절 |
| 3 | ψ가 무엇을 대신 재는지: 대조 arm 2개 | 미착수 | 2번 측정에 얹음 |
| 4 | 주장 확정과 원고 재작성 | 일부는 지금 가능 | – |
| 5 | 리뷰어 세부 요구 | 미착수 | 4번 뒤 |
| – | A1-2M 계측 학습 | 보류. 1번이 mixed라 지금 근거로는 불필요 | 학습 약 2일 |

### 0. SABER 이득의 재현성 확인 (2026-11-10 추가)

2번을 진행하다 드러난 문제다. 설명하려던 성능 이득이 이 PC의 실행에서는 관측되지 않는다.

- FPP–TD3 주 비교를 같은 코드·명령·seed로 다시 학습하니 SABER − HER가 +0.071에서 +0.004가 됐다(2번 절).
- 기계가 확인된 cohort만 보면 이득이 뚜렷한 것은 다른 기계(`/home/mimiclab/`)의 실행이고, 이 PC의 실행은 이득이 거의 없다. DrawerOpen의 seed 1–10 대 11–20 반전도 같은 모양이다.
- 우연인지, 기계·소프트웨어·device의 체계적 차이인지는 아직 가를 수 없다.

할 일:

1. 다른 기계의 6월 환경(패키지 버전, GPU, 드라이버)을 찾아 이 PC와 비교한다. 원 실행 폴더에는 환경 기록이 없다.
2. 이 PC에서 cuda로 일부 seed를 다시 돌려 device 효과를 분리한다. GPU는 다른 작업의 LLM 서버가 쓰고 있어 조율이 필요하다.
3. 가능하면 다른 기계에서 같은 코드를 다시 돌린다.
4. 그 결과에 따라 논문의 성능 주장 범위를 다시 정한다. 메커니즘 작업(1–3번)은 그 뒤다.

진행 기록:

- 2026-11-12: "다른 기계"가 무엇인지 확인했다. 저장소의 실행 설정에 기록된 출력 경로의 home 폴더가 네 가지다. `/home/mimiclab`(FPP, 5–7월. 주 비교가 여기서 돌았다), `/home/mimiclab-gigabyte-gen4`(DrawerOpen·DrawerClose DDPG 일부, 7월), `/home/mimiclab_asus`(FetchPush, DrawerOpen REDQ), `/home/mimiclab-gigabyte-gen5`(이 PC). 이 PC에는 `/home/mimiclab` 계정이 없다.
- cpu 재학습의 평가 곡선은 30k–180k까지 원 실행과 값이 같고 그 뒤 갈라진다. 환경과 seed는 재현되고, 궤적은 부동소수 차이로 갈라진다.
- 2026-11-12 02:30: 할 일 2번을 시작했다. 같은 10개 실행을 이 PC에서 cuda로 다시 돌린다. 설정은 원 실행과 출력 경로·thread 키만 다르다. 읽는 규칙은 [재학습 protocol](../experiments/fpp_td3_primary_checkpoint_rerun_20261110/PROTOCOL_ko.md)의 "추가: GPU 재실행"에 미리 적었다(평균 ≥ +0.035이고 4개 이상 양수면 재현, ≤ +0.020이면 재현 안 됨).

### 1. DrawerOpen 후반부 측정

- 대상: assignment extension의 2M 최종 저장본 30개(SABER·Random·TD × seed 11–20), 성숙 정책 pilot 8개(SABER·HER seed 1 × 500k/1M/1.5M/2M).
- 준비한 것: 로더 확장, 검증 58개 통과, 판정 규칙 고정([PROTOCOL_late_ko.md](../experiments/draweropen_ddpg_first_order_usefulness_20261109/PROTOCOL_late_ko.md)). 초반부 checkpoint 재측정 결과가 바이트 단위로 같아 기존 결과에 영향이 없다.
- 주 판정: 성공한 실행 전체(20개 예상)에서 K = 100 SABER gain의 부호, 정확 부호 검정 p ≤ 0.05.
- 답하는 질문: 값 정확도 경로가 정책이 과제를 푸는 구간에서는 살아 있는가.
- **결과(2.1b절): mixed.** protocol 6절의 규칙대로, DrawerOpen에서는 값 정확도 경로를 접고 2번과 3번으로 간다. DrawerOpen에서 SABER 자체의 효과가 보이지 않으므로, 3번의 대조 arm은 여기서 돌려도 설명할 효과가 없다. 2번에 얹는다.

### 2. FPP에서 같은 측정

- 이유: 리뷰 이후 진단이 전부 DrawerOpen–DDPG에 몰려 있다. 이 과제는 실행이 성공과 실패로 갈려 변동이 가장 크다. 제출 논문이 "DDPG·TD3·SAC에 걸쳐 반복된다"고 쓴 곳은 FPP다.
- 쓸 수 있는 저장본(`experiment_FPP/runs/`, 50k마다 1M까지, replay 포함 확인):
  - `td3_inclusive_future_sensitivity_ls512_seeds6_10`: TD3 HER·SABER seed 6–10. HER 규칙을 바꾼 민감도 변형이다.
  - `td3_saber_context_eligible_count_matched_ls512`, `td3_saber_query_replay_ls512`, `td3_saber_query_shuffled_ls512`, `td3_saber_context_count_only_ls512`: 대조군 seed 1–5.
  - 주 비교 cohort(`td3_her_baseline_ls512`, `td3_saber_critic_only_v4_ls512`, DDPG `B0` 등)는 저장본을 남기지 않았다.
- 필요한 작업: critic 2개와 target smoothing을 다루는 TD3 갱신, FPP task spec, Fetch 환경의 R1 query bank. R2(snapshot 복원)는 나중에.

**포팅 상태 (2026-11-10 완료).** 측정 코드를 critic 여러 개와 환경별 어댑터 구조로 일반화했다.

- TD3: critic 2개의 loss 합, 공동 clip, 공용 Adam을 그대로 따른다. 정책 조회 오차 R과 κ는 actor가 읽는 critic1 기준이다. target smoothing 잡음은 배치마다 seed로 고정해 모든 arm이 공유하고, 정책 조회 target은 잡음 16회 평균을 쓴다.
- 검증 43개 통과(`validation_20261110_fpp`). **우리의 critic 갱신이 학습 코드의 `TD3Agent.update`와 비트 단위로 같다.** 같은 검사를 DrawerOpen에도 추가했고 거기서도 비트 단위로 같다.
- Fetch의 시뮬레이터 상태 복원은 reset부터 다시 재생한 결과와 비트 단위로 같다. R2도 구현했다.
- DrawerOpen 결과는 일반화 뒤에도 바이트 단위로 같다(초반부·후반부 checkpoint 각 1개 재측정).
- checkpoint당 약 115초(4 threads).

**측정 대상에 대해 새로 확인한 사실.** "FPP는 효과가 안정적인 곳"이라는 전제가 저장본이 남은 cohort에는 그대로 맞지 않는다.

| FPP–TD3 cohort | HER AUC | SABER AUC | 차이 | 저장본 |
|---|---|---|---|---|
| 주 비교, seed 1–5 (exclusive future) | 0.372 | 0.442 | +0.070 | **없음** |
| inclusive-future, seed 6–10 | 0.327 | 0.339 | +0.011 | 있음 (50k마다 20개) |

- 저장본이 있는 HER·SABER 쌍은 효과가 거의 없는 inclusive-future seed 6–10뿐이다. 이 차이(+0.0114)는 제출 논문 supplement에도 보고되어 있다.
- 주 비교 규칙으로 학습한 저장본은 대조군(count-matched 0.391, query-replay 0.368, query-shuffled 0.347, context-count-only 0.396)만 남아 있다.

**결정 (2026-11-10): A와 C를 모두 진행한다.** 판정 규칙은 측정 전에 [PROTOCOL_fpp_ko.md](../experiments/draweropen_ddpg_first_order_usefulness_20261109/PROTOCOL_fpp_ko.md)에 고정했다. seed마다 9개 시점 중 7개 이상이 같은 부호이고 5개 seed 모두에서 성립하면 consistent다.

- A: `fpp_incl` 90개 checkpoint 측정을 10:08에 시작해 12:24에 끝냈다. CPU 과부하 때문에 중간에 thread 수를 줄여 다시 시작했다(끝난 40개는 유지).

**A의 결과: 주 판정 mixed.** [보고서](../paper_artifacts/first_order_usefulness_20261109/fpp_incl/REPORT_ko.md). SABER로 학습한 checkpoint 45개(seed 5개 × 시점 9개)에서:

| 검사 | R1 (+ / −, 중앙값) | R2 (+ / −, 중앙값) |
|---|---|---|
| 100-step에서 SABER > uniform (주 판정) | 19 / 26, −1.71% | 16 / 29, −0.58% |
| 실제 1-step에서 SABER > uniform | 15 / 30, −0.06% | 13 / 32, −0.04% |
| 1차 예측(G3)에서 SABER > uniform | 15 / 30 | 13 / 32 |
| 100-step에서 Random > uniform | 15 / 30, −2.11% | 14 / 31, −0.80% |
| 100-step에서 역방향 > uniform | 22 / 23, −0.15% | 23 / 22, +0.03% |
| 100-step에서 SABER > Random | 22 / 23 | 22 / 23 |
| κ(M) > κ(E\M), 행동 거리 맞춤 | consistent + (0.786 대 0.693, +0.048) | 같음 |
| 1차 예측 부호 일치율 | 0.97 | 0.91 |

- 어느 seed 규칙으로도 consistent가 아니다. 방향은 약하게 음수 쪽이다. SABER로 학습한 checkpoint에서는 SABER 가중이 uniform보다 정책 조회 오차를 조금 덜 줄인다. 1차 예측이 실제 부호를 91–97% 맞히는 조건인데, 그 예측부터 음수 쪽이다.
- eligible Random도 uniform보다 같은 만큼 나쁘고 SABER와 구분되지 않는다. 역방향은 uniform과 같다. 즉 "ψ가 고른 위치"의 효과는 보이지 않는다.
- HER로 학습한 checkpoint 45개에서도 100-step은 16 / 29(R1), 18 / 27(R2)이다.
- κ는 여기서도 일관되게 양수이고 DrawerOpen(+0.010 ~ +0.016)보다 크다.
- 양성 비율 p는 평균 0.20으로 DrawerOpen(0.4 ~ 0.6)보다 낮다. R1 query의 85%가 approach phase다.
- protocol에 적은 대로 A만으로는 결론을 내리지 않는다. 이 cohort는 성능 이득이 없던 조건이다. C와의 대조가 필요하다.
- C: [재학습 protocol](../experiments/fpp_td3_primary_checkpoint_rerun_20261110/PROTOCOL_ko.md). HER·SABER × seed 1–5를 10:10에 시작했다. GPU는 다른 작업의 LLM 서버가 쓰고 있어 cpu로 돌린다(속도 차이는 벤치마크에서 60 대 67 env step/s). 새 실행의 설정은 원 실행과 device·출력 경로·thread 키만 다르다. 끝나면 `fpp_primary`로 같은 측정을 한다.

**C 재학습의 결과 (2026-11-10 17:15 완료): 원 실행의 이득이 재현되지 않았다.** [비교 자료](../paper_artifacts/fpp_td3_primary_rerun_20261110/README.md).

| | HER AUC | SABER AUC | 차이 | SABER가 높은 seed |
|---|---|---|---|---|
| 원 실행 (다른 기계, cuda, 6월) | 0.3716 | 0.4424 | +0.0708 | 4 / 5 |
| 재학습 (이 PC, cpu) | 0.3363 | 0.3406 | +0.0043, 95% 구간 [−0.015, +0.023] | 3 / 5 |

- 같은 코드, 같은 명령, 같은 seed다. 설정은 device·출력 경로·thread 키만 다르다. 10개 실행 모두 끝났고 실패는 없다.
- 두 arm 모두 원 실행보다 낮다. HER는 평균 −0.035(5개 seed 모두 낮음), SABER는 평균 −0.102(5개 seed 모두 낮음).
- 저장소에 남은 FPP–TD3 기록을 기계별로 보면, 이득이 뚜렷한 cohort(+0.071, +0.023)는 다른 기계의 실행이고, 이 PC의 두 cohort(+0.011, +0.004)는 이득이 거의 없다. cohort마다 seed, HER 규칙, device가 함께 달라서 기계 효과라고 단정할 수는 없다.
- 논문에서 SABER(0.442, 다른 기계)와 비교한 FPP–TD3 대조군들은 모두 이 PC에서 돌았다. 그 비교에는 기계 차이가 섞여 있다.
- DrawerOpen에서도 같은 모양이 있었다. 원래 cohort(seed 1–10)는 SABER가 높았고, 이 PC에서 돌린 seed 11–20은 반전됐으며, 원래 cohort의 실행 환경은 확인하지 못했다.
- **이것이 지금 가장 먼저 풀어야 할 문제가 됐다.** 메커니즘을 묻기 전에, 설명할 효과가 이 PC에서는 관측되지 않는다. 다른 기계의 6월 환경을 찾아 비교하고, device 효과를 분리하고, 가능하면 그 기계에서 다시 돌려 본다.

**C의 측정 결과 (2026-11-10 18:35 완료): 주 판정 mixed.** [보고서](../paper_artifacts/first_order_usefulness_20261109/fpp_primary/REPORT_ko.md). 재학습한 checkpoint 90개다. 재학습에서 성능 이득이 나오지 않았으므로, protocol에 적은 대로 "이득이 있던 조건"이 아니라 "그 조건을 다시 돌린 결과"로 읽는다.

| SABER로 학습한 checkpoint 45개 | A: inclusive-future | C: 주 비교 규칙 재학습 |
|---|---|---|
| 그 cohort의 SABER − HER AUC | +0.011 | +0.004 |
| 100-step에서 SABER > uniform, R1 (주 판정) | 19 / 26, −1.71% | 16 / 29, −1.98% |
| 같은 검사, R2 | 16 / 29, −0.58% | 16 / 29, −1.21% |
| 실제 1-step에서 SABER > uniform, R1 | 15 / 30 | 23 / 22 |
| 100-step에서 Random > uniform, R1 | 15 / 30, −2.11% | 15 / 30, −1.49% |
| 100-step에서 역방향 > uniform, R1 | 22 / 23 | 23 / 22 |
| 100-step에서 SABER > Random, R1 | 22 / 23 | 21 / 24 |
| κ(M) 대 κ(E\M) | 0.786 대 0.693, consistent + | 0.809 대 0.718, consistent + |
| 1차 예측 부호 일치율, R1 | 0.97 | 0.95 |

- 두 cohort의 모양이 같다. SABER 가중은 100 step 뒤에 uniform보다 약간 나쁜 쪽이고, eligible Random과 구분되지 않으며, 역방향은 uniform과 같다. κ만 일관되게 양수다.
- HER로 학습한 checkpoint에서도 같다(C의 R1 100-step 20 / 25).
- **이 측정으로 원 실행의 +0.071이 어디서 왔는지는 알 수 없다.** 성능 이득이 있는 cohort의 저장본을 끝내 얻지 못했기 때문이다. 말할 수 있는 것은 "이 PC의 FPP–TD3 실행에서는 SABER가 성능도, 정책 조회 critic의 적합도 개선하지 않았고 두 사실이 서로 맞는다"까지다.

선택지(결정 전의 기록):

| | 대상 | 비용 | 답할 수 있는 것 |
|---|---|---|---|
| A | inclusive-future HER·SABER 10개 실행 × 9개 시점 | 측정 약 45분, 학습 없음 | 효과가 없는 cohort에서 값 경로가 어떤지. null이면 +0.070의 원인에 대해서는 말해 주지 못함 |
| B | 주 비교 규칙의 대조군 저장본에 반사실 SABER 가중 | 그 code 폴더용 포팅 조금 + 측정 | 효과가 있던 규칙에서의 값 경로. 단 학습 규칙이 SABER가 아님 |
| C | 주 비교 규칙으로 HER·SABER seed 1–5를 저장본을 남기며 재학습 | 실행당 약 8.4시간(기존 실행 로그 기준, GPU 1개에서 병렬 실행), 10개 실행 | 효과가 있던 바로 그 조건에서의 값 경로. +0.070이 재현되는지도 함께 확인 |

### 3. ψ가 무엇을 대신 재는지

- 가설: ψ는 "support"가 아니라 "최근 정책이 만든 데이터"를 고르고 있을 수 있다. Tail-512가 support를 더 놓치는데도 Random-512보다 학습이 낫고, ψ = 1 샘플은 ‖a^r − a^π‖가 일관되게 작다.
- 추가할 arm: 같은 개수를 (a) ‖a^r − a^π‖가 작은 순으로, (b) 최신 순으로 고르는 배정.
- 이 arm들이 SABER와 같거나 낫다면 SABER는 on-policy 정도 가중(LFIW, ReF-ER 계열)의 근사다. 신규성 서술을 그에 맞춰야 한다.
- 준비: 측정 코드가 이제 샘플별 나이(`age`)와 행동 거리(`dist`)를 저장한다. 1번 결과에서 1차 수준의 사전 확인은 재실행 없이 할 수 있다.

### 4. 주장 확정과 원고 재작성

| 1–3의 결과 | 논문 방향 |
|---|---|
| 어느 구간이나 과제에서 일관된 양성 | "support가 유용한 조건"을 명시하는 논문. 조건(시기, phase, 과제)이 핵심 기여 |
| 어디서도 양성 없음 | 메커니즘 주장을 내린다. 리뷰 이후 성능 결과까지 합치면 지금 형태로는 방어가 어렵다. 판정 위치와 가중 위치를 일치시키는 식으로 방법을 고치는 쪽을 검토 |

결과와 무관하게 지금 할 수 있는 원고 작업:

- 본문 `.tex`의 낡은 수치 5개를 제출 PDF에 맞추기(0.417 / 1.16x / 1.79x / 73.0% / 0.306)
- Method를 알고리즘 박스와 task별 context 표로 줄이기
- 4.3절 진단을 표로 옮기기
- Tail-512 서술 고치기(2.2절)
- "평균 1 정규화" 설명에 clipping 사실 추가
- 소소한 수정: 에피소드 끝의 실효 relabel 확률, SAC의 "current-policy action" 표현, d·ε_g·h 정의

### 5. 리뷰어 세부 요구

APER 비교, 공통 context 전이([설계](03_common_context_transfer.md)), 같은 조건의 wall-clock 비교, N_local·N_scan 민감도([설계](02_support_hyperparameter_sensitivity.md)).

### 보류: A1-2M 계측 학습

1번 결과가 양성이면 시점별 추세를 보려고 실행할 계획이었다. 1번이 mixed라 지금은 실행할 이유가 없다. 2번에서 FPP가 양성으로 나오고 DrawerOpen과의 차이를 설명해야 할 때 다시 검토한다.

- HER·SABER·eligible Random × 새 seed 201–205 × 2M. TD top-k는 측정 단계에서 오프라인으로 비교하므로 학습 arm에서 뺀다.
- 저장 시점 16개: 0, 4.8k, 5k, 10k, 20k, 50k, 100k, 200k, 300k, 500k, 750k, 1M, 1.25M, 1.5M, 1.75M, 2M.
- 목적은 AUC 비교가 아니라 실험 4를 학습 전 구간에 적용하는 것이다.
- 비용: 실행당 약 16시간, 15개를 5개씩 병렬로 약 2일. 저장 공간 약 36GB.

## 4. 지금 하지 않는 것

- DrawerOpen–DDPG의 AUC seed 추가. 이미 20개가 있고 공백은 메커니즘이다.
- 측정 전에 이론을 더 쌓는 것. 조건이 성립하는지부터 본다.
- 유리한 시점이나 seed만 골라 보고하는 것.

## 5. 정리가 필요한 것

- 측정 코드, 설계 문서, 이 로드맵은 2026-11-10 현재 커밋하지 않았다.
- 측정 원자료(`experiments/draweropen_ddpg_first_order_usefulness_20261109/outputs/`)는 git에서 제외된다. 요약 표와 보고서만 `paper_artifacts/first_order_usefulness_20261109/`에 복사했다.
- 실험 4 설계가 기대는 「SABER support 정당화 분석 및 검증 설계」(2026-11-05)는 저장소에 없다. 원문을 `experment_design/04_first_order_usefulness/`에 넣어 두어야 한다.

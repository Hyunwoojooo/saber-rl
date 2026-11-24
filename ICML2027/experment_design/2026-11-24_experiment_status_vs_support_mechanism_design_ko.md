# 실험 현황: support 메커니즘 검증 설계와의 대응

- 작성일: 2026-11-24
- 기준 문서: `origin/main:ICML_27/SABER_support_mechanism_study_design_ko.md` (commit ddb188d, 2026-11-24). 아래에서는 "설계서"라고 부른다.
- 범위: 이 PC(`work/icml2027` 브랜치, `ICML2027/`)에서 리뷰 이후 수행한 실험과 지금 돌고 있는 실험. 각 실험이 설계서의 네 검증(이론 1, 실험 1, 이론 2, 실험 2) 중 어디에 해당하는지, 무엇이 이미 답해졌고 무엇이 남았는지를 적는다.
- 세부 기록: [논리 공백 로드맵](2026-11-10_logic_gap_roadmap_ko.md)(단일 진입점), 각 실험 폴더의 `PROTOCOL_ko.md`. 이 브랜치의 f228a57 이후 작업은 아직 커밋하지 않았다.

## 0. 요약

| 설계서 항목 | 이 PC에서 한 것 | 상태 | 한 줄 결론 |
|---|---|---|---|
| 실험 2 (같은 체크포인트에서 배정만 바꾸기) | 실험 4 "1차 유용성 측정" 5개 cohort + actor 신호 측정 | **대부분 완료** | 배정만 바꾼 native Adam 100회 업데이트에서 SABER가 Uniform·eligible Random보다 정책 행동 Q 오차를 줄이지 않음. FPP TD3 이득 cohort에서는 오히려 일관된 음의 방향 |
| 실험 1 (주변 데이터 제거·복구) | 2026-11-10 CPU pilot만 | **미통과 (fitting gate)** | 설계서 4.3절에 적힌 대로 training RMSE 1.79 > 0.05. 본 측정 안 함 |
| 실험 1 보조 (4.4절, 판정과 경험의 대응) | 실험 4에서 부수적으로 일부 측정 | 일부 | ψ=1 샘플 중 자기 행동이 정책 bin에 속하는 비율 평균 18%, ψ=1은 정책 행동에 더 가까움, 거리+나이로 ψ 예측 ROC AUC 0.65 |
| 이론 1, 이론 2 | 이 PC에서는 하지 않음 | – | 실험 4 결과가 이론 2의 "부호 미정" 경우에 해당한다는 근거로 쓸 수 있음 |
| (설계서 범위 밖) 성능 재현성 | FPP TD3 n20 + 대조군 2개, stage 0 inventory, stage 1 n20 확장 | FPP 완료, **Drawer 진행 중** | FPP TD3만 SABER > HER 확정(+0.045). 이 이득은 a^π-bin 조회가 있어야 나타남. 다른 cell은 n20에서 모두 판정 불가 또는 진행 중 |

가장 중요한 사실 두 가지:

1. **설계서 실험 2가 묻는 질문은 이미 상당 부분 측정됐고 답은 음성이다.** 설계서 6.1절은 A1 초반부(DrawerOpen DDPG)와 2M 후반부 결과만 인용한다. 그 이후 FPP TD3에서 세 cohort를 더 측정했고, 이득이 확인된 n20 cohort에서도 SABER 배정은 정책 행동 Q 오차를 줄이지 않았다(2절).
2. **설계서 1절의 "count-matched 대조는 세 과제 모두 CI가 0 포함"은 기존 소수 seed cohort의 결과다. 어느 seed인지는 postAAAI_Review.md 6.1절에서 확인해야 한다.** 새 seed 11–30을 쓴 FPP TD3 20쌍에서는 count-matched 대조와 context-count-only 대조 모두 SABER가 우위다. CI가 0을 포함하지 않는다(3절). 이득이 "같은 개수를 무작위로 올려 주는 것"이나 "context 조건"만으로는 생기지 않는다는 뜻이다.

## 1. 설계서의 핵심 질문과 지금까지의 답

설계서는 SABER의 논리를 세 질문으로 나눈다.

| 질문 | 설계서 검증 | 현재 증거 |
|---|---|---|
| 정책 행동 주변의 데이터가 Q 예측에 중요한가? (coverage의 중요성) | 이론 1, 실험 1 | 직접 증거 없음. 실험 1 pilot이 fitting gate를 통과하지 못함 |
| SABER의 이산 support 판정이 그 데이터를 잘 포착하는가? (지표의 적합성) | 실험 1 보조(4.4절) | 일부. ψ=1은 정책 행동에 더 가깝지만 자기 행동이 정책 bin에 드는 비율은 18%뿐 |
| 같은 예산에서 support에 따라 배정하면 정책 행동 Q가 좋아지는가? (배정의 효과) | 이론 2, 실험 2 | **음성.** 5개 cohort에서 SABER 배정 > Uniform이 일관되게 나온 적 없음. FPP TD3 n20에서는 SABER 배정 < Uniform이 일관됨 |

그런데 성능 쪽에서는 FPP TD3에서 SABER > HER, SABER > count-matched, SABER > context-count-only가 모두 확정됐다. 즉 **"a^π-bin 조회가 있어야 이득이 생긴다"는 것은 성능으로 확인됐지만, 그 경로가 "정책 행동 Q 정확도"는 아니다.** 설계서 7절 해석표 다섯째 줄("전이 경로는 관찰되나 … 순효과가 없음")과 여섯째 줄("일관된 이득 없음")에 해당한다.

## 2. 설계서 실험 2에 해당하는 측정 (완료)

### 2.1 무엇을 했나

- 설계: `ICML2027/experment_design/04_first_order_usefulness/README_ko.md`
- 코드: `ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/` (`fou/core.py`, `run.py`, `validate.py`, `aggregate.py`, `analyze_fpp_n20.py`)
- 요약 결과: `ICML2027/paper_artifacts/first_order_usefulness_20261109/`
- 방식: 저장된 checkpoint(critic, Adam moment, actor, target, replay)를 불러와 **같은 batch 계획**으로 critic을 분기한다. 분기는 Uniform, SABER(실제 scanner의 M), eligible Random(E에서 같은 k개), TD-error Top-k, inverse(ψ=0에 2배)다. 각각 1회와 100회 native Adam + clip 업데이트 뒤 정책 행동 query 오차를 비교한다. 1차 예측(G3 = ⟨∇R_π, g_M − g_C⟩)과 실제 부호의 일치도 잰다.
- 설계서 6.2절의 분기 표(Uniform / SABER / Eligible Random / Eligible TD Top-k)와 같은 구성이고, inverse arm이 하나 더 있다.

### 2.2 설계서와 다른 점 (중요)

| 항목 | 설계서 실험 2 | 실험 4 (수행) |
|---|---|---|
| 주 평가 기준 | **고정 정책 rollout return** 대비 MSE (HER-goal query) | **고정 Bellman target** 대비 오차 R1(원 목표 query), R2(HER 목표 query) |
| 저장본 | A1 4개 출처 × seed 101–103 × 4,800–500k 고정 subset | A1 10k–500k + 2M 저장본 + FPP TD3 세 cohort |
| Actor 연결 | 보조 endpoint로 actor 1회 갱신 후 실제 return | 별도 측정(2.4절)으로 실제 return 기반 PA·GC·AI 지표 |

설계서 6.1절의 지적대로 R1/R2는 "frozen target 오차"이지 실제 Q 정확도가 아니다. 따라서 **설계서 실험 2의 주 endpoint(rollout 기준 MSE)는 아직 측정되지 않았다.** 다만 2.4절의 actor 신호 측정은 실제 rollout(H=50)을 기준으로 썼고, 결과는 역시 mixed였다.

### 2.3 결과

| cohort | 과제·backbone | 저장본 | 주 결과 (SABER 학습 checkpoint, K=100, SABER vs Uniform) | 그 밖 |
|---|---|---|---|---|
| A1 초반부 | DrawerOpen DDPG, seed 101–103 | 10k–500k | 1-step 5/3, 100-step 중앙값 −2.8%. Random·inverse와 구별 안 됨 | G3–실제 1-step 부호 일치 70% (사전 기준 80% 미달). 초반에는 clip이 거의 항상 걸림 |
| 2M 후반부 | DrawerOpen DDPG | 38개 2M 저장본 | 성공 실행 20개 중 SABER 우위 R1 8/20, R2 10/20. **판정 mixed** | 실패 실행 10개 중 9개는 support-positive 비율 ≥ 0.965 → SABER가 HER로 퇴화 |
| fpp_incl | FPP TD3 inclusive-future, seed 6–10 | 45개 | R1 +19/−26 (중앙값 −1.7%), R2 +16/−29. **mixed, 음 쪽** | 이 cohort의 AUC 이득은 +0.011뿐 |
| fpp_primary | FPP TD3 primary 재학습, seed 1–5 | 45개 | R1·R2 모두 +16/−29. **mixed** | 재학습에서 AUC 이득이 재현 안 됨(+0.004) |
| **fpp_n20** | FPP TD3 primary, **seed 11–30 (AUC 이득 확정 cohort)** | 360개 | **R1 SABER > Uniform 5/15 seed, sign test p=0.041, 중앙값 −1.9%. 후반 500k–1M 4/16, p=0.012.** SABER < eligible Random 5/15 | **일관된 음성.** Spearman(측정 점수, AUC 이득) ≈ 0.2 |

모든 cohort에서 공통으로 관찰된 것:

- κ = cos(∇_θQ(a^π), ∇_θQ(a^r))는 ψ=1 샘플에서 항상 더 높다(예: fpp_n20 20/0). 설계서 5.2절의 "Jacobian 결합"은 실제로 존재한다.
- 그런데도 실제 업데이트의 순효과는 없거나 음이다. 설계서 5.2절이 말한 대로 결합이 큰 것과 오차를 올바른 방향으로 줄이는 것은 다르다.
- SABER와 Uniform의 차이는 공통 Adam momentum 이동의 약 6–7%에 불과하다. 그래서 1차 예측과 실제 부호가 초반에 잘 맞지 않는다.

### 2.4 Actor 신호 측정 (fpp_n20, 2026-11-18)

- 설계서 6.4절 "actor와의 연결" 보조 endpoint와 가장 가까운 측정이다. 코드는 `fou/actor_signal.py`, 규칙은 `PROTOCOL_actor_signal_ko.md`에 있다.
- MuJoCo snapshot에서 결정적 rollout(H=50)으로 실제 Q^π(s,a)를 구했다. 비교한 지표는 세 가지다.
  - PA: a^π ± σ 쌍의 순위를 맞히는 정확도
  - GC: ∇_a Q와 실제 기울기의 cosine
  - AI: 기울기 방향으로 움직였을 때 실제 return 변화
- 결과는 **주 판정 mixed**다.
  - PA는 SABER > Uniform 9/7, 중앙값 0이다. GC는 일관된 음성으로 4/16이다.
  - 모든 critic의 PA가 0.49–0.59라 우연 수준에 가깝다. 100회 업데이트 뒤 arm 간 차이는 0.02 이하다.

### 2.5 누적 효과 분석 (학습 없음, 2026-11-18)

- `ICML2027/experiments/fpp_td3_cumulative_effect_n20_20261118/`
- 100k–400k 구간은 평가 성공률이 같은 구간이다. 여기서 지표 13개를 Holm 보정으로 비교했고, **critic Q 수준이 SABER에서 더 높다**(+0.54, 17/3)는 것만 남았다.
- 사후 분석: 400k에서 실제 return이 같은데 SABER의 Q0−G0가 HER보다 +2.87 크다(18/2). TD3의 과소추정이 덜하다는 뜻이다.
- 이것이 "가치 전파 경로" 후보다. 인과 검증에는 학습 도중 배정을 바꾸는 실험이 필요하다. 아직 하지 않았다.

### 2.6 설계서 실험 2를 그대로 실행할 경우 남은 것

- rollout 기준 HER-goal query bank(설계서 6.3절 1번)를 만들어 **실제 return 대비 MSE**를 주 endpoint로 측정하는 일
- A1 원 `.pt` 파일 확보. 설계서 6.2절의 조건이고, 이 PC에는 A1 checkpoint가 있어서 실험 4에서 썼다.
- 이미 FPP TD3 n20에서 frozen target 기준과 실제 rollout 기준(actor 신호)이 모두 음성이나 mixed다. 그래서 새 측정의 우선순위는 해석표 기준으로 판단해야 한다. 이 측정을 SABER에 유리한 결과가 나올 때까지 반복하는 용도로 쓰지 않는다.

## 3. 배정 대조 성능 실험 (설계서 범위 밖, 해석에 직접 영향)

모두 FPP TD3 primary 설정에서 새 seed 11–30(20쌍), 이 PC의 cuda로 돌렸다. 분석 규칙은 실행 전에 고정했다. paired bootstrap 100,000회, seed 20261113, CI 하한 > 0이면 우위로 판정한다.

| 대조 | 결과 (AUC) | 판정 | 위치 |
|---|---|---|---|
| SABER − HER | **+0.045 [+0.018, +0.073]**, 14/6, p=0.006 | SABER > HER | `paper_artifacts/fpp_td3_primary_rerun_20261110/n20_seeds11_30/` |
| SABER − count-matched (eligible 중 같은 개수를 무작위로 2배) | **+0.051 [+0.015, +0.086]**, 15/5 | SABER > count-matched | `paper_artifacts/fpp_td3_count_matched_n20_20261116/` |
| count-matched − HER | −0.006, 10/10 | 차이 없음 | 같은 곳 |
| SABER − context-count-only (context 최소 개수만 보고 a^π-bin 조회 없음) | **+0.046 [+0.013, +0.077]**, 12/8 | SABER > CCO | `paper_artifacts/fpp_td3_context_count_only_n20_20261117/` |
| CCO − HER | −0.001, 9/11 | 차이 없음 | 같은 곳 (CCO는 positive 비율 약 0.96이라 거의 uniform) |

해석:

- **같은 개수의 재가중만으로도, context 조건만으로도 이득이 없다. a^π-bin 조회가 들어가야 이득이 생긴다.** 설계서 7절 표 셋째 줄("SABER가 Uniform보다 낫지만 Random과 구별되지 않음")과는 반대 결과이고, 정확한 support 배정의 추가 가치를 지지한다.
- 그런데 2절에 따르면 그 이득은 정책 행동 Q 정확도를 거치지 않는다. 그래서 "배정이 효과가 있다"와 "설계서가 가정한 경로로 효과가 있다"는 분리해서 서술해야 한다.
- 이득은 학습 후반에 몰려 있다(FQ-AUC +0.137, 최종 성공률 +0.258). 효과 크기는 원 논문 +0.071의 약 2/3다.

## 4. 성능 재현성: stage 0과 stage 1 (진행 중)

### 4.1 배경

- 원 논문의 FPP TD3 이득(+0.071, seed 1–5)은 이 PC에서 5쌍으로 다시 돌리면 재현되지 않았다. CPU에서 +0.004, cuda에서 +0.017이었다.
- 같은 seed라도 실행마다 AUC SD가 0.04–0.06이라 5쌍으로는 판정할 수 없다. 그래서 새 seed 20쌍으로 다시 추정하기로 했다. FPP TD3는 3절 첫 줄의 결과가 나왔다.

### 4.2 Stage 0: 9개 cell inventory (2026-11-18, 완료)

`ICML2027/paper_artifacts/stage0_cohort_inventory_20261118/`. 기존 실행을 n20 규칙으로 다시 계산한 값이다.

| cell | 기존 쌍 수 | SABER − HER AUC |
|---|---:|---|
| FPP TD3 | 30 | +0.046 (CI가 0 제외) |
| FPP DDPG | 5 | +0.070 |
| FPP SAC | 5 | +0.112 (5/0) |
| DrawerOpen DDPG | 10 | +0.251 (CI가 0 포함, seed 6–10은 −0.045) |
| DrawerOpen TD3 | 10 | +0.321 (CI 하한 +0.006) |
| DrawerOpen SAC | 10 | +0.008 |
| DrawerClose DDPG | 10 | +0.035 |
| DrawerClose TD3 | 5 | +0.220 (분산 매우 큼) |
| DrawerClose SAC | 5 | −0.157 |

### 4.3 Stage 1: 새 seed 11–30, 20쌍 확장

- 폴더: `ICML2027/experiments/stage1_n20_extension_20261118/`. 규칙은 같은 폴더의 `PROTOCOL_ko.md`에 실행 전에 고정했다.
- 분석 규칙: AUC는 [10k, T] 구간 사다리꼴 면적이다. 판정 방식은 FPP TD3 n20과 같다. 결과는 `ICML2027/paper_artifacts/stage1_n20_extension_20261118/<cell>/`에 쓴다.

| 배치 | cell | 결과 | 상태 |
|---|---|---|---|
| 1 | FPP SAC | +0.023 [−0.018, +0.065], 12/8, p=0.30 | **판정 불가** (seed 1–5에서는 +0.112였음) |
| 1 | FPP DDPG | −0.015 [−0.052, +0.019], 11/9 | **판정 불가** (paper 시기 n5 +0.070은 옛 HER 코드 기준. 같은 코드 기준 HER 0.600, SABER 0.585) |
| 2 | DrawerOpen SAC | – | **진행 중** |
| 2 | DrawerClose SAC | – | **진행 중** |
| 2 | DrawerClose DDPG | – | **진행 중** |
| 보류 | DrawerOpen DDPG, DrawerOpen TD3, DrawerClose TD3 | – | 분산이 커서 20쌍이어도 CI 반폭이 ±0.22 수준. 시드를 더 늘릴지, 주장을 좁힐지 결정 필요 |

**배치 2 현황 (2026-11-24 04:17 UTC):**

- 실행은 120개(3 cell × HER/SABER × 20 seed, 각 2M step)다. 동시 18개를 돌리고 있고 실패는 0개다.
- 지금 SAC SABER 18개가 약 65%(126만–135만 step)까지 왔다. 이 18개는 2026-11-25 오전(KST)에 끝날 예정이다.
- 전체 완료 예상은 2026-12-05 전후다.
- 실행 코드는 기존 Drawer 결과를 만든 source를 origin/main 5facb92에서 그대로 꺼낸 것이다. 학습 전에 설정을 기존 실행 기록과 대조한다(PROTOCOL §7).
- 끝나면 `analyze.py --cells draweropen_sac drawerclose_sac drawerclose_ddpg`를 실행한다.

## 5. 설계서 실험 1에 관한 메모

- 설계서 4.3절에 적힌 2026-11-10 pilot 기록이 이 PC 작업과 같은 것이다. 복원은 통과했지만 fitting gate를 통과하지 못했다(36 fits 중 최대 training RMSE 1.786875). 4개 상태 중 2개는 기준 return 범위가 거의 0이었다.
- 4.4절 보조 분석의 일부 항목은 실험 4가 저장한 샘플별 기록으로 계산할 수 있다. 실험 4는 `dist`(‖a^r − a^π‖), `age`, ψ, κ를 남겼다. 이미 나온 값은 다음과 같다.
  - ψ=1 샘플 중 자기 replay 행동이 정책 bin에 속하는 비율은 평균 18%다(DrawerOpen DDPG의 SABER 학습 checkpoint).
  - ψ=1 샘플은 정책 행동에 더 가깝다(DrawerOpen 2M 기준 1.50 vs 2.06, 20/20 실행). 그리고 약간 더 최근 것이다.
  - 거리와 나이로 ψ를 예측한 ROC AUC는 0.65다(fpp_n20). 따라서 ψ는 단순한 최근성·근접성 지표로 환원되지 않는다.
- 코드 확인 사항도 있다(로드맵 2.2절).
  - Tail-512는 정확한 "최근 512개"가 아니다. ring buffer에서 덮어쓴 slot이 중복 제거 과정에서 예전 위치를 남긴다.
  - 로그의 `len_a_q_local`은 조기 종료 때문에 정의대로의 n_i가 아니다.
  - 초반 100k 이하에서는 gradient clipping이 거의 항상 걸린다.
- 아직 하지 않은 것도 있다.
  - Tail-512 판정과 같은 bucket의 전체 후보 조회를 비교해 놓친 비율을 세는 일
  - 최소 개수 2/4/8, K 3/5/7, 128/512/전체의 고정 replay 민감도

## 6. 지금 기준으로 쓸 수 있는 결론과 피할 결론

| 쓸 수 있음 | 피해야 함 |
|---|---|
| FPP TD3에서 SABER는 HER보다 AUC가 높다(20쌍, +0.045). 이득은 후반 수렴에 몰려 있다 | 원 논문의 +0.071을 그대로 확립된 값으로 인용 |
| 같은 개수의 무작위 재가중이나 context 조건만으로는 그 이득이 나지 않는다. a^π-bin 조회가 필요하다 | "재가중 자체가 효과의 원인" |
| 같은 checkpoint에서 배정만 바꾸면 정책 행동 Q 오차(frozen target 기준)가 줄지 않고, FPP TD3 이득 cohort에서는 오히려 늘어난다 | "SABER는 정책 행동의 Q 정확도를 높여서 성능을 올린다" |
| ψ=1 샘플은 정책 행동 쪽 Jacobian 결합(κ)이 일관되게 높다 | κ가 높으므로 올바른 방향으로 오차를 줄인다 |
| FPP SAC, FPP DDPG는 20쌍에서 판정 불가다 | 7개 cohort 모두 SABER > HER |

## 7. 남은 일 (우선순위 순)

1. **Stage 1 배치 2 완료 후 사전 규칙대로 분석한다**(약 2026-12-05). 그 결과에 따라 성능 주장의 범위(과제·backbone)를 확정한다.
2. 보류한 Drawer 3개 cell(DrawerOpen DDPG·TD3, DrawerClose TD3)을 어떻게 할지 정한다. seed를 크게 늘리거나 주장에서 빼는 것 중 하나다.
3. FPP TD3의 이득 경로를 찾는다. 후보는 "가치 전파 경로"(Q 과소추정 완화, 2.5절)다. 인과 검증에는 학습 도중 배정을 바꾸는 실험이 필요하다.
4. 설계서 실험 2의 rollout 기준 주 endpoint를 측정할지 결정한다. 이미 있는 음성·mixed 결과를 바꿀 가능성이 낮으면 우선순위를 낮춘다.
5. 설계서 실험 1을 재개하려면 fitting gate 문제부터 푼다. 4.4절의 보조 분석은 실험 4의 기존 기록으로 싸게 일부 계산할 수 있다.
6. 원고: 설계서 2절의 "두 행동" 설명 구조를 유지한다. 메커니즘 서술은 위 6절 표에 맞춘다.

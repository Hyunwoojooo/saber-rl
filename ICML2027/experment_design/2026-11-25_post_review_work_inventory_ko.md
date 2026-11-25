# 리뷰 이후 작업 전체 정리 (2026-11-25)

- 범위: 2026-09-25 AAAI 결정 이후의 실험, 진단, 문서 전부. GitHub에 올라간 것과 이 PC에만 있는 것을 모두 포함한다.
- 확인 기준:
  - `origin/main` 76753b2 (2026-11-25)
  - `origin/work/icml2027` fbda536 (2026-11-24)
  - 이 PC의 작업 트리와 `~/ICML2027/`
- 숫자는 각 artifact의 README와 분석 파일에서 옮겼다. 이 PC의 n20 결과는 `analysis.json`과 다시 대조했다.
- 구간은 모두 짝 bootstrap 95% 구간이다. 구간이 0을 포함한다는 것은 "효과 없음"이 아니라 "이 자료로는 못 가른다"는 뜻이다.
- 관련 문서: [논리 공백 로드맵](2026-11-10_logic_gap_roadmap_ko.md), [설계서 대응 문서](2026-11-24_experiment_status_vs_support_mechanism_design_ko.md), [11-05 미팅 자료](../meetings/2026-11-05_post_review_first_meeting_ko.md)

## 1. 자료가 있는 세 곳

| 위치 | 내용 | 올린 계정 |
|---|---|---|
| GitHub `main` | 공유 archive. `ICML_27/`(실험 6종, 인계 보고서, 설계서)와 과제별 폴더의 seed 확장 결과 | 리뷰 이후 커밋은 모두 rlatmddn0211 |
| GitHub `work/icml2027` | 이 PC의 작업 브랜치. `ICML2027/` 아래 설계, 코드, 결과 요약 | 모두 Hyunwoojooo |
| 이 PC에만 | `~/ICML2027/`의 초기 실험 8종(git 밖, 25 GB)과 모든 실험의 원자료·checkpoint | – |

커밋 계정은 누가 올렸는지만 알려 준다. 누가 어느 기계에서 실행했는지는 커밋만으로는 알 수 없다. `main`과 `work/icml2027`은 2026-10-06 커밋(84407b3)에서 갈라진 뒤 합쳐진 적이 없다.

## 2. 먼저 읽을 결론

1. **성능이 확립된 조건은 FPP TD3 하나다.** 새 seed 20쌍에서 SABER − HER AUC가 +0.045 [+0.018, +0.073]이다. DrawerOpen TD3는 10쌍에서 구간이 0을 겨우 벗어난다. 나머지 7개 조건은 구간이 0을 포함한다. 그중 DrawerClose SAC는 SABER 평균이 더 낮다.
2. **제출 당시의 5-seed 이득은 새 seed에서 대부분 줄었다.** 제출 논문이 주장한 7개 조건 가운데 6개에서 새 seed의 차이가 작아지거나 부호가 바뀌었다. 커진 것은 DrawerOpen TD3뿐이다(4절).
3. **기존 방법(PER, HGR, PAL)보다 낫다는 근거는 없다.** Drawer SAC에서는 PER가 더 높고, DDPG와 TD3에서는 구간이 모두 0을 포함한다(5절).
4. **"정책 행동 bin 조회가 있어야 이득이 난다"는 것은 FPP TD3에서만 확정됐다.** 같은 개수를 무작위로 올려 준 대조군과 context 조건만 쓴 대조군은 HER와 같았다. DrawerClose DDPG는 같은 방향이지만 구간이 0에 걸친다. DrawerOpen DDPG는 seed 집합에 따라 방향이 뒤집힌다(6절).
5. **"정책 행동의 Q가 더 정확해진다"는 설명은 어느 측정에서도 지지되지 않았다.** 6개 측정 설정 어디에서도 SABER 배정이 균일 배정보다 일관되게 낫지 않았다. 일관되게 나온 것은 두 가지다. ψ=1 샘플의 gradient 결합(κ)이 더 높다는 것, 그리고 FPP TD3에서 SABER critic의 Q 수준이 더 높다는 것이다. 300-step 실제 return과 비교하면 두 critic 모두 실제보다 높고, 초반의 Q 차이는 정책 가치의 차이가 아니라 SABER critic이 조금 더 낙관적인 것이다(7절, 7.7절).
6. **실패한 실행에서는 SABER가 HER와 같아진다.** 정책 출력이 한계값에 붙으면(actor 포화) 거의 모든 샘플이 support-positive가 되어 가중치가 전부 1이 된다(7.5절).
7. **지금 돌고 있는 것:** Drawer 3개 조건의 새 seed 20쌍(120개 실행 중 18개 완료). 12월 6일 전후에 끝난다(11절).

## 3. 시간순 흐름

| 시기 | GitHub `main` | 이 PC (`work/icml2027`, `~/ICML2027`) |
|---|---|---|
| 10월 초 | DrawerOpen DDPG 배정 비교(Support/Random/TD top-k), PAL, 조회 진단, Random-512. 10-06 현황 보고서 | – |
| 10-13 ~ 10-18 | – | 저장소 정리. policy-query 가치 진단 pilot(20k, 250k, 성숙 정책), optimizer·Q 오차 감사, λ 민감도 |
| 10-19 ~ 10-29 | `ICML_27/` archive 구성. Drawer SAC 4방법. DrawerOpen DDPG seed 6–10 확장 시작. 10-27 인계 보고서 | 배정 비교 seed 11–20 (30개 실행) |
| 10-30 ~ 11-05 | DrawerOpen DDPG 8방법 확장 완료(11-01). DrawerOpen TD3 확장 시작 | seed 11–20 실패 감사, 소스 재감사, A1(초기 학습 추적), 11-05 미팅 자료 |
| 11-06 ~ 11-12 | DrawerOpen TD3 10-seed 완료(11-07). DrawerOpen SAC 확장 | 실험 4(1차 유용성 측정) 4개 cohort. FPP TD3 재학습(cpu, cuda)에서 이득 재현 실패 |
| 11-13 ~ 11-18 | DrawerOpen SAC 10-seed 완료(11-13). DrawerClose DDPG 확장 시작 | FPP TD3 새 seed 20쌍으로 이득 확정. 이득 cohort 측정, 대조군 2종, actor 신호, 누적 효과, stage 0 |
| 11-18 ~ 11-25 | DrawerClose DDPG 8방법 확장 완료(11-21). support 메커니즘 설계서(11-24). DrawerClose TD3 4방법 확장 완료(11-25) | stage 1: FPP SAC·DDPG 20쌍 완료, Drawer 3개 조건 진행 중 |

## 4. 성능: SABER − HER (과제 × backbone)

AUC는 평가 성공률 곡선의 정규화 면적이다. "제출 당시"는 seed 1–5, "새 seed"는 리뷰 이후에 추가한 seed만의 값이다.

| 조건 | 제출 당시 (5쌍) | 새 seed만 | 현재 최선의 추정 | 판정 | 자료 |
|---|---|---|---|---|---|
| **FPP TD3** | +0.071 | 6–10: +0.023 / **11–30: +0.045 [+0.018, +0.073], 14/6** | 30쌍 합산 +0.046 [+0.023, +0.067] | **SABER > HER** | work: `fpp_td3_primary_rerun_20261110/n20_seeds11_30` |
| FPP SAC | +0.112 [+0.064, +0.161] | 11–30: +0.023 [−0.018, +0.065], 12/8 | 새 20쌍 | 판정 불가 | work: `stage1_n20_extension_20261118/fpp_sac` |
| FPP DDPG | +0.070 [−0.016, +0.193] | 11–30: −0.015 [−0.052, +0.019], 11/9 | 새 20쌍 | 판정 불가 | work: `stage1_n20_extension_20261118/fpp_ddpg` |
| DrawerOpen DDPG | +0.546 [+0.340, +0.741] | 6–10: −0.045 [−0.478, +0.456], 1/4 | 10쌍 +0.251 [−0.068, +0.544] | 판정 불가 | main: `drawer_open_ddpg_main_seed_extension_2m` |
| DrawerOpen TD3 | +0.235 | 6–10: +0.406, 4/1 | 10쌍 +0.321 [+0.017, +0.603] | 구간이 0을 겨우 제외 | main: `drawer_open_td3_main_seed_extension_2m` |
| DrawerOpen SAC | (제출 논문에 없음) | 1–5: +0.001 / 6–10: +0.015 | 10쌍 +0.008 [−0.060, +0.077] | 차이 거의 없음 | main: `drawer_open_sac_main_seed_extension_2m`. **n20 진행 중** |
| DrawerClose DDPG | +0.073 [+0.016, +0.143] | 6–10: −0.003 [−0.055, +0.053], 2/3 | 10쌍 +0.035 [−0.010, +0.087] | 판정 불가 | main: `drawer_close_ddpg_primary_seed_extension_2m`. **n20 진행 중** |
| DrawerClose TD3 | +0.220 [−0.216, +0.656] | 6–10: −0.165 [−0.515, +0.031], 2/3 | 10쌍 +0.027 [−0.261, +0.325] | 판정 불가 | main: `drawer_close_td3_main_seed_extension_2m` (11-25 공개) |
| DrawerClose SAC | (제출 논문에 없음) | 1–5: −0.157 [−0.419, +0.025], 1/4 | 5쌍 | SABER가 낮은 쪽 | main: `drawer_close_sac_her_saber_hgr_per_2m`. **n20 진행 중** |

읽을 때 주의할 점:

- **FPP DDPG의 제출 당시 값은 옛 HER 코드 기준이다.** 그때 HER는 0.539였다. 같은 코드 계열로 다시 돌린 20쌍에서는 HER가 0.600, SABER가 0.585다.
- **DrawerOpen TD3는 HER가 대부분 실패하는 조건이다**(HER 평균 AUC 0.137). 구간은 다중 비교 보정을 하지 않은 값이다.
- **DrawerClose TD3는 분산이 매우 크다.** 실행마다 AUC가 0.04에서 0.89까지 갈린다. 10쌍으로는 아무것도 가를 수 없다.
- **DrawerClose SAC에서 SABER는 5개 seed 모두 최종 성공률이 0이다.** 한 번은 성공률 0.9에 닿지만 유지하지 못한다.
- stage 0 표(11-18 작성)는 DrawerClose TD3를 5쌍(+0.220)으로 적었다. 11-25에 seed 6–10이 공개되어 위 값으로 바뀌었다.

**새 seed에서 줄어든 정도:**

| 조건 | 제출 당시 | 새 seed | 변화 |
|---|---|---|---|
| FPP TD3 | +0.071 | +0.045 (11–30) | 약 2/3로 줄어듦 |
| FPP SAC | +0.112 | +0.023 | 약 1/5로 줄어듦 |
| FPP DDPG | +0.070 | −0.015 | 사라짐 |
| DrawerOpen DDPG | +0.546 | −0.045 | 사라짐 |
| DrawerClose DDPG | +0.073 | −0.003 | 사라짐 |
| DrawerClose TD3 | +0.220 | −0.165 | 부호 바뀜 |
| DrawerOpen TD3 | +0.235 | +0.406 | 커짐 |

가장 단순한 설명은 seed 5개의 추정이 우연히 높은 쪽에 있었다는 것이다. 다른 원인(실행 환경 차이 등)은 8절에서 조사했고 확인된 것이 없다.

## 5. 기존 방법과의 비교

SABER − 비교 방법의 AUC 차이다. 양수면 SABER가 높다. 모두 `main`에 있다.

| 조건 | 쌍 | SABER − HGR | SABER − PER | 그 밖 |
|---|---|---|---|---|
| DrawerOpen DDPG | 10 | +0.087 [−0.181, +0.345] | +0.001 [−0.105, +0.149] | SABER − PAL: +0.107 [−0.153, +0.358] |
| DrawerOpen TD3 | 10 | +0.214 [−0.098, +0.502] | +0.063 [−0.150, +0.300] | |
| DrawerOpen SAC | 10 | −0.196 [−0.413, −0.012] | **−0.272 [−0.498, −0.069]** | PER, HGR 모두 10개 seed 전부 최종 성공률 1.0 |
| DrawerClose DDPG | 10 | −0.006 [−0.106, +0.075] | +0.007 [−0.036, +0.050] | |
| DrawerClose TD3 | 10 | −0.167 [−0.473, +0.141] | −0.005 [−0.246, +0.251] | |
| DrawerClose SAC | 5 | −0.108 [−0.235, +0.004] | **−0.384 [−0.624, −0.144]** | |

- Drawer SAC 두 조건에서 SABER는 PER보다 낮다. 다중 비교 보정 구간에서도 0 아래다.
- DrawerOpen SAC의 SABER − HGR은 보정 구간 [−0.460, +0.027]에서 0을 포함한다.
- DDPG와 TD3에서는 어느 쪽도 가를 수 없다.
- PAL 비교는 DrawerOpen DDPG 10쌍 하나뿐이다. FPP에서는 PER, HGR, PAL을 리뷰 이후에 돌리지 않았다.

## 6. 배정과 조회 대조군

"높은 가중치를 어디에 주는가"와 "support를 어떻게 찾는가"를 따로 바꾼 실험들이다.

### 6.1 DrawerOpen DDPG: Support / Random / TD top-k 배정 비교

세 방법 모두 같은 context 조건과 같은 개수를 쓰고, 높은 가중치를 줄 위치만 다르다.

| seed 집합 | SABER | Random | TD top-k | SABER − Random | SABER − TD | 위치 |
|---|---|---|---|---|---|---|
| 1–10 | 0.653 | 0.516 | 0.535 | +0.137 [−0.128, +0.384] | +0.118 [−0.055, +0.325] | main: `ICML_27/experiments/draweropen_ddpg_saber_vs_random_vs_td_topk` |
| 11–20 | 0.309 | 0.537 | 0.609 | −0.229 [−0.547, +0.105] | **−0.301 [−0.530, −0.084]** | **이 PC에만**: `~/ICML2027/draweropen_ddpg_assignment_extension_20261020` |
| 1–20 합산 | 0.481 | 0.527 | 0.572 | −0.046 [−0.250, +0.163] | −0.091 [−0.237, +0.051] | 11-05 미팅 자료 5절 |

- 처음 10개 seed의 SABER 우위가 다음 10개에서 뒤집혔다. seed 11–20의 SABER는 10개 중 6개가 최종 실패다.
- seed 11–20에는 HER 실행이 없어서 SABER − HER 표(4절)에는 들어가지 않는다.
- seed 11–20의 결과 보고서는 GitHub에 없다. work 브랜치에는 그 실행들의 사후 감사만 있다(7.5절).

### 6.2 원 논문의 mechanism 대조 4종, seed 6–10 확장

10쌍 합산 기준 SABER − 대조군이다. 모두 `main`에 있다.

| 대조군 | 무엇을 바꿨나 | DrawerOpen DDPG | DrawerClose DDPG |
|---|---|---|---|
| Replay-Action Query | 정책 행동 대신 replay 행동으로 조회 | +0.219 [−0.092, +0.516] | +0.033 [−0.002, +0.077], 6/10 |
| Shuffled-Policy Query | 배치 안에서 정책 행동을 섞어 조회 | +0.029 [−0.200, +0.237] | +0.028 [−0.019, +0.073], 7/10 |
| Random-Support | 배치 전체에서 같은 개수를 무작위 선택 | +0.086 [−0.024, +0.220] | +0.033 [−0.003, +0.072], 7/10 |
| Context-Eligible Count-Matched | context 조건을 통과한 것 중 같은 개수를 무작위 선택 | +0.136 [−0.119, +0.374] | +0.019 [−0.015, +0.056], 6/10 |

- 여덟 칸 모두 SABER 평균이 높지만 구간은 모두 0을 포함한다.
- DrawerClose DDPG는 네 대조군에서 차이가 +0.02 ~ +0.03으로 비슷하고 구간 하한이 0 바로 아래다. 지금 돌고 있는 DrawerClose DDPG n20은 HER와 SABER만 돌리므로 이 대조군 질문에는 답하지 않는다.

### 6.3 FPP TD3: 새 seed 20쌍 대조군 (이 PC)

| 비교 | AUC 차이 | +/− | 판정 |
|---|---|---|---|
| SABER − HER | +0.045 [+0.018, +0.073] | 14/6 | SABER > HER |
| SABER − count-matched | **+0.051 [+0.015, +0.086]** | 15/5 | SABER > count-matched |
| count-matched − HER | −0.006 [−0.034, +0.023] | 10/10 | 차이 없음 |
| SABER − context-count-only | **+0.046 [+0.013, +0.077]** | 12/8 | SABER > context-count-only |
| context-count-only − HER | −0.001 [−0.028, +0.025] | 9/11 | 차이 없음 |

- count-matched는 context 조건을 통과한 샘플 중 SABER와 같은 개수를 무작위로 골라 2배 가중한다.
- context-count-only는 context 최소 개수 조건만 보고 정책 행동 bin 조회를 하지 않는다. positive 비율이 약 0.96이라 거의 균일 가중이다.
- 이득은 학습 후반에 몰려 있다. final-quarter AUC는 +0.137, 최종 성공률은 +0.258이다(SABER 0.80, HER 0.54).
- 위치: work `paper_artifacts/fpp_td3_count_matched_n20_20261116`, `fpp_td3_context_count_only_n20_20261117`

### 6.4 조회 방식: Tail-512와 Random-512 (DrawerOpen DDPG, main)

| 측정 | Tail-512 | Random-512 |
|---|---|---|
| 전체 조회 기준 positive 중 놓친 비율 (최종 checkpoint 10개) | 41.0% | 31.2% |
| 온라인 학습 AUC (10 seeds) | 0.653 | 0.430 |
| 평가 성공이 한 번도 없는 실행 | 1/10 | 4/10 |

- Tail − Random AUC는 +0.222 [−0.081, +0.518]이다.
- Random-512가 support를 덜 놓치는데도 학습 평균은 Tail이 높다. "support를 더 정확히 찾으면 학습이 좋아진다"는 관계는 성립하지 않았다.
- 전체 조회로는 query의 90%가 positive다. Tail-512는 54%만 positive로 판정한다. 즉 ψ=0의 상당수는 "512개 안에서 못 찾음"이다.

### 6.5 종합

| 조건 | "정확한 support 위치가 필요한가"에 대한 근거 |
|---|---|
| FPP TD3 | 필요하다. 20쌍, 대조군 2종 모두 구간이 0을 제외 |
| DrawerClose DDPG | 같은 방향. 10쌍, 대조군 4종 모두 구간이 0에 걸침 |
| DrawerOpen DDPG | 미정. seed 1–10은 SABER 쪽, 11–20은 반대쪽 |
| 그 밖 6개 조건 | 대조군 실험 없음 |

## 7. 메커니즘 진단

질문은 하나다. **support-positive 샘플의 오차를 더 맞추면, 정책이 실제로 쓰는 행동에서의 critic이 좋아지는가?**

### 7.1 Policy-query 가치 진단 pilot (10-13 ~ 10-15, 이 PC에만)

- 방식: checkpoint 하나에서 critic과 optimizer를 복사하고 배정만 바꿔 1회 업데이트한다. 정책을 실제로 실행해 얻은 return을 기준으로 Q 오차를 잰다.
- 대상: DrawerOpen DDPG, 학습 seed 1개. HER 20k, HER 250k, HER·SABER 각각 500k/1M/1.5M/2M.
- 결과: 시점마다 방향이 달랐고 일관된 SABER 우위가 없었다.
- 대표 사례(SABER 2M 저장본): 학습 배치의 Bellman residual은 1.210 → 1.106으로 줄었는데, 정책 행동의 Q MSE는 183.26 → 186.83으로 늘었다.
- 부속 감사: optimizer 감사(20k, 250k)와 2M Q 오차 감사. 결과를 Adam 문제 하나로 설명할 수 없었다.
- 위치: `~/ICML2027/draweropen_ddpg_policy_query`, `..._her_source_pilot`, `..._her_250k_pilot`, `..._mature_policy_pilot_20261014`, `..._optimizer_audit`, `..._q_error_audit_20261015`

### 7.2 실험 4: 1차 유용성 측정 (11-09 ~ 11-17, work)

- 방식: 저장된 checkpoint에서 같은 배치 계획으로 critic을 분기한다. 분기는 Uniform, SABER, eligible Random, TD top-k, inverse(ψ=0에 2배)다. 1회와 100회 native Adam 업데이트 뒤 정책 행동 query 오차를 비교한다.
- 오차 기준은 고정 target network의 1-step Bellman target이다. 실제 return이 아니다.
  - R1: 정책을 노이즈 없이 실행한 상태, 원래 목표
  - R2: 탐색 노이즈가 섞인 상태, HER로 바꾼 목표. actor가 학습 때 묻는 분포에 더 가깝다
- 규칙은 cohort마다 측정 전에 고정했다.

| cohort | 대상 | 100회 업데이트, SABER vs Uniform (SABER로 학습한 checkpoint) | 판정 |
|---|---|---|---|
| A1 초반부 | DrawerOpen DDPG, seed 101–103, 10k–500k, 48개 | 중앙값 −2.8%. Random·inverse와 구별 안 됨 | 이점 없음 |
| 2M 후반부 | DrawerOpen DDPG, 2M 저장본 38개 | 성공 실행 20개 중 SABER 우위 R1 8개, R2 10개 | mixed |
| fpp_incl | FPP TD3 inclusive-future, seed 6–10, 90개 | R1 +19/−26, R2 +16/−29 (checkpoint 단위) | mixed, 음 쪽 |
| fpp_primary | FPP TD3 재학습, seed 1–5, 90개 | R1·R2 모두 +16/−29 | mixed, 음 쪽 |
| **fpp_n20** | **FPP TD3 seed 11–30 (이득 확정 cohort), 360개** | **R1: 5/15 seed, p=0.041, 중앙값 −1.9%. R2: 10/10, 중앙값 +0.2%** | **R1 일관된 음성, R2 차이 없음** |

fpp_n20의 나머지 결과:

- 후반(500k–1M)의 R1은 4/16(p=0.012)이다. eligible Random과 비교해도 후반 R1은 5/15(p=0.041)로 SABER가 낮다.
- HER로 학습한 checkpoint에 SABER 배정을 적용해도 R1은 2/18(p<0.001)로 낮다.
- inverse 배정은 Uniform과 구별되지 않는다.
- seed별 측정 점수와 그 seed의 AUC 이득 사이 순위 상관은 약 0.2다.

모든 cohort에서 공통인 것:

- **κ는 ψ=1 샘플에서 항상 높다.** κ는 정책 행동과 replay 행동에서의 critic gradient 방향이 얼마나 비슷한지다. fpp_n20에서 20/0이다.
- 그런데도 실제 업데이트의 순효과는 없거나 음이다. 결합이 큰 것과 오차를 올바른 방향으로 줄이는 것은 별개다.
- 초반(100k 이하)에는 critic gradient clipping이 거의 항상 걸린다. 이 구간에서 SABER는 step 크기가 아니라 방향만 바꾼다.

위치: 설계 `experment_design/04_first_order_usefulness`, 코드 `experiments/draweropen_ddpg_first_order_usefulness_20261109`, 결과 `paper_artifacts/first_order_usefulness_20261109`

### 7.3 Actor 신호 측정 (11-17 ~ 11-18, work)

- 대상: fpp_n20의 checkpoint 360개
- 방식: MuJoCo 상태를 복원해 결정적 rollout(50 step)으로 실제 return을 구한다. critic이 정책 행동 주변의 행동 순위와 기울기 방향을 맞히는지 본다.
- 결과: 주 판정 mixed.
  - 순위 정확도(PA)는 SABER > Uniform 9/7/동률 4, 중앙값 0이다.
  - 모든 critic의 PA가 0.49–0.59로 우연 수준에 가깝다.
  - 100회 업데이트 뒤 배정 간 차이는 0.02 이하다.
- 위치: `paper_artifacts/first_order_usefulness_20261109/actor_signal_n20`

### 7.4 누적 효과 분석 (11-18, 새 학습 없음, work)

- 대상: FPP TD3 seed 11–30의 학습 로그와 저장본. 평가 성공률이 아직 같은 100k–400k 구간을 본다.
- 지표 13개를 Holm 보정으로 비교했다. 해석할 수 있는 것 중 남은 것은 **critic Q 수준이 SABER에서 더 높다**는 것 하나다(+0.54, 17/3, 보정 p=0.031).
- 탐색·데이터 분포·actor 포화·기울기 크기 지표는 모두 유의하지 않다.
- 사후 확인: 실제 return이 같은 400k 시점에서 SABER critic의 Q − return이 HER보다 2.87 높다(18/2). 처음에는 이를 "TD3의 과소추정이 덜하다"로 읽었다. 이 return은 50 step에서 자른 값이었다. 300-step return으로 다시 잰 결과 두 critic 모두 실제 가치보다 높았으므로 이 해석은 틀렸다(7.7절).
- 이것이 지금 남은 유일한 경로 후보다. 인과를 보려면 학습 도중 배정을 바꾸는 실험이 필요하다. 아직 하지 않았다.
- 위치: `paper_artifacts/fpp_td3_cumulative_effect_n20_20261118`

### 7.5 실패 분석

| 작업 | 위치 | 알게 된 것 |
|---|---|---|
| 실패와 support 포화 (배정 비교 seed 1–10, 30개 실행) | main: `ICML_27/experiments/draweropen_ddpg_retrieval_and_failure_diagnostics` | 평가 성공이 전혀 없던 7개 실행 모두 support 포화가 관측됐고 나머지 23개에서는 없었다. 포화는 500k 이후에 나타났다 |
| seed 11–20 실패 심층 감사 | work: `diagnostics/assignment_extension_deep_audit_20261030` | 실패한 SABER 6개의 최종 정책은 행동이 한계값에 붙고 actor gradient가 0이었다. Random·TD 실패 일부도 같았다 |
| seed 1–10 대 11–20 소스 재감사 | work: `diagnostics/assignment_source_reaudit_20261030` | 핵심 코드 51개 파일이 byte 단위로 같다. 반전을 코드 차이로 설명할 수 없다 |
| A1: 초기 학습 추적 | work: `experiments/draweropen_ddpg_actor_dynamics_20261031`, `paper_artifacts/a1_actor_dynamics_20261031` | DrawerOpen DDPG, 4방법 × seed 101–103 × 500k. 500k까지의 AUC는 SABER가 가장 낮았다(0.012, HER는 0.132). 한 실행에서 일시적 전체 포화를 봤지만 영구적이지 않았다 |
| 2M 후반부 측정의 부수 결과 | work: `paper_artifacts/first_order_usefulness_20261109/late_ext2m` | 실패 실행 10개 중 9개는 support-positive 비율이 0.965 이상이다. 가중치가 거의 전부 1이 되어 SABER가 HER와 같아진다 |

### 7.6 코드를 읽고 확인한 사실 (work, 로드맵 2.2절)

- **Tail-512는 정확한 "최근 512개"가 아니다.** ring buffer가 slot을 덮어쓰면 중복 제거가 예전 위치를 남긴다.
- ψ=1 샘플 가운데 자기 replay 행동이 정책 행동과 같은 bin인 비율은 평균 18%다. 나머지 82%는 "같은 context의 다른 기록"이 근거다.
- ψ=1 샘플은 정책 행동에 더 가깝고 조금 더 최근 것이다. 그러나 거리와 나이로 ψ를 맞히는 정확도(ROC AUC)는 0.65다. ψ는 단순한 최근성·근접성 지표가 아니다.
- 로그의 `len_a_q_local`은 조기 종료 때문에 정의대로의 개수가 아니다. ψ 값 자체는 정의와 같다.
- Drawer SAC의 관측에는 원래 목표 좌표가 남아 있고 HER는 따로 붙인 목표만 바꾼다. 네 방법이 모두 같은 입력을 쓴다. 영향은 미확인이다.

### 7.7 정책 행동의 가치 편향 (11-25, 새 학습 없음, work)

- 대상: FPP TD3 seed 11–30의 checkpoint 8개 시점(320개). 정책 행동의 Q1과 300-step 실제 return G를 비교했다. 학습은 50-step 시간 제한을 넘어 bootstrap하므로 50 step에서 자른 return은 비교 기준으로 맞지 않았다.
- 주 판정(원래 목표, 초반 100k–400k):
  - 정책의 실제 가치: SABER − HER = +0.25 [−1.07, +1.52]. 차이가 없다.
  - 편향 Q1 − G: HER +5.46, SABER +6.42. 두 critic 모두 실제보다 높다. 차이는 +0.96 [+0.07, +1.81](13/7, p=0.050)로 경계선이다.
  - 절대 오차 차이는 +0.44 [−0.30, +1.21]이다. SABER critic이 더 정확하지 않다.
- 시점별로 보면 400k에서 편향 차이가 +2.89 [+1.29, +4.49](17/3)로 뚜렷하다. 1M에서는 SABER 정책의 실제 가치가 높고(+6.54) 절대 오차가 작다. 이 시점에는 정책이 이미 달라 가중의 효과로 읽을 수 없다.
- 해석:
  - TD3 HER critic은 이 구간에서 과소추정 상태가 아니다. 그래서 "과소추정 완화"는 맞는 이름이 아니다.
  - 7.4절의 초반 Q 수준 차이는 SABER critic이 같은 가치의 정책을 조금 더 낙관적으로 잡는 것이다.
  - "정책 행동 Q가 정확해진다"는 설명은 Bellman target, actor 신호, 실제 return의 세 기준 모두에서 지지되지 않는다.
- 판정 규칙을 적기 전에 seed 11의 400k 한 쌍을 봤다. 그 쌍에서는 SABER 정책의 실제 가치가 높았지만 20쌍 전체에서는 재현되지 않았다.
- 사후 확인(규칙에 없던 것): ψ = 1 샘플의 |TD 잔차|가 ψ = 0보다 약 1.7배 크다(20/20, 두 arm 모두). 부호 있는 잔차 차이는 0을 포함한다.
- 위치: `experiments/fpp_td3_value_bias_n20_20261125`, 결과 `paper_artifacts/fpp_td3_value_bias_n20_20261125/README_ko.md`

## 8. 재현성 조사 (FPP TD3, 이 PC)

| 실행 | 설정 | HER | SABER | 차이 |
|---|---|---|---|---|
| 원 실행 (2026-06) | seed 1–5, cuda | 0.372 | 0.442 | +0.071 |
| 재학습 (11-10) | 같은 seed, cpu | 0.336 | 0.341 | +0.004 [−0.015, +0.023] |
| 재학습 (11-12) | 같은 seed, cuda | 0.394 | 0.411 | +0.017 [−0.035, +0.071] |
| 새 seed (11-13 ~ 14) | seed 11–30, cuda | 0.371 | 0.416 | **+0.045 [+0.018, +0.073]** |

- 같은 seed를 세 번 돌린 AUC의 표준편차가 HER 0.039, SABER 0.061이다. 원 논문의 효과 크기와 비슷하다. 그래서 5쌍으로는 이 효과를 볼 수 없다.
- 5쌍 재학습의 "재현 실패"는 표본이 작아서 나온 결과였다. 20쌍에서는 이득이 확인됐다.
- 실행 기계에 대해 확인된 것: 공동 연구자의 감사 기록에 GB10 기기가 4대 있고 라이브러리 버전은 모두 같다. 6월의 FPP TD3 원 실행이 어느 기기였는지는 기록이 없다.
- DrawerOpen DDPG 코드는 같은 seed면 기기가 달라도 평가 곡선이 비트 단위로 같다(stage 0에서 확인).

## 9. 설정 민감도

| 실험 | 범위 | 결과 | 위치 |
|---|---|---|---|
| 가중 강도 λ | DrawerOpen DDPG, λ = 0.5 / 1 / 2, seed 1–3, 2M (새 학습 8개) | AUC 0.478 / 0.745 / 0.538. 기본값 1이 가장 높다. 3 seed의 탐색 결과 | 이 PC에만: `~/ICML2027/draweropen_ddpg_lambda_sensitivity_20261015` |

리뷰 이후 한 민감도 실험은 이것 하나다. 최소 개수, 조회 수, bin 수, context 구간은 하지 않았다. bin 수 3/5/7 비교는 제출 자료(FPP TD3)에 이미 있다.

## 10. 문서와 설계

| 문서 | 위치 | 내용 |
|---|---|---|
| 10-06 리뷰 이후 실험 현황 | main `ICML_27/2026-10-06_post_review_experiment_status_ko.md` | 초기 상태 기록 |
| 인계 보고서 (10-27) | main `ICML_27/postAAAI_Review.md` | 리뷰 요약, 본문 수정 목록, 신규 100개 실행의 결과와 해석 |
| 저장소 README (11-14 기준) | main `README.md`, `ICML_27/README.md` | DrawerOpen 결과 총괄표, archive 범위 |
| support 메커니즘 검증 설계 (11-24) | main `ICML_27/SABER_support_mechanism_study_design_ko.md` | 이론 2개와 실험 2개의 설계안. 아직 실행 전 |
| 실험 설계 01–03 | work `experment_design/01~03` | policy-query 진단, 설정 민감도, 공통 context 전이. 02와 03은 실행하지 않음 |
| 추가 실험 검토 (10-30) | work `experment_design/2026-10-30_additional_experiments` | 그때까지의 결과와 다음 후보 정리 |
| 실험 4 설계 (11-09) | work `experment_design/04_first_order_usefulness` | 1차 유용성 측정 |
| 미팅 자료 (11-05) | work `meetings/2026-11-05_post_review_first_meeting_ko.md` | main과 이 PC 결과를 합친 설명 자료 |
| 논리 공백 로드맵 (11-10 ~) | work `experment_design/2026-11-10_logic_gap_roadmap_ko.md` | 비어 있는 논리 고리, 할 일과 상태. 계속 갱신 중 |
| 설계서 대응 문서 (11-24) | work `experment_design/2026-11-24_...` | 설계서의 네 검증과 기존 측정의 대응 |

## 11. 진행 중과 대기

**2026-11-25 07:11 UTC에 순서를 바꿨다.** Stage 1 배치 2(DrawerOpen SAC, DrawerClose SAC, DrawerClose DDPG, 120개)에서 SAC 두 조건을 보류했다. SAC SABER 한 개가 62시간 걸려 남은 SAC만 장비 전체로 약 7일이었고, 결과가 주장 범위를 바꾸기 어렵다고 봤다. 사유와 처리는 `ICML2027/experiments/stage1_n20_extension_20261118/PROTOCOL_ko.md` 8절에 있다.

**지금 도는 것 (이 PC, 앞 묶음이 끝나는 대로 다음 묶음이 시작):**

| 순서 | 실험 | 실행 | 위치 |
|---|---|---|---|
| 1 | FPP TD3 target 비관성 검정. TD3의 target을 `min + α·|Q1' − Q2'|`로 바꾼 HER·SABER, α = 0.25·0.5, seed 11–30 | 80개, 1M | `ICML2027/experiments/fpp_td3_target_pessimism_20261125/` |
| 2 | DrawerOpen TD3 HER·SABER, seed 11–30 | 40개, 2M | 같은 stage 1 폴더, PROTOCOL 9절 |
| 3 | DrawerClose DDPG HER·SABER, seed 11–30 | 40개, 2M | 같은 stage 1 폴더, PROTOCOL 7절 |
| 완료 | FPP TD3 정책 행동의 가치 편향 측정. 기존 20쌍 checkpoint에서 Q와 300-step 실제 return 비교 (학습 없음). 11-25 09:41 UTC에 끝남, 결과는 7.7절 | checkpoint 320개 | `ICML2027/experiments/fpp_td3_value_bias_n20_20261125/` |

1번 실험의 이름은 "비관성 검정"이지만, 7.7절 결과에 따라 보는 것은 "target을 덜 비관적으로 해서 Q 수준을 더 올리면 HER도 좋아지는가"다. 판정 규칙은 바꾸지 않았다(그 PROTOCOL 6절).

**보류한 것:**

- DrawerOpen SAC, DrawerClose SAC의 새 seed 20쌍. SABER seed 11–19는 끝나 있고 보존했다. HER는 시작하지 않았다.
- DrawerOpen DDPG, DrawerClose TD3의 새 seed 20쌍. 실행 간 분산이 커서 20쌍이어도 구간 반폭이 ±0.21 수준인데 추정 이득이 그보다 작거나(DrawerClose TD3 +0.027) 새 seed에서 유지되지 않았다(DrawerOpen DDPG, seed 6–10만 −0.045).

**설계만 있고 실행하지 않은 것:**

- 설계서의 실험 1(주변 데이터 제거·복구) 본 측정. 설계서를 쓴 쪽에서 pilot만 했다
- 설계서 실험 2의 실제 return 기준 측정
- 공통 context 전이, 최소 개수·조회 수 민감도
- 같은 조건의 wall-clock 비교

## 12. 실행 수

**GitHub `main`** (2M step 실행):

| 묶음 | 새 실행 |
|---|---:|
| `ICML_27/` 6종 (배정 30, PAL 10, Random-512 10, Drawer SAC 40) | 90 |
| DrawerOpen DDPG 8방법 × seed 6–10 | 40 |
| DrawerOpen TD3 4방법 × seed 6–10 | 20 |
| DrawerOpen SAC 4방법 × seed 6–10 | 20 |
| DrawerClose DDPG 8방법 × seed 6–10 | 40 |
| DrawerClose TD3 4방법 × seed 6–10 | 20 |
| 합계 | 230 |

앞의 170개는 main README의 집계와 같다. DrawerClose 60개는 각 artifact README에서 내가 더한 값이다.

**이 PC:**

| 묶음 | 실행 | 길이 |
|---|---:|---|
| policy-query pilot (HER 20k, HER 250k, 성숙 정책 HER·SABER) | 4 | 20k / 250k / 2M |
| λ 민감도 | 8 | 2M |
| 배정 비교 seed 11–20 | 30 | 2M |
| A1 초기 학습 추적 | 12 | 500k |
| FPP TD3 재학습 (cpu 10, cuda 10) | 20 | 1M |
| FPP TD3 새 seed 20쌍 | 40 | 1M |
| FPP TD3 count-matched, context-count-only | 40 | 1M |
| Stage 1 배치 1 (FPP SAC, FPP DDPG) | 80 | 1M |
| Stage 1 배치 2 중 Drawer SAC SABER (보류, seed 11–19만 완료) | 18 | 2M |
| FPP TD3 target 비관성 검정 (11-25 시작) | 80 (진행 중) | 1M |
| DrawerOpen TD3, DrawerClose DDPG 새 seed 20쌍 (대기) | 80 (대기) | 2M |

## 13. GitHub에 없는 것

| 자료 | 크기 | 비고 |
|---|---:|---|
| `~/ICML2027/draweropen_ddpg_assignment_extension_20261020` | 12 GB | **배정 비교 seed 11–20의 결과 보고서와 원자료.** 6.1절의 반전 결과가 여기에만 있다 |
| `~/ICML2027/draweropen_ddpg_lambda_sensitivity_20261015` | 8.7 GB | λ 민감도 결과 |
| `~/ICML2027/`의 policy-query pilot과 감사 6종 | 약 4 GB | 7.1절. 실행 코드 `draweropen_ddpg_policy_query` 포함 |
| `ICML2027/experiments/*/outputs/` | 약 530 GB | 모든 실행의 로그와 checkpoint. 요약만 `paper_artifacts/`에 있다 |

배정 비교 seed 11–20과 λ 민감도는 결과 숫자가 11-05 미팅 자료에 들어 있을 뿐, 재분석할 수 있는 파일은 GitHub에 없다. 이 디스크가 유일한 사본이다.

## 14. 리뷰 질문별 현재 답

| 질문 | 지금까지의 근거 | 남은 것 |
|---|---|---|
| 기존 재가중·우선순위 방법과 무엇이 다르고 더 나은가 | PAL, PER, HGR과 직접 비교했다. 어느 조건에서도 SABER 우위가 확정되지 않았고 Drawer SAC에서는 PER가 낫다 | FPP에서의 PER·HGR·PAL 비교가 없다. 성능 우위로는 답하기 어렵다 |
| 왜 이 support 기준인가 | FPP TD3에서는 정책 행동 bin 조회가 있어야 이득이 난다(대조군 2종). 그러나 그 경로가 Q 정확도는 아니다 | 실제 경로. 후보는 SABER critic의 초반 낙관(7.4, 7.7절). target 검정(11절 1번)이 이 경로를 시험한다 |
| 정책이 쓰는 critic이 정확해지는가 | 6개 측정 설정 어디에서도 일관된 개선이 없다. FPP TD3 20쌍의 300-step 실제 return 기준으로도 더 정확하지 않다(7.7절) | 설계서 실험 2의 HER 목표 query 기준 측정. 7.7절의 사후 목표 결과도 같은 방향이다 |
| seed와 backbone 범위 | 9개 조건 모두 10쌍 이상 또는 진행 중. 확립된 것은 FPP TD3 하나 | Drawer 3개 조건 n20 결과(12월 초). 주장 범위 축소 |
| Tail 조회의 근거 | Tail이 support를 더 놓치는데 학습 평균은 더 높다. 구간은 0 포함 | 이유 미상. Tail-512의 실제 동작이 "최근 512개"도 아니다 |
| 설정 민감도 | λ 3값 pilot뿐 | 최소 개수, 조회 수, context 구간 |
| 수작업 context | 설계만 있다 | 공통 context 전이 실험 |
| 계산 비용 | 기록된 속도로는 SABER가 HER보다 느리다(Drawer SAC에서 약 20 대 38 steps/s) | 같은 조건의 시간 비교 |
| 설명의 복잡함 | 설계서 2절이 "두 행동" 중심의 Method 재구성안을 제시한다 | 원고 재작성 |

## 15. 11-24 문서에서 고친 것

[설계서 대응 문서](2026-11-24_experiment_status_vs_support_mechanism_design_ko.md)에 틀린 곳이 두 군데 있어 고쳤다.

1. **제거·복구 pilot의 출처.** 설계서 4.3절의 2026-11-10 CPU pilot을 이 PC의 작업이라고 적었다. 이 PC에는 그 pilot의 코드도 기록도 없다. 설계서를 쓴 쪽의 작업이다.
2. **fpp_n20 결과의 범위.** "이득 cohort에서 정책 행동 Q 오차가 오히려 늘었다"고만 적었다. 정확히는 R1(원래 목표, 노이즈 없는 상태)에서 일관된 음성이고, R2(HER 목표)에서는 10/10으로 차이가 없다. 설계서 실험 2의 주 평가 대상은 HER 목표 query라서 R2 쪽에 가깝다.

## 16. 결정이 필요한 것

1. **성능 주장의 범위.** 지금 자료로 방어되는 문장은 "FPP TD3에서 SABER는 HER보다 높고, 그 이득에는 정책 행동 bin 조회가 필요하다"다. 12월 초 Drawer n20 결과를 본 뒤 나머지 조건의 서술을 정한다.
2. **메커니즘 서술.** "정책 행동 Q가 정확해진다"는 설명은 내려야 한다. 대신 쓸 설명은 아직 후보 하나뿐이다. 그 후보도 "과소추정 완화"가 아니라 "초반의 조금 더 높은 낙관"이며 구간 하한이 0에 가깝다(7.7절). target 검정 결과(11월 27일 전후)를 보고 정한다.
3. **설계서 실험 2를 실행할지.** 이미 있는 측정이 음성이거나 mixed다. 실제 return 기준이라는 차이만으로 결과가 달라질 가능성과 비용을 견줘야 한다.
4. **분산이 큰 Drawer 3개 조건.** seed를 크게 늘릴지, 주장에서 뺄지.
5. **`main`과 `work/icml2027`의 공유.** 두 브랜치가 10-06 이후 갈라져 있다. 이 PC의 n20 결과와 측정 결과는 `main`만 보는 사람에게 보이지 않는다.
6. **이 PC에만 있는 자료의 보존.** 배정 비교 seed 11–20과 λ 민감도의 요약을 work 브랜치에 올릴지.

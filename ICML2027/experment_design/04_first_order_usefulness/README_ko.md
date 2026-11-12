# 실험 4. SABER의 가중이 정책 조회 오차를 줄이는 방향인가: 1차 유용성 u 측정

- 작성일: 2026-11-09
- 상태: 설계 v0.1. 아래 본문은 측정 전에 쓴 그대로다.
  - 2026-11-09: SABER·HER 학습 checkpoint 48개의 초반부 측정을 마쳤다. [결과 요약](../../paper_artifacts/first_order_usefulness_20261109/early_phase_a1/REPORT_ko.md). Random·TD 학습 checkpoint 48개(강건성)와 R3는 하지 않았다.
  - 2026-11-10: 후반부 저장본으로 확장할 준비를 마쳤다. [후반부 protocol](../../experiments/draweropen_ddpg_first_order_usefulness_20261109/PROTOCOL_late_ko.md), [전체 로드맵](../2026-11-10_logic_gap_roadmap_ko.md).
- 대상: DrawerOpen–DDPG, A1 checkpoint(`ICML2027/experiments/draweropen_ddpg_actor_dynamics_20261031`).
- 연결 문서:
  - [실험 1. policy-query 가치 진단](../01_policy_query_value_diagnostic.md): 같은 저장 상태에서 critic만 분기하는 틀
  - 「SABER support 정당화 분석 및 검증 설계」(2026-11-05) §5, §9.4–9.5: u의 정의와 1차 항등식. **저장소에는 없다.** 2026-11-05 작업 세션에 붙여 넣은 문서이며, 원문을 이 폴더에 넣어 두어야 한다.
  - [리뷰 이후 첫 미팅 자료](../../meetings/2026-11-05_post_review_first_meeting_ko.md) §10, §15 우선 1(A2)

## 1. 묻는 것

SABER 논리에서 비어 있는 고리는 다음 한 문장이다.

> support-positive 샘플의 replay-action 오차를 더 맞추면, actor가 쓰는 정책 행동 a^π에서의 critic이 좋아진다.

이 문장을 다음 측정 가능한 부등식으로 바꿔 검사한다.

> 같은 checkpoint, 같은 minibatch에서 ψ = 1 샘플의 gradient가 ψ = 0 샘플보다 정책 조회 오차 R_π를 더 줄이는 방향인가? 즉 **ū₊ > ū₋** 인가?

장기 성공률(AUC)은 묻지 않는다. 이 실험은 **한 번 또는 몇 번의 critic 업데이트가 어느 방향을 향하는지**만 본다.

## 2. 측정하는 양

### 2.1 정의

| 기호 | 정의 | 코드 |
|---|---|---|
| z_i^r | 샘플 i의 critic 입력 (o_i, g̃_i, a_i^r) | `concat(observation, desired_goal_train)`, `action_exec` |
| ℓ_i | δ_i² (제출 코드처럼 ½ 없음) | `agent.py:102` |
| R_π | 정책 조회 query bank에서의 오차 (3절) | 새로 구현 |
| J_i | ∇_φ Q_φ(z_i^r) | |
| u_i | ⟨∇_φ R_π, ∇_φ ℓ_i⟩ = 2 δ_i ⟨J_i, ∇_φ R_π⟩ | **jvp 한 번**으로 배치 전체를 계산 |

u_i > 0이면 샘플 i 하나로 SGD 한 step을 밟았을 때 R_π가 줄어든다.

### 2.2 1차 항등식

Ω_i는 SABER 가중치(평균 1), p는 배치의 양성 비율, M은 ψ = 1 집합, E는 context-eligible 집합(n ≥ 4)이다. 정의만으로 다음이 성립한다.

| 비교 | ⟨∇R_π, g_A − g_B⟩ | 양수면 |
|---|---|---|
| SABER − HER | p(1−p)/(1+p) · (ū₊ − ū₋) | SABER가 R_π를 더 줄임 |
| SABER − eligible Random (기댓값) | p/(1+p) · (ū_M − ū_E) | a^π bin 일치가 eligible 안에서 추가 정보를 줌 |
| SABER − eligible TD Top-k | p/(1+p) · (ū_M − ū_TD) | support가 TD 크기보다 유용한 위치를 고름 |
| 역방향(ψ = 0을 키움) − HER | p(1−p)/(2−p) · (ū₋ − ū₊) | 부호가 SABER와 반대 |

따라서 **ū를 그룹별로 재기만 하면 Random 기댓값과 역방향은 별도 branch 없이 얻는다.** TD Top-k는 결정적 mask라 바로 계산한다. 이 항등식은 무한소 SGD step에서만 정확하다. 실제 step의 문제는 4절에서 다룬다.

## 3. R_π: 무엇을 "정책 조회 오차"로 볼 것인가

actor가 학습에서 실제로 Q를 읽는 지점은 **replay 상태 + HER goal + a^π**다. 그런데 replay에는 시뮬레이터 상태가 없어서 그 지점의 정답을 만들 수 없다. 그래서 세 단계로 근사한다.

| 수준 | query | 기준값 | 비용 | 역할 |
|---|---|---|---|---|
| **R1 (주)** | checkpoint actor를 노이즈 없이 실행한 상태, 원래 goal, a^π | 고정 target network의 1-step Bellman target. a^π를 실제로 실행했으므로 전이가 정확함 | checkpoint당 1초 미만 | 기본 측정 |
| R2 | 노이즈 섞인 behavior 실행 상태, 미래 achieved goal로 relabel, 그 goal에서의 a^π | snapshot 복원 후 a^π를 1-step 실행한 Bellman target | R1의 수 배 | actor의 학습 query 분포에 더 가까움 |
| R3 (보조) | 성숙 정책 pilot의 저장 snapshot | 긴 rollout return(Q^π 근사) | 큼 | frozen target이 아닌 실제 가치 기준 |

- R2의 복원은 `~/ICML2027/draweropen_ddpg_policy_query/saber_qdiag/simulator.py`의 `DrawerSimulator.snapshot/restore`를 쓴다. 이 adapter는 MetaWorld 3.1.1, MuJoCo 3.3.0, Gymnasium 1.3.0 기준으로 감사되었고 `.venv`도 같은 버전이다.
- R3는 `~/ICML2027/draweropen_ddpg_mature_policy_pilot_20261014`의 HER/SABER seed 1, 500k–2M checkpoint의 저장 snapshot과 기준값을 재사용한다.
- **재지 않는 것:** ∇_a Q(x, a^π)의 정확도. DDPG actor가 실제로 쓰는 것은 값이 아니라 이 기울기다. 기준 기울기에는 유한차분 rollout이 필요해 후속 단계로 둔다. 이번 결론은 "정책 조회 **값** 오차"에서 멈춘다.

## 4. 업데이트 기하: 이론 step과 실제 step

타당성 점검에서 세 가지를 확인했다(8절).
- **clipping이 사실상 매번 걸렸다.** 24개 arm-배치 중 23개는 clip 전 norm이 10을 확실히 넘었고(11.3–65.7), 1개는 경계값 10.0이었다.
- **SABER 쪽 norm이 더 컸다.** 가중치가 residual과 상관되어 있기 때문이다.
- **SGD 1차 부호와 실제 Adam step의 부호가 12번 중 8번만 일치했다.**

그래서 같은 u를 네 가지 기하로 계산하고 실제 branch와 대조한다.

| 기하 | 예측식 | 의미 |
|---|---|---|
| G1 SGD | ⟨∇R, g_S − g_U⟩ | 2.2절 이론 그대로 |
| G2 clip | 10 · ⟨∇R, g_S/‖g_S‖ − g_U/‖g_U‖⟩ (clip이 걸린 경우) | clipping이 항상 걸리면 SABER는 step 크기가 아니라 **방향**만 바꾼다 |
| G3 Adam + clip | ⟨∇R, P(c(g_S) − c(g_U))⟩, P = lr(1−β₁)/((1−β₁^{t+1})(√v̂ + ε)) | 저장된 Adam moment에서 새 gradient가 들어가는 몫. 공통 momentum 항은 두 arm에서 상쇄됨 |
| 실제 1-step | critic과 Adam 상태를 복제해 한 step | G3의 정답 |
| 실제 K-step | K ∈ {1, 10, 100}. actor와 target은 고정, 배치 순서는 arm 간 공유 | 한 step의 작은 차이가 누적되는지 |

- 실제 1-step에서 R 변화의 대부분은 두 arm에 공통인 momentum 이동이다. 100k 예: 공통 변화 +0.070, arm 차이 −0.003. 한 step으로는 검정력이 낮다는 뜻이므로 K-step이 필요하다.
- **H4(5절)를 통과한 기하만** 논문의 1차 분석에 쓴다.

## 5. 사전 고정 가설

| 가설 | 판정 양 | 기본 기하/수준 |
|---|---|---|
| H1 방향 | ū₊ − ū₋ > 0 | G3, R1 |
| H2 정확한 bin 일치 | ū_M − ū_E > 0, ū_M − ū_TD > 0 | G3, R1 |
| H3 전달 | ψ = 1의 결합도 κ_i가 더 큼. ‖a^r − a^π‖와 \|δ\| 구간을 맞춘 뒤 비교 | 기하 무관 |
| H4 예측 타당성 | G3 예측 부호와 실제 1-step 부호의 일치율 ≥ 80% | 실제 branch |

**H3의 결합도:** κ_i = cos(∇_φ Q(x_i, a_i^π), ∇_φ Q(x_i, a_i^r)). 같은 상태에서 자기 replay 행동을 맞추면 자기 정책 행동의 Q가 얼마나 같이 움직이는지를 나타낸다. 정답 기준이 필요 없으므로 R의 정의와 무관하다.

**집계 규칙(실행 전 고정):**
- 단위는 학습 seed다. 배치와 query는 checkpoint 안의 반복이므로 독립 표본으로 세지 않는다.
- checkpoint마다 64개 배치의 평균을 쓴다. 0 < p < 1인 배치만 H1, H2에 들어간다.
- A1은 arm마다 3 seeds라 CI로 유의성을 주장하지 않는다. "일관됨"은 **세 seed 모두에서 8개 시점 중 6개 이상이 같은 부호**인 경우로 정의한다.
- 유리한 시점만 고르지 않는다. 정한 시점은 모두 보고한다.

## 6. 분해 지표: ū가 왜 다른지

u_i = 2 δ_i ⟨J_i, ∇R⟩이므로, ū₊ > ū₋는 다음 두 가지 이유로 생길 수 있다.
- ψ = 1의 |δ|가 크기 때문이다(크기 효과). 논문의 "support-positive는 high-TD에 농축"과 같은 현상이다.
- ψ = 1의 gradient 방향이 R을 더 잘 겨냥하기 때문이다(방향 효과).

SABER의 주장에 필요한 것은 방향 효과다. 그래서 다음을 함께 기록한다.

| 지표 | 목적 |
|---|---|
| \|δ\| 사분위별 ū₊, ū₋ | 크기 효과를 통제한 비교 |
| cos(∇ℓ_i, ∇R) 그룹 평균 | 크기를 뺀 방향 정렬 |
| ū_M vs ū_TD | TD 크기로 고른 위치와 직접 비교 (H2) |
| κ_i, ‖a_i^r − a_i^π‖ | 전달 경로 (H3) |
| 자기 support 여부 | 샘플 자신의 a^r bin이 a^π bin과 같은 양성과, 다른 이웃만으로 양성인 경우를 분리 |
| E \ M, M, E 밖 세 그룹 | eligibility와 bin 일치의 기여를 분리 |
| phase(approach/contact/opening/open)별 | "어떤 상황에서 유용한가" |

## 7. 대상과 규모

| 항목 | 설정 |
|---|---|
| checkpoint | A1의 4 arm × 3 seeds × {10k, 20k, 50k, 100k, 200k, 300k, 400k, 500k} = **96개**. 0과 4,800은 학습 전이라 제외하고, 5k/6k/8k는 선택 사항 |
| 주 분석 | SABER와 HER로 학습한 checkpoint. HER checkpoint의 ψ는 "SABER였다면 무엇을 키웠을까"라는 반사실 값이다 |
| 강건성 | Random, TD로 학습한 checkpoint |
| 배치 | checkpoint당 64개, seed `900_000 + b` |
| R1 query | 20 episodes, 최대 4,000개, reset seed `50_000 + e` |
| R2 query | 20개 behavior episode에서 1,024개를 추출 |
| K-step | K ∈ {1, 10, 100}. 배치 순서는 seed로 고정하고 arm 간 공유 |
| 보조(R3) | 성숙 pilot HER/SABER seed 1, 500k/1M/1.5M/2M, 8개 |
| 예상 비용 | 점검 기준 checkpoint당 4 s(4 batches, 1,000 queries). 64 batches, 결합도, R2, K = 100까지 넣으면 checkpoint당 수 분으로 추정. 전체는 CPU 몇 시간 수준이고 새 학습은 없음 |

## 8. 타당성 점검 (2026-11-09). 결과가 아님

[feasibility_probe.py](feasibility_probe.py)를 A1 checkpoint 3개에 실행했다. 각 4 batches, R1 query 1,000개(5 episodes)다. 원본 출력은 [feasibility_20261109.txt](feasibility_20261109.txt)에 있다.

| 확인 항목 | 결과 |
|---|---|
| jvp로 구한 u와 per-sample gradient 내적 | 4개 샘플 일치 (상대오차 1e-4 이내) |
| 1차 예측과 float64 SGD 유한차분 | η = 1e-6, 1e-7, 1e-8에서 예측으로 수렴. 예: +184.2 대 +182.4, +184.0, +184.2 |
| η = 1e-4 float32 | 1차 근사가 깨짐. R 변화(약 0.4)가 R(0.23)보다 큼 |
| support mask | 제출 코드 `LiteWeightComputer.compute`에서 직접 얻음 |
| 비용 | checkpoint당 3.2–3.8 s |
| clipping | 24개 arm-배치 중 23개에서 걸림(11.3–65.7). 1개는 경계값 10.0 |
| G1 부호와 실제 Adam 1-step 부호 | 8/12 일치 (SABER 100k 4/4, SABER 500k 1/4, HER 300k 3/4). G3로 다시 확인해야 함 |

같은 실행에서 보인 숫자는 seed 1개, checkpoint 3개, 배치 4개에서 나온 **예비 관찰**이다. 결과로 인용하지 않는다.
- ū₊ > ū₋: 12개 배치 중 10개
- κ₊ > κ₋: 12개 중 12개 (예: 0.665 대 0.561)
- ‖a^r − a^π‖: 12개 모두 ψ = 1 쪽이 작음

## 9. 구현 계획

**재사용:**
- `saber_qdiag`: checkpoint 검증, 제출 scanner와 parity가 확인된 `support_masks`, `select_mask`, `weights`, `update_critic`, `DrawerSimulator`
- A1 `code/src/metacog_lite_v0`: 모델·replay·task 클래스

**새로 만들 것:** u의 jvp 계산, G2/G3 예측, R1/R2 query bank, 결합도 지표, K-step branch, 집계와 보고서.

**실행 전 테스트:**
1. jvp u와 per-sample gradient 내적의 일치
2. 1차 예측과 float64 유한차분의 수렴
3. mask가 `LiteWeightComputer`, numba scanner와 일치
4. 원본 actor·target·critic·Adam·replay의 hash가 불변
5. 2.2절 항등식을 임의 가중치로 수치 확인
6. 11/5 문서 §8.1의 반례(2-context affine critic)를 재현해 Γ < 0이 나오는지

**순서:**
1. 구현과 테스트
2. SABER/HER seed 101의 8개 시점
3. 96개 전체
4. R2
5. R3 보조
6. 보고서

각 단계 결과는 새 출력 경로에 쓰고 덮어쓰지 않는다.

## 10. 결과를 어떻게 쓸 것인가

| 결과 | 논문과 수식에 주는 의미 |
|---|---|
| H1과 H2가 모두 일관되게 성립 | support가 유용한 업데이트를 고른다는 직접 근거. 11/5 문서의 L1 모델로 **왜** 그런지를 수식으로 설명 |
| H1은 성립, H2는 불성립 | 이득이 eligibility나 residual 크기에서 옴. a^π bin 일치라는 SABER 고유 신호는 지지되지 않음. 주장을 "context-population 가중"으로 좁힘 |
| H1이 시점이나 phase에 따라 다름 | "support가 언제 유용한가"에 대한 답. A1의 포화 시점과 연결 |
| H1 불성립 | 메커니즘 주장 철회. 11/5 문서 §8 반례 유형으로 실패 조건을 설명 |
| H4 불성립 | 1차 분석을 논문에 쓰지 않고 K-step branch 결과만 보고 |

## 11. 해석 주의

- R1은 노이즈 없는 on-policy 상태와 원래 goal이다. actor의 학습 query(replay 상태 + HER goal)와 분포가 다르다. 두 수준이 다르게 나오면 둘 다 보고한다.
- frozen-target 기준은 Q^π 정확도가 아니다. R3로 일부만 확인한다.
- 값 오차가 개선돼도 ∇_a Q가 개선됐다는 뜻은 아니다.
- 한 step 효과는 작다. 결론은 K-step과 시점 추세로 낸다.
- A1은 3-seed pilot이다. 확증 실험이 아니다.

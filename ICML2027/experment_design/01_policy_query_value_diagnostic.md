# 실험 1. SABER가 정책이 사용하는 가치 예측을 개선하는가?

- 작성일: 2026-10-08
- 상태: 설계 초안 v0.1. 아래 규모와 설정은 제안이며, 학습·진단을 실행하거나 예약한 문서가 아니다.
- 대상: DrawerOpen–DDPG. 공동저자가 준비 중인 DrawerOpen / DrawerClose SAC 성능 비교와 역할을 분리한다.
- 연결 문서: [설정 민감도](02_support_hyperparameter_sensitivity.md), [공통 context 전이](03_common_context_transfer.md)

## 1. 쉽게 설명하면

SABER는 과거 데이터 중 일부를 더 강하게 학습시킨다. 이 실험은 그렇게 학습했을 때 **정책이 지금 선택하려는 행동의 가치 예측도 더 정확해지는지** 확인한다.

같은 저장 상태를 네 개로 복사하고, 같은 데이터에 다른 가중치만 적용한다. 정책은 움직이지 않고 critic만 업데이트한다. 이후 독립적인 시뮬레이터 실행에서 얻은 누적 보상과 critic의 예측을 비교한다.

| 항목 | 설계 |
|---|---|
| 리뷰의 질문 | 왜 현재 정책 행동의 support로 replay 데이터의 학습 가중치를 정해야 하는가? |
| 바꾸는 것 | critic loss의 데이터별 가중치 배정 |
| 고정하는 것 | 초기 critic·optimizer, 정책, target network, replay, minibatch, HER goal, Bellman target |
| 주 평가 | 현재 정책 행동에서 rollout 기준값에 대한 Q 오차의 업데이트 전후 변화 |
| 첫 작업 | 시뮬레이터 상태 복원과 시간 제한 처리가 평가 목적에 맞는지 검증 |
| 얻을 수 있는 근거 | 같은 학습 상태에서 특정 가중 규칙이 가치 예측에 주는 직접적인 단기 효과 |
| 별도 질문 | 장기 성공률, 실패 확률, actor gradient의 정확도, 다른 알고리즘에서의 효과 |

## 2. 현재 근거와 구현 확인 사항

사용자가 제공한 2026-10-06 보고서에는 DrawerOpen–DDPG의 SABER / Random assignment / TD-error Top-k / PAL 각 10회 결과가 있다. SABER의 평균 AUC가 높지만 세 비교의 95% CI는 모두 0을 포함한다. 이 자료는 본 진단의 동기이며, 이미 Q 정확도를 검증한 자료는 아니다.

문서 작성 시 로컬 HEAD는 `7becc22`이고 사용자 작업 변경분이 존재한다. 다음은 **현재 로컬 파일에서 확인한 사실**이다. 이 작업공간에는 보고서의 `experiments_post_review/` 폴더가 없으므로, 리뷰 이후 실행 코드와 동일하다고 가정하지 않는다.

| 확인한 파일 | 관찰 | 설계에 반영할 점 |
|---|---|---|
| [trainer.py](../../metaworld_Drawer_Open_DDPG/code/src/metacog_lite_v0/trainer.py)의 `checkpoint_payload` | 모델·optimizer·replay·일부 RNG를 저장하지만 전체 시뮬레이터 상태는 payload에 없음 | 기존 checkpoint의 replay observation만으로 물리 상태를 복원했다고 주장하지 않음 |
| [envs.py](../../metaworld_Drawer_Open_DDPG/code/src/metacog_lite_v0/envs.py)의 `snapshot` | 환경 ID·차원·설정 등을 기록 | `env_snapshot.json`을 물리 상태 저장본으로 취급하지 않음 |
| [agent.py](../../metaworld_Drawer_Open_DDPG/code/src/metacog_lite_v0/agent.py)의 `update` | critic 뒤에 actor와 target도 업데이트 | 본 진단은 별도의 critic-only update 경로 필요 |
| [trainer.py](../../metaworld_Drawer_Open_DDPG/code/src/metacog_lite_v0/trainer.py)의 target 구성 | timeout에서 bootstrap, success 조기 종료는 설정으로 구분 | 200-step까지의 보상합을 그대로 현재 critic의 정답으로 쓰지 않음 |

실행 구현은 리뷰 이후 cohort의 고정 source와 effective config를 우선 확보해 작성한다. 로컬 파일은 구조를 파악하는 참고이며, 사용자 변경분이나 기존 실험을 수정하는 출발점으로 삼지 않는다.

## 3. 비교군과 고정 조건

한 minibatch의 context-eligible 위치 집합을 E, 정책 행동 support-positive 위치 집합을 M, 그 크기를 각각 m, k라고 한다. M은 E의 부분집합이다. 배치 내 중복 transition도 서로 다른 위치로 취급한다.

| arm | 큰 가중치를 받을 위치 |
|---|---|
| HER | 모든 위치에 가중치 1 |
| SABER | M |
| Context-Eligible Random | E에서 정확히 k개를 균일 비복원 선택 |
| Context-Eligible TD-error Top-k | E에서 업데이트 전 절대 TD 오차가 큰 k개; 경계 동률만 무작위 처리 |

세 가중 arm은 `w_raw = 1 + z`, `Omega = w_raw / mean(w_raw)`를 사용한다. HER도 같은 squared-residual loss와 optimizer를 사용하며 계수만 1이다. TD-error Top-k의 backbone 역시 DDPG다.

- 정책 행동은 support 조회와 평가에 사용한다. 회귀 residual은 실제 replay 행동에서 계산한다.
- target actor와 target critic은 원 checkpoint의 값을 그대로 복제해 고정한다. target을 현재 정책이나 Monte Carlo return으로 교체하지 않는다.
- 동일 batch의 target tensor를 한 번 계산해 모든 arm에 전달한다. actor와 target network는 모든 진단 update 동안 고정한다.
- critic parameter와 Adam의 momentum·variance·step·parameter group을 모두 복제한다. tensor storage를 공유하지 않는다.
- 학습률, loss convention, gradient clipping 등은 해당 source config와 동일하게 유지한다. 로컬 DDPG는 critic gradient clipping 상한 10을 사용한다.
- k=0이면 모든 가중 arm은 uniform이다. k=m이면 세 가중 arm은 E를 선택하지만, E가 batch 전체가 아닐 때 HER와 같지는 않다.
- Random은 별도 RNG를 사용한다. 실제 TD loss forward의 detached residual로 순위를 정하며, 다른 target을 다시 계산하지 않는다.

## 4. 단계 A: 상태 복원 pilot

### 4.1 평가 상태를 새로 수집한다

기존 checkpoint에서 정책과 replay를 불러오되, 평가 상태는 **독립적인 환경에서 고정 정책을 실행해 새로 수집**한다. 이때 평가용 전체 상태를 저장한다. 이 방법은 과거 replay observation에서 잃어버린 물리 상태를 추측할 필요가 없다.

평가 trajectory는 critic 학습용 replay에 넣지 않는다. 관측 밖의 시뮬레이터 상태는 복원에만 사용하며 SABER나 critic의 입력에 추가하지 않는다.

### 4.2 저장·복원 범위

MuJoCo는 물리 상태 외에도 control, mocap, warmstart 등 적분에 영향을 주는 상태를 구분한다. 실행에 설치된 버전의 state API를 확인하고, 지원되는 integration state를 기준으로 복원한다. qpos와 qvel만 복사하는 것으로 충분하다고 가정하지 않는다. [MuJoCo 공식 문서](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#state-and-control)

그 밖에 환경이 실제로 사용하는 다음 상태도 감사한다.

- task/reset parameter, 변경된 model property, 목표 위치와 observation history
- MetaWorld의 episode counter와 내부 task 상태
- Gymnasium wrapper의 elapsed-step 상태와 자동 reset 동작
- 환경과 wrapper의 RNG, controller가 추가로 유지하는 상태
- 설치 버전·모델 자산 hash·환경 config

API 이름과 필요한 필드는 설치 버전을 확인해 확정한다. 최신 문서의 API가 현재 설치판에 있다고 가정하지 않는다.

### 4.3 pilot 규모와 통과 기준

1. 성능을 기준으로 고르지 않은 가용 checkpoint 2개에서 총 20개 상태를 수집한다.
2. 각 상태의 원 환경과 복원 환경에 동일한 20-step 행동열을 적용한다.
3. observation, qpos/qvel, reward, goal-success, terminated/truncated를 매 step 비교한다.
4. 그중 사전 선택한 4개 상태는 뒤의 1,024-step 평가 경로까지 재현성을 확인한다.

초기 수치 기준은 observation 최대 절대 오차 `1e-6`, float64 qpos/qvel `1e-9`, reward와 이산 판정의 불일치 0이다. 허용오차 변경이 필요하면 설치 정밀도에 대한 근거를 남기고, 방법별 효과를 보기 전에 확정한다. 복원 누락에 의한 차이를 환경의 확률성으로 처리하지 않는다.

복원이 실패하면 Q 비교를 진행하지 않고 snapshot adapter를 수정한다. 이 pilot에는 새로운 2M 본학습이 필요하지 않다.

## 5. 단계 B: 무엇을 Q의 기준값으로 삼을 것인가?

### 5.1 정책과 행동을 정확히 맞춘다

평가 상태 x에서 학습 입력과 같은 observation o와 goal g를 구성한다. 첫 행동은 고정된 현재 정책 `a = pi(o, g)`이며, 이후에도 같은 정책을 따른다. 탐색 잡음은 넣지 않는다.

기준값은 `G_L = sum_{t=0}^{L-1} gamma^t r_t`이다. critic은 `Q(o, pi(o,g), g)`로 평가한다. 원 코드처럼 observation과 goal을 이어 붙이는 입력 구조를 유지하며, 새로운 horizon 입력을 network에 추가하지 않는다.

Replay 행동으로 얻은 reward와 next state를 다른 정책 행동의 정답으로 재사용하지 않는다. 예측 오차는 독립적인 rollout reference에 대한 오차로 명명하며, 모든 숨은 상태에 대한 정확한 Q를 계산했다는 표현은 피한다.

### 5.2 timeout bootstrap과 일치시킨다

보고서의 기본 조건은 `gamma=0.98`, success 조기 종료 없음, timeout bootstrap이다. 외부 시간 제한에 따른 truncation과 실제 terminal state는 bootstrap에서 구분해야 한다. [Gymnasium 공식 문서](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

본 설계는 이 학습 목표를 **외부 시간 제한을 넘어 계속되는 task dynamics의 할인 return**으로 해석할 수 있는지 먼저 감사한다. 진단 전용 환경에서 인위적인 시간 제한만 해제하고, 물리·관측·보상·실제 termination 규칙은 유지한다. 원래 시간 제한 이전 transition parity와 내부 time-limit 동작을 검증한다. reset 뒤의 다음 episode 보상을 이어 붙이지 않는다.

이 해석이 성립할 때 제안하는 rollout 길이는 query 이후 `L=1,024` steps이다. 보상이 {-1, 0}이고 gamma=0.98이면 생략한 tail의 절대값 상한은 다음과 같다.

`epsilon_tail = gamma^L / (1-gamma)`

| rollout 길이 | 생략한 return 절대값 상한 |
|---|---:|
| 200 | 약 0.8794 |
| 512 | 약 0.001610 |
| 1,024 | 약 5.18e-8 |

이 값은 할인합의 산술적 상한이다. 복원 오류나 확률적 rollout의 추정 오차를 포함하지 않는다. 실제 terminal state에서는 종료 후 기여를 0으로 둔다. query 시점의 남은 horizon이 200보다 짧으면 그 시점에서 끊는 편향은 더 커질 수 있다.

시간 제한 뒤의 일관된 dynamics를 만들 수 없다면 finite-horizon return을 같은 Q의 정답이라고 부르지 않는다. 독립적인 finite-horizon 진단이나 fixed-target residual 분석으로 설계를 분리하고, true policy-value accuracy 주장은 보류한다. 원 본학습의 종료 규칙을 조용히 바꾸지 않는다.

### 5.3 반복 rollout과 목표 종류

- 복원 상태와 정책을 고정했을 때 환경이 결정적임을 확인하면 query당 reference trajectory 1개를 사용한다. 똑같은 trajectory의 반복을 독립 표본으로 세지 않는다.
- 환경에 실제 transition 확률성이 있으면 query당 8개 독립 환경 RNG trajectory를 제안한다. 모든 비교 arm이 같은 reference 평균과 추정 불확실성을 공유한다. noise를 새로 넣어 인위적으로 확률성을 만들지 않는다.
- 주 평가는 환경의 원래 desired goal이다. HER goal 평가는 별도 부 분석으로 둔다.
- HER 평가를 추가할 때는 독립 평가 trajectory의 strict-future achieved goal을 사용하고, 조건 정책과 reward를 그 goal에 맞춘다. 내부 목표 변경이 observation이나 dynamics를 바꾸는지 감사한다. 원래 goal 평가와 합쳐 하나의 수치로 보고하지 않는다.

## 6. 단계 C: checkpoint와 query 구성

다음은 본 진단의 제안 규모다. source inventory를 확인한 뒤 결과를 보기 전에 manifest를 고정한다.

| 항목 | 제안 |
|---|---|
| checkpoint 출처 | 동일 protocol의 SABER와 HER |
| 독립 training seeds | 각 출처 1–10 |
| 학습 시점 | 250k / 1M / 2M의 사전 지정 checkpoint |
| 평가 상태 | checkpoint당 32개 독립 reset episode에서 t=0,50,100,150의 행동 직전 snapshot, 최대 128개 |
| 정책 | 각 checkpoint의 현재 deterministic actor; 평가와 fork에서 고정 |
| 주 업데이트 진단 | 16개 독립 minibatch 각각에서 원 checkpoint로 다시 돌아가 1-step fork |
| 부 업데이트 진단 | 원 checkpoint에서 새로 시작하는 공통 20-minibatch sequence, 총 20 critic updates |

Zero-evaluation seed를 포함한다. best checkpoint나 성공한 seed만 선택하지 않는다. 예정 시점이 없으면 가장 잘 나온 인접 시점으로 교체하지 않는다. Q 결과를 열기 전에 모든 seed에 공통으로 가능한 시점으로 protocol을 개정하거나, 자료 부족을 명시한다.

평가 episode가 실제 terminal state에 먼저 도달하면 이후 예정 snapshot을 만들지 않는다. 빠진 시점과 종료 이유를 기록하고, 성공한 trajectory로 대체하지 않는다. 이 경우 episode별 유효 query를 먼저 평균하고 episode를 동일 가중치로 집계한다. 복원 실패에 의한 누락은 이러한 정상 termination과 구분해 평가를 중단한다.

동일 조건의 HER checkpoint가 없으면 먼저 재사용 감사를 한다. SABER 출처만으로 pilot을 할 수 있으나, 그 결과는 SABER가 만든 replay/policy에 조건부인 결과다. 두 출처를 균형 있게 평가한 본 설계가 완료됐다고 보고하지 않는다. 새 source training은 이 진단 예산에 자동 포함하지 않는다.

Reset seed는 같은 training seed·학습 시점에서 출처 간 공통으로 사용하되, 정책이 달라 trajectory가 같아지는 것은 아니다. 진단 seed는 기존 학습·평가 seed와 분리한다. `SeedSequence`에 고정 master seed `20261008`과 용도별 ID를 넣고 reset, batch/HER, Random 배정, TD 동률, bootstrap stream을 분리한다. 최종 숫자 목록을 manifest에 보존한다.

## 7. fork 실행 순서

1. source checkpoint·config·replay/index를 읽고 hash와 구조를 검증한다.
2. 평가 snapshot, query action, rollout reference를 생성해 고정한다.
3. 학습 batch와 HER goal, terminal mask, target tensor를 미리 고정한다.
4. 네 arm의 critic과 optimizer를 원 checkpoint에서 독립 복제한다.
5. actor·replay가 같으므로 해당 batch의 E, M, k를 공유한다. Random mask는 고정된 전용 stream으로 만든다.
6. 각 arm의 critic loss만 업데이트하고 독립 query에서 Q를 다시 계산한다.
7. 주 진단의 다음 minibatch는 모두 원 checkpoint에서 새로 시작한다. 앞선 fork의 학습을 이어받지 않는다.
8. 20-step 부 진단은 별도 복제본에서 수행한다. target은 계속 고정하지만 TD Top-k의 residual 순위는 해당 arm의 현재 critic에서 매 step 다시 계산한다.

원 학습용 `agent.update()`를 그대로 호출하지 않는다. actor/target parameter hash가 변하지 않는지, replay가 진단 trajectory로 오염되지 않았는지 검증한다. TD selector를 위해 다른 target이나 별도의 확률적 forward를 생성하지 않는다.

## 8. 평가 지표와 통계

### 8.1 주 지표

동일 query 집합에서 `error = mean((Q_prediction - G_reference)^2)`로 계산한다.

- 방법별 개선량: `G_method = error_before - error_after_method`; 양수면 오차 감소.
- 고정 비교 방향: `D_SABER-control = G_SABER - G_control`; 양수면 SABER가 더 크게 개선.
- 세 primary contrast: SABER−HER, SABER−Random, SABER−TD Top-k의 **1-step 개선량**.

공통 초기 critic을 사용하므로 같은 fork 안의 개선량 차이는 업데이트 후 오차 차이와 일치한다. 여러 출처의 초기 Q 정확도를 직접 방법 순위로 비교하지 않는다.

### 8.2 부 지표

| 지표 | 목적 |
|---|---|
| MAE, signed bias | 큰 오류에 대한 민감도와 과대/과소 추정 방향 |
| 20-step 후 Q 오차 | 단일 update 관찰이 짧은 반복 업데이트에도 나타나는지 |
| clipping 전후 critic gradient norm, clipping 빈도 | gradient 크기 차이 파악 |
| 실제 parameter update norm과 HER 대비 방향 cosine | mean-one weight가 같은 optimizer step을 보장하지 않는다는 점 점검 |
| eligible/support 비율, k/m, weight 분산 | 어느 batch에서 배정 규칙이 실제로 달랐는지 |
| replay–policy action 거리 | 거리에 따른 효과를 탐색적으로 분석 |
| held-out fixed-target residual | 실제 rollout 기준 Q 오차와 Bellman residual 변화를 구분 |

행동 거리 분석은 실제 replay 행동이 있는 별도 paired-query 자료가 필요하다. 새로 수집한 on-policy snapshot에 존재하지 않는 과거 행동을 임의로 붙이지 않는다. 원 replay 상태 복원이 검증되거나 별도 행동 분기 자료를 만들었을 때만 수행한다. parameter 방향의 cosine은 norm 0일 때 NA다.

### 8.3 집계 단위

- query는 평가 episode 안에서 평균한 뒤 episode를 동일 가중치로 집계하고, batch fork를 평균해 checkpoint별 값을 만든다. 이어 시점과 출처를 같은 가중치로 평균해 training seed별 차이를 만든다.
- 같은 seed의 모든 출처·시점·방법을 묶어 paired bootstrap한다. 독립 표본 수는 training seed 수이며 query 7,680개가 아니다.
- 10,000회 bootstrap, analysis seed `20261008`. 모든 primary contrast에 같은 재표본 seed index를 사용한다.
- 95% CI와 세 primary contrast의 nominal Bonferroni 98.33% CI를 함께 보고한다. 작은 n에서 percentile bootstrap의 정확한 유한표본 coverage를 보장한다고 쓰지 않는다.
- 출처별·학습 시점별 결과를 함께 공개한다. 일부 시점이나 metric만 골라 primary로 바꾸지 않는다.
- reference가 확률적이면 rollout 추정 오차도 기록한다. 그것이 결론을 지배하면 불확실성을 보고하며 query 반복을 독립 training seed로 대체하지 않는다.

## 9. 구현 검증과 산출물

실행 전 검증 항목은 다음과 같다.

- snapshot 복원, 시간 제한 해제 전후 공통 구간 parity, 희소 reward 재계산 일치
- original target 계산과 진단 target tensor parity
- actor/target 고정 및 optimizer clone의 독립성
- lambda=0과 HER loss/gradient/update parity; 모든 mask가 같을 때 mean-one weight 확인
- Random의 k개 선택, TD Top-k의 동일 residual 사용, k=0/k=m/동률/중복 batch 위치 처리
- 평가 자료가 replay에 들어가지 않음, 진단 RNG가 원 학습 RNG를 소비하지 않음
- Adam state·gradient clipping·실제 update norm 기록

제안 산출물: `checkpoint_inventory.csv`, `protocol.json`, `source_manifest.json`, `restore_validation.json`, `snapshot_manifest.json`, `rollout_references.csv`, `fork_results.csv`, `seed_summary.csv`, `report.json`, `REPORT.md`.

원 checkpoint와 원 replay는 읽기 전용 입력으로 취급한다. 구현·결과는 새 독립 실험 디렉터리에 저장한다. 이 문서 위치에 대형 checkpoint나 simulator snapshot을 넣지 않는다.

## 10. 예상 규모와 해석 기준

두 출처 × 10 seeds × 3시점이면 60 checkpoint다. 최대 128 queries씩 총 7,680 queries이며 결정적 환경의 reference rollout 상한은 7,864,320 환경 steps다. 평가 상태 수집은 별도이며 상한은 32×200×60=384,000 steps다. 실제 termination이 있으면 유효 query 수와 사용 steps를 따로 보고한다. 네 arm은 같은 reference를 공유하므로 rollout 비용을 네 배로 계산하지 않는다.

1-step 주 진단은 `60×16×4=3,840` critic updates, 20-step 부 진단은 `60×20×4=4,800` updates다. 확률적 reference의 8회 반복, HER goal 부 분석, 새로운 source training 비용은 별도다. pilot에서 초/query와 디스크 사용량을 측정해 실제 예산을 계산한다.

| 관측 결과 | 허용되는 해석 |
|---|---|
| SABER 자체의 Q 오차가 줄고 대조군 대비 개선량도 명확함 | 이 고정 조건에서 support 배정이 정책 행동의 가치 예측에 유용한 근거 |
| SABER가 상대적으로 덜 나쁘지만 자체 Q 오차는 증가 | 상대적 손상 완화; Q 정확도가 개선됐다고 쓰지 않음 |
| training residual만 감소 | loss가 의도대로 작동했다는 점검; 실제 가치 정확도 근거는 아님 |
| SABER와 엄격한 대조군의 차이가 불확실 | exact membership의 추가 가치가 아직 불확실 |
| SABER 출처나 특정 시점에서만 효과 | 그 분포와 학습 단계에 한정한 설명 |

이 실험이 성공해도 정확한 actor gradient, 장기 성공률 증가, support의 보편적 최적성을 증명한 것은 아니다. 문서의 첫 실행 단계는 모델 학습 재개가 아니라 **입력 자료 확보 → snapshot/target 감사 → 소규모 복원 pilot**이다.

# 실험 3. 환경별 수작업을 줄인 공통 context로도 SABER가 작동하는가?

- 작성일: 2026-10-08
- 상태: 설계 초안 v0.1. 아래 공통 context는 제안한 신규 변형이며 아직 구현·학습·예약하지 않았다.
- 역할 분담: 공동저자의 DrawerOpen / DrawerClose SAC는 backbone 확장을 담당한다. 본 설계는 DDPG를 유지해 context 설계 의존성을 평가한다.
- 연결 문서: [정책 행동 Q 진단](01_policy_query_value_diagnostic.md), [설정 민감도](02_support_hyperparameter_sensitivity.md)

## 1. 쉽게 설명하면

기존 SABER는 물체를 잡는 단계, 서랍을 여는 단계, 높이, gripper 상태 등의 정보를 이용해 비슷한 상황을 묶는다. 리뷰어는 이 규칙을 새 환경마다 다시 만들어야 하는지 궁금해했다.

이 실험에서는 **손과 물체의 거리, 물체와 목표의 거리, 남은 시간 비율**만으로 상황을 묶는 단순한 변형을 만든다. 한 과제에서 규칙을 확정한 뒤 다른 과제에서는 그대로 사용한다.

| 항목 | 설계 |
|---|---|
| 리뷰의 질문 | task-specific feature와 수동 threshold에 얼마나 의존하는가? |
| 비교 | HER / 기존 SABER / 공통 context SABER 변형 |
| 개발 과제 | FetchPickAndPlace–DDPG |
| 주 전이 과제 | DrawerOpen–DDPG |
| 후순위 확장 | DrawerClose–DDPG; 기본 예산에 포함하지 않음 |
| 바꾸는 것 | support 조회에 쓰는 context feature·경계·그로부터 만드는 index key |
| 유지하는 것 | policy-action membership, action bin, critic 가중식, uniform replay, HER, target, actor update, UTD |
| 핵심 절차 | 전이 결과를 보기 전에 공통 규칙을 고정하고 이후 수정하지 않음 |

새 환경에서는 정책을 처음부터 다시 학습한다. 여기서 옮기는 것은 **context 설계 규칙**이며, 학습한 정책의 zero-shot generalization을 시험하는 것이 아니다.

## 2. 현재 근거와 비교의 범위

사용자가 제공한 2026-10-06 보고서에는 공통 context 전이의 새로운 실행 결과가 없다. 추가 40회 학습과 조회 진단은 DrawerOpen–DDPG에 집중돼 있다. SAC 두 과제를 추가해도 context 수작업 부담은 별도로 확인해야 한다.

현재 로컬 HEAD `7becc22`와 사용자 변경분에서 읽은 context는 다음과 같다. 리뷰 이후 고정 코드가 현재 작업공간에 없으므로 실행 전에 원 source와 대조한다.

| 로컬 구현 | 기존 context 정보 |
|---|---|
| [task_fetch_pick_and_place.py](../../metaworld_Drawer_Open_DDPG/code/src/metacog_lite_v0/task_fetch_pick_and_place.py) | phase, 손–물체 거리, 물체–목표 거리, 물체 높이, gripper gap, horizon |
| [task_drawer_open.py](../../metaworld_Drawer_Open_DDPG/code/src/metacog_lite_v0/task_drawer_open.py) | phase, 손–손잡이 거리, 손잡이–목표 거리, 손잡이 y 위치, gripper 상태, horizon |

두 구현은 feature 위치, phase 정의, bin 경계가 다르다. 새로운 변형은 이 task별 phase·높이·축 위치·gripper 구분을 제거한다. 기존 `context_count_only` 실험은 정책 행동 membership을 제거하는 대조군이므로 본 변형과 다르다. 이름이나 결과를 혼용하지 않는다.

## 3. 비교군

| arm ID | 방법 | context | policy-action membership |
|---|---|---|---|
| `her` | 기본 HER | 학습 가중치에 사용하지 않음 | 사용하지 않음 |
| `saber_original` | 기존 SABER | task별 기존 규칙을 고정해 사용 | 유지 |
| `saber_common_context_v1` | 공통 context SABER | 아래의 동일한 세 feature와 bin 규칙 | 유지 |

공통 context arm은 별도 SABER 변형으로 표기한다. 실험 1의 Random/TD 배정 대조군처럼 high-weight 개수를 맞추는 실험은 아니다. Context 변경으로 local population, membership, k가 달라지는 것 자체가 평가 대상이다.

기존 SABER의 task별 설정은 논문/고정 protocol의 값을 사용한다. 새 공통 변형을 본 뒤 기존 SABER만 다시 tuning하지 않는다. Task 안에서는 세 arm의 architecture·HER·환경 상호작용·학습 budget을 동일하게 맞춘다.

## 4. 제안하는 공통 context v1

### 4.1 사용 정보와 남는 수작업

| 기호 | 의미 | 확보 방법 |
|---|---|---|
| p_ee | end-effector 위치 | 기존 관측에서 추출하는 명시적 adapter |
| ag | achieved goal | 해당 HER 환경의 기존 achieved_goal |
| g | 해당 query의 desired goal | 원 목표 또는 HER로 relabel한 goal |
| epsilon_g | 희소 reward의 성공 거리 기준 | 실제 학습 reward 함수가 쓰는 값 |
| h / H | 남은 시간 / episode horizon | 기존 replay h와 task별 학습 horizon |

이 설계도 end-effector 관측 매핑과 3차원 achieved/desired goal이라는 지식을 사용한다. 따라서 domain-knowledge-free라고 부르지 않는다. 비교 범위는 이러한 표현을 공유하는 조작 과제다. 이미지 관측, navigation, 다른 action 차원까지 보장하지 않는다.

Fetch와 Drawer에서 좌표가 같은 단위인지, ag와 g가 같은 물리 좌표계인지 확인한다. epsilon_g가 양수이며 실제 reward 기준과 일치해야 한다. Goal wrapper나 reward threshold 자체는 이 실험에서 바꾸지 않는다.

### 4.2 feature와 discretization

다음 세 scalar를 사용한다.

`u = ||p_ee - ag||_2 / epsilon_g`

`v = ||ag - g||_2 / epsilon_g`

`eta = h / H`

| feature | bin 경계 | 해석 |
|---|---|---|
| u | 1, 2, 4, 8 | 성공 거리 단위로 정규화한 손–물체 거리 |
| v | 1, 2, 4, 8 | 같은 단위의 물체–목표 거리 |
| eta | 0.25, 0.50, 0.75 | 남은 episode 시간 비율 |

Bin 함수는 `bin(x; edges) = sum(x > edge)`로 고정한다. 경계와 정확히 같은 값은 낮은 쪽 bin에 속한다. u와 v는 각 5개 bin, eta는 4개 bin이다. Non-finite feature, epsilon_g≤0, h가 [0,H] 밖인 자료는 오류로 처리하며 임의 보정하지 않는다.

환경의 성공 거리 epsilon_g와 horizon H는 task에 따라 다를 수 있다. **옮기는 것은 위의 정규화 공식과 무차원 경계**다. 이를 모든 환경에서 물리적 거리나 episode 길이가 동일하다는 뜻으로 설명하지 않는다.

### 4.3 retrieval key와 full-context key

- Coarse retrieval key: `(bin(u), bin(eta))`.
- Full-context key: `(bin(u), bin(v), bin(eta))`.
- Dictionary tuple key 또는 충돌이 없는 mixed-radix encoding을 사용한다.
- Full key는 coarse key 성분을 반드시 포함한다.
- Coarse key는 goal에 의존하지 않으므로 replay index에 저장할 수 있다.
- Candidate의 full key를 계산할 때는 **candidate에 원래 저장된 goal 대신 현재 query의 goal g**를 사용한다. Candidate observation과 candidate의 저장 h는 그대로 사용한다.

기존 phase, object height, handle-y, gripper feature를 coarse retrieval에 몰래 남겨두지 않는다. 제거한 정보가 full key에서는 사라졌지만 조회 단계에서 계속 쓰이는 구성을 공통 context라고 부르지 않는다.

### 4.4 support와 weight는 유지한다

기본값은 `K=5`, `N_scan=512`, `N_local_min=4`, `M_Q_min=4`, `M_B=16`, `lambda_support=1`이다. Task 간 reward·action 정의가 다른 부분은 기존 adapter와 action normalization을 그대로 사용한다.

1. 같은 coarse key의 유효 후보 중 기존 유지 순서의 Tail-512를 선택한다.
2. 선택한 후보에 query goal을 적용해 full-context matching을 수행한다.
3. 기존 population 조건과 현재 정책 행동의 action-bin membership으로 support를 판정한다.
4. `w_raw = 1 + support`, `Omega = w_raw / mean(w_raw)`로 critic loss만 가중한다.

항상 고정 개수의 이웃을 반환하는 kNN으로 바꾸지 않는다. 일치하는 이웃이 적으면 population 조건이 실패할 수 있어야 한다. Tail 유지 순서에 시간적 최신성 의미를 부여하지 않는다.

이 변형은 feature 수와 bucket 크기를 바꾸므로 검사 후보와 support 활성률이 달라질 수 있다. 같은 N_scan이 같은 계산 시간이나 같은 high-weight 개수를 보장하지 않는다.

## 5. 개발과 전이의 분리

### 5.1 개발: FetchPickAndPlace

제안한 v1을 먼저 구현하고 source 과제에서 adapter·bin·index·weight 계산을 검증한다. 제안된 세 feature와 경계를 이번 버전의 출발 규칙으로 고정하며, 대규모 hyperparameter 탐색을 기본 예산에 넣지 않는다.

개발 실행안은 세 arm × seeds 101, 102, 103 × 1M steps다. 이는 구현 및 source 동작을 확인하는 탐색 단계이며, n=3의 성능 우월성을 확정하려는 단계가 아니다. Development 결과도 모두 보존한다.

개발 결과를 바탕으로 feature나 경계를 바꾸면 `v2` 등 새로운 버전과 변경 이유를 남긴다. 그런 변경과 추가 개발 budget은 전이 결과를 보기 전에 끝내야 한다. 기존 v1 실행을 최종 변형의 결과에 섞지 않는다.

### 5.2 전이 직전 고정

다음 내용을 담은 `transfer_freeze.json`과 source manifest를 저장한다.

- 공통 feature 공식, bin 경계, coarse/full key 구현과 hash
- action normalization, action bins, population threshold, N_scan, lambda
- 과제별 observation adapter, achieved/desired goal 의미, epsilon_g, horizon
- 학습 protocol, 평가 grid, seed 목록, primary contrast, 실패 처리
- source 개발에 사용한 모든 설정 후보와 실행 ID
- 전이 과제 목록과 포함/제외를 결정한 시점

전이 과제에서는 adapter shape·단위·reward 일치 같은 correctness 검사를 할 수 있다. 목표 과제의 support 활성률이나 학습 성능이 마음에 들지 않는다는 이유로 threshold나 feature를 바꾸지 않는다.

### 5.3 주 전이: DrawerOpen

제안된 기본 비교는 **세 arm × seeds 1–10 × 2M steps**다. DDPG를 유지해 backbone 변경과 context 변경을 동시에 섞지 않는다.

이 task의 과거 성능과 기존 seed 결과는 이미 알려져 있다. 따라서 완전히 처음 보는 benchmark에 대한 blind 평가라고 부르지 않는다. 본 실험의 분리는 **공통 변형의 target 결과를 보고 설계 규칙을 다시 고르지 않는 것**이다. 기존 baseline을 재사용한다면 그 사실과 사후 설계라는 점을 공개한다.

Transfer에서 성능이 낮게 나오면 그대로 보고한다. 이를 보고 수정한 context는 별도 개발 버전이며, 같은 target 결과를 미관측 확인 결과로 재사용하지 않는다.

### 5.4 후순위 DrawerClose

DrawerClose–DDPG를 추가하면 MetaWorld 내 다른 목표 동작에서 같은 규칙을 확인할 수 있다. 기본 실행 예산에는 넣지 않는다. 확장하려면 포함 여부·seed·budget을 별도 protocol에 고정한다. DrawerOpen의 불리한 결과를 숨기고 DrawerClose만 선택해 보고하지 않는다.

공동저자의 DrawerClose SAC 결과는 별도 backbone 확장 자료다. 그 실행에 본 공통 context 변형을 실제로 적용하지 않았다면 공통 context의 전이 결과로 합치지 않는다.

## 6. 공통 학습 설정과 기준군 재사용

DrawerOpen에서는 보고서의 DDPG–HER 조건을 사용한다: UTD=1, batch=256, replay=300k, gamma=0.98, tau=0.005, actor/critic LR=3e-4, hidden=256, warmup=5k, 초기 random actions=10k, exploration sigma=0.2, horizon=200, nominal HER=0.8, success 조기 종료 없음, timeout bootstrap.

FetchPickAndPlace의 정확한 env ID·설치 버전·horizon·reward·HER 조건은 기존 고정 protocol을 확보해 별도 config로 확정한다. Drawer의 horizon=200을 그대로 복사하지 않는다. 같은 task 안의 세 arm은 context 개입 외 모든 학습 조건이 같아야 한다.

평가는 각 task에서 매 10k steps, deterministic policy, 20 episodes를 제안한다. DrawerOpen의 reset seed는 보고서와 같은 `10000 + training_seed + 10 * checkpoint_step + episode_index`다. Source task도 세 arm의 평가 규칙을 통일하고 manifest에 기록한다.

HER 및 기존 SABER 재사용에는 [실험 2의 재사용 감사](02_support_hyperparameter_sensitivity.md)에 준하는 config·code·환경·평가·RNG 검증이 필요하다. 보고서의 SABER 10회를 그대로 새 학습 10회로 다시 세지 않는다. 새 공통 context 학습은 fresh run이다.

기존 데이터가 적합하지 않으면 새 기준군을 구성한다. 이전 seed의 성공률이 높거나 낮다는 이유로 재사용 여부를 정하지 않는다. 같은 training seed가 서로 다른 arm의 replay trajectory를 동일하게 만든다는 주장은 하지 않는다.

## 7. 구현 검증

### 7.1 adapter와 context

- 두 task의 p_ee·ag·g 위치가 의도한 물체/손잡이 좌표와 일치하는지 확인한다.
- epsilon_g가 실제 학습 reward의 성공 조건과 같은지 확인한다. Native 환경 success와 wrapper goal-success가 다르면 둘을 구분한다.
- 거리 bin 경계의 직전·동일·직후 값과 horizon 0/H의 처리를 검증한다.
- 전체 좌표를 같은 벡터만큼 이동해도 거리 기반 key가 유지되는지 확인한다.
- 후보의 저장 goal을 바꾸어도 동일 query goal로 평가한 full key가 바뀌지 않아야 한다.
- Full-context match는 반드시 coarse match를 포함해야 한다.
- Coarse index 갱신, replay slot 덮어쓰기, generation validation, candidate 중복 제거를 검증한다.
- 공통 arm의 support 결정에서 legacy phase/height/gripper 경로가 호출되지 않는지 확인한다.

### 7.2 support와 학습

- 느린 reference 구현과 최적화 구현의 candidate 집합, E, M, weight parity를 대조한다.
- 빈 bucket, N_local 경계, 512보다 작은/큰 bucket, query action-bin 경계, k=0과 k=B를 검증한다.
- 기존 SABER arm은 원 고정 구현과 짧은 trajectory parity를 확인한다.
- 공통 context에서 lambda=0이면 HER와 loss/gradient/update parity가 성립해야 한다.
- Replay sampling·HER·actor update·target·UTD가 context branch에 따라 바뀌지 않는지 확인한다.
- 진단 RNG는 학습 RNG와 분리한다. 환경별 추가 계측도 동일한 규칙으로 적용한다.

Training 실행의 non-finite 값은 실패로 기록하고, 유한하지만 성공률이 0인 run은 유효 결과에 남긴다. 원 결과나 checkpoint에 덮어쓰지 않으며 실패 seed를 성과에 따라 재실행하지 않는다.

## 8. 성능, 설계 부담, 조회 신호를 함께 평가한다

### 8.1 주 성능 평가

DrawerOpen의 primary 지표는 10k–2M의 200개 checkpoint AUC, 분모 1,990,000이다. Secondary는 FQ-AUC 1.5M–2M, Final, Hit90 도달 수와 조건부 중앙값, zero-evaluation 수다.

고정 primary contrast는 다음 두 개다.

1. `AUC_common - AUC_HER`: 단순한 공통 규칙에서도 재가중의 이득이 있는가?
2. `AUC_common - AUC_original`: 수작업을 줄였을 때 성능이 얼마나 달라지는가?

Source 1M 개발 실행은 10k–1M의 100개 checkpoint, 분모 990,000으로 별도 집계한다. Source와 target의 다른 구간을 합쳐 하나의 seed-level AUC로 만들지 않는다.

Target에서는 training seed를 단위로 10,000회 paired bootstrap, analysis seed `20261010`을 사용한다. 두 contrast가 같은 재표본 index를 공유한다. 평균·SD·seed별 차이·95% CI와 nominal Bonferroni 97.5% CI를 보고한다. 재사용된 기본 SABER가 다른 실험에도 등장하면 독립 반복으로 합산하지 않는다.

Common−original의 CI가 0을 포함한다고 동등성이나 성능 보존을 입증한 것은 아니다. 허용 가능한 AUC 하락폭을 이용한 비열등성 주장을 추가하려면, 그 폭의 과학적 근거와 충분한 정밀도를 결과 확인 전에 별도 고정해야 한다.

### 8.2 수작업 부담 표

| 기록 항목 | 기존 SABER | 공통 context v1 |
|---|---|---|
| Task별 phase 정의 | 각 task 코드에서 추출해 기록 | 없음 |
| 높이·축 위치·gripper의 별도 구간 | 각 task별 목록 기록 | 없음 |
| 거리 bin | 각 task별 기존 경계 | 두 거리 모두 [1,2,4,8] |
| horizon bin | task별 기존 경계 | h/H의 [0.25,0.5,0.75] |
| 남는 observation mapping | 기존 task adapter | p_ee와 기존 achieved/desired goal mapping |
| 환경별 입력 상수 | 기존 설정 전체 | reward tolerance와 horizon을 기존 task 정의에서 가져옴 |
| Target 성능을 보고 수정한 항목 | 원 baseline의 알려진 개발 이력 명시 | 본 전이 평가 중에는 없음 |

단순히 코드 줄 수만 줄었다고 engineering 부담이 줄었다고 판단하지 않는다. Target에 적용할 때 새로 정한 feature, threshold, 관측 mapping과 개발 실행 수를 공개한다.

### 8.3 support와 근사 조회 진단

Eligibility, support-positive 비율, 배정 포화, 실제 weight 분산, coarse bucket 크기, scan 수를 학습 구간별로 보고한다. 단순한 context 때문에 대부분의 query가 positive가 되면, 학습 가중치가 거의 uniform해지는지 함께 확인한다.

소규모 고정 checkpoint에서는 각 context 정의 안에서 Tail-512와 Full을 비교한다. Full은 **그 context 정의 안의 기준**이며 연속 공간의 진짜 support가 아니다. 서로 다른 context의 positive 비율을 어느 쪽이 더 정확한지의 순위로 사용하지 않는다.

기존 Tail-vs-Random retrieval 연구와 본 context 변경을 같은 개입으로 섞지 않는다. 성능 차이에는 neighborhood 정의와 그에 따른 후보/가중치 변화가 함께 포함된다.

Runtime은 부 지표로 기록한다. 동일 hardware·부하·진단량을 맞춘 측정이 없으면 실제 로그 시간만으로 알고리즘 효율 순위를 정하지 않는다.

## 9. 예산과 산출물

| 단계 | 제안 규모 | 환경 steps |
|---|---|---:|
| 개발 | FPP, 3방법 × 3 seeds × 1M | 9M |
| 주 전이 비교 전체 | DrawerOpen, 3방법 × 10 seeds × 2M | 60M |
| 전이에서 HER와 기존 SABER 모두 재사용 가능 | 새 공통 context 10회만 학습 | 신규 20M |
| 전이에서 기존 SABER만 재사용 가능 | 새 HER 10회 + 공통 context 10회 | 신규 40M |
| 전이 기준군 재사용 불가 | 3방법 모두 새 실행 | 신규 60M |

따라서 기본 계획의 신규 학습량은 개발을 포함해 29M–69M steps이며, correctness smoke·정적 조회·변경 버전 재개발은 별도다. 실험 2와 동일한 기준군을 공유할 수 있으면 중복 학습 예산을 제외하고 원 run ID를 함께 기록한다. 두 문서의 nominal run 수를 그대로 더해 고유 학습 횟수로 세지 않는다.

DrawerClose 확장은 같은 3방법×10 seeds×2M이면 비교 전체 60M steps이며 재사용 가능분을 별도로 뺀다. 기본 계획에는 포함하지 않는다. 모든 규모는 설계 제안이며 현재 실행 중인 작업 수에 더하지 않는다.

제안 산출물: `context_spec.json`, `adapter_spec.json`, `development_runs.csv`, `transfer_freeze.json`, `reuse_catalog.json`, `source_manifest.json`, `validation_report.json`, `seed_metrics.csv`, `support_windows.csv`, `engineering_inventory.csv`, `report.json`, `REPORT.md`.

## 10. 결과에 따라 쓸 수 있는 주장

| 결과 | 해석 |
|---|---|
| 공통 규칙이 HER보다 개선되고 기존 SABER와의 성능 차이도 작고 정밀함 | 시험한 task 범위에서 더 적은 수작업 설계로 유용한 loss allocation을 유지한 근거 |
| HER보다 개선되지만 기존 SABER보다 성능 하락 | 적용 부담과 성능 사이의 trade-off |
| 공통 규칙에서 효과가 사라짐 | 기존 feature 또는 구분 해상도의 중요성을 보여주는 한계 |
| Source에서는 좋고 target에서는 나쁨 | 고정 설계 규칙의 전이성이 충분하지 않음 |
| Target마다 다시 조정해야 개선됨 | task별 tuning 결과로 보고; 수정 없는 전이 성공으로 부르지 않음 |

첫 구현 순서는 **공통 adapter와 세 feature 작성 → reference key/support 검증 → source smoke·개발 → 규칙 동결 → target 비교**다. 단순화된 방법의 결과가 나온 뒤 기존 SABER의 결과와 섞어 하나의 알고리즘처럼 제시하지 않는다.

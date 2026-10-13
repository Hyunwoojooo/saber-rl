# 실험 2. Support 설정을 바꿔도 SABER의 효과가 유지되는가?

- 작성일: 2026-10-08
- 상태: 설계 초안 v0.1. 실행·예약 없음. 제안한 전체 조합을 자동으로 학습시키지 않는다.
- 대상: DrawerOpen–DDPG의 대표 조건. 공동저자가 담당하는 Drawer SAC 비교와 별도 관리한다.
- 연결 문서: [정책 행동 Q 진단](01_policy_query_value_diagnostic.md), [공통 context 전이](03_common_context_transfer.md)

## 1. 쉽게 설명하면

현재 SABER는 주변 데이터가 일정 개수 이상이고 정책 행동의 support가 확인되면 가중치를 높인다. 이 실험은 **주변 데이터의 최소 개수와 가중치 강도를 조금 바꾸었을 때 결과가 얼마나 달라지는지** 확인한다.

목적은 잘 나오는 설정을 찾아 기존 결과를 교체하는 것이 아니다. 기본 설정이 어느 범위에서 작동하고, 어느 조건에서 민감한지 보여주는 것이다.

| 항목 | 설계 |
|---|---|
| 리뷰의 질문 | 여러 수동 설정 중 특정 값에만 효과가 의존하는가? |
| 우선 바꿀 것 | 주변 데이터 최소 개수, support 가중치 강도 |
| 고정할 것 | 환경·DDPG·HER·replay·target·actor update·UTD·기존 context·Tail 후보 선택 |
| 비교 방식 | 한 번에 설정 하나씩 변경; 기본 조건은 공유 |
| 주 지표 | 같은 10k–2M 평가 grid의 성공률 AUC |
| 함께 볼 것 | eligibility, support 활성률, 배정 포화, weight 분산, 실패 seed |
| 첫 작업 | requested config와 실제 적용 config가 일치하는지 확인 |

## 2. 자료와 로컬 구현에서 확인한 주의점

사용자 제공 2026-10-06 보고서의 기준값은 `N_local=4`, `M_Q_min=4`, `M_B=16`, `N_scan=512`, `K=5`, `lambda_support=1`이다. 완료된 SABER 10회는 재사용 후보이며, 이 문서에서 새 실행으로 세지 않는다.

로컬 HEAD `7becc22`와 사용자 변경분을 읽은 결과, 다음 두 가지를 확인했다. 리뷰 이후 원본 코드와 결과 폴더는 현재 작업공간에 없으므로 원 source에서도 다시 감사해야 한다.

### 2.1 N_local을 낮춰도 조건이 달라지지 않을 수 있다

[weights.py](../../metaworld_Drawer_Open_DDPG/code/src/metacog_lite_v0/weights.py)의 `_context_count_eligible_from_scan`은 local matching 개수 n에 대해 다음 조건을 사용한다.

`eligible = (n >= N_local_min) and (min(n, M_B) >= M_Q_min)`

`M_B >= M_Q_min`일 때 유효한 최소 개수는 `max(N_local_min, M_Q_min)`이다. 따라서 M_Q_min=4를 그대로 두고 N_local을 1과 4로 바꾸면 population 판정이 같을 수 있다. 이 조합으로 성능 차이가 없었다고 해서 넓은 범위의 강건성을 확인했다고 쓰면 안 된다.

본 설계의 주 sweep은 **N_local_min = 4 / 8 / 16**, M_Q_min=4, M_B=16 고정이다. 4보다 낮은 실제 threshold를 시험하려면 M_Q_min도 함께 바뀌므로 별도의 공동 변경 실험으로 명명해야 한다.

### 2.2 요청한 8이 실제로 4가 될 수 있다

[config.py](../../metaworld_Drawer_Open_DDPG/code/src/metacog_lite_v0/config.py)의 `normalize()`는 현재 로컬 DrawerOpen 경로에서 `N_local_min == 8`이면 4로 바꾼다. M_Q_min에도 같은 형태의 기본값 변환이 있다.

따라서 CLI에 `--N_local_min 8`을 넣었다는 사실만으로 해당 조건이 실행됐다고 볼 수 없다. 독립 실험 구현에서 명시적으로 제공한 설정을 보존하고, 초기화가 모두 끝난 뒤 effective config를 assert해야 한다. 기존 기본값 동작을 유지하는 parity 검증과 명시적 8 보존 검증을 모두 수행한다. 이 문서를 만드는 작업에서는 학습 코드를 수정하지 않는다.

## 3. 기본 실험표

| arm ID | N_local_min | M_Q_min | M_B | lambda_support | 역할 |
|---|---:|---:|---:|---:|---|
| `n4_l1` | 4 | 4 | 16 | 1 | 기본 SABER |
| `n8_l1` | 8 | 4 | 16 | 1 | 더 많은 local evidence 요구 |
| `n16_l1` | 16 | 4 | 16 | 1 | population 조건 강화 |
| `n4_l0` | 4 | 4 | 16 | 0 | uniform loss / HER 기준 |
| `n4_l05` | 4 | 4 | 16 | 0.5 | 약한 추가 가중치 |
| `n4_l2` | 4 | 4 | 16 | 2 | 강한 추가 가중치 |

전체 factorial 조합이 아니라, 기본점에서 한 번에 하나씩 바꾸는 6조건이다. 이 설계만으로 N_local과 lambda 사이의 상호작용까지 설명하지 않는다.

가중치는 모든 SABER 조건에서 다음과 같다.

`w_raw_i = 1 + lambda_support * support_i`

`Omega_i = w_raw_i / mean(w_raw)`

두 집단이 존재할 때 큰/작은 가중치 비율은 각각 lambda=0.5에서 1.5:1, lambda=1에서 2:1, lambda=2에서 3:1이다. 평균 계수는 1이지만 gradient norm과 Adam update 크기까지 같다는 뜻은 아니다.

N_local 변경은 eligible 집합, support-positive 개수와 배정까지 바꿀 수 있다. lambda 변경도 online 정책과 replay를 통해 이후 support를 바꾼다. 서로 다른 학습 실행에서 같은 mask가 끝까지 유지된다고 해석하지 않는다.

## 4. 공통 학습 조건

다음 값은 보고서의 DrawerOpen–DDPG protocol에 맞춘 제안이다. 실행 시 source의 전체 effective config와 비교해 아래 표에 없는 항목도 고정한다.

| 항목 | 값 |
|---|---|
| 환경 | DrawerOpen-v3, MetaWorld reward-function v2, sparse reward |
| backbone / 학습량 | DDPG, run당 2,000,000 환경 steps |
| batch / replay capacity / UTD | 256 / 300,000 / 1 |
| horizon / warmup / 초기 random actions | 200 / 5,000 / 10,000 |
| gamma / tau | 0.98 / 0.005 |
| actor·critic learning rate / hidden size | 각각 0.0003 / 256 |
| exploration sigma | 0.2 |
| HER | nominal 0.8, strict future stored-transition의 next achieved goal |
| 종료 | success 조기 종료 없음, timeout bootstrap |
| 평가 | 매 10,000 steps, deterministic policy, 20 episodes |
| 평가 reset seed | `10000 + training_seed + 10 * checkpoint_step + episode_index` |
| 조회 / action bin | 기존 Tail-512 / K=5 |
| context | 기존 DrawerOpen context와 discretization을 그대로 사용 |
| loss 적용 범위 | critic-only, minibatch mean-one normalization |

Tail의 유지 순서를 시간적 최신성으로 설명하지 않는다. 진행 중이던 Random-512 retrieval 결과가 좋거나 나쁘다는 이유로 본 sweep의 retrieval 방식을 바꾸지 않는다.

## 5. 단계별 진행과 seed 예산

### 5.1 단계 A: 학습 없는 판정 점검

가용 checkpoint 2개에서 동일한 replay·정책·HER query를 사용해 4/8/16의 eligibility와 support mask를 계산한다. 정확히 같은 scanned 후보를 사용할 때 threshold를 높여 eligibility가 늘어나면 구현을 점검한다.

Synthetic local-count 0, 1, 3, 4, 7, 8, 15, 16, 17 사례와 action membership 유무를 조합해 판정을 확인한다. 이는 성능 결과가 아니라 설정이 실제로 다른 연산을 만들고 있는지 확인하는 작업이다.

### 5.2 단계 B: 구현 smoke

- 6조건 각각 6k steps, 기술 검증 seed 1001을 제안한다. smoke는 성능 평가에 합치지 않는다.
- 공통 학습 config, HER/target/actor 동작과 비유한 값 검사를 수행한다.
- `n4_l1`은 고정된 원 SABER 코드와 짧은 trajectory parity를 검증한다.
- `n4_l0`은 동일 초기화의 HER와 target·loss·gradient·parameter update 및 RNG 흐름을 대조한다.
- support 계산을 끈 HER와 켠 lambda=0이 수학적으로 같은 loss라는 것만으로 end-to-end parity를 가정하지 않는다. 진단 RNG와 replay RNG를 분리한다.
- 요청값 8이 실제로 8인지, normalize 이후 설정을 읽어 검증한다.

### 5.3 단계 C: 기본 민감도 실행안

**기본안은 6조건 × seeds 1–5 × 2M이다.** 총 30개 비교 run, 60M 환경 steps에 해당한다. 기존 결과를 보고 구성한 n=5 민감도 분석이라는 점을 공개한다. 독립적인 대규모 확증 연구나 강한 동등성 검정으로 부르지 않는다.

- 기본 SABER 5회가 재사용 가능하고 동일 조건 HER도 5회 있으면 신규는 20회 / 40M steps다.
- SABER만 재사용 가능하면 신규는 25회 / 50M steps다.
- 둘 다 재사용할 수 없으면 신규는 30회 / 60M steps다.

처음부터 n=10을 택하면 6조건 × seeds 1–10, 총 60개 비교 run / 120M steps다. 두 기준군을 모두 재사용하면 신규 40회 / 80M steps다. **5 또는 10의 규모는 새 조건의 본 결과를 보기 전에 manifest에 고정**한다. 5-seed 결과의 CI가 유리해질 때까지 seed를 하나씩 추가하지 않는다. 이후 확증이 필요하면 별도 protocol과 독립 seed cohort로 구분한다.

기존 1–5 또는 1–10의 baseline 결과를 이미 알고 있으므로 완전히 미관측인 비교라고 표현하지 않는다. 다른 seed 번호를 선택할 경우에도 기본점과 모든 변형을 같은 목록으로 맞추고 재사용 예산을 다시 계산한다.

## 6. 기존 결과 재사용 감사

보고서에는 새 HER-only 10-seed arm이 없다. 원 논문의 HER가 이번 조건에 맞는지는 별도 확인한다.

| 감사 대상 | 확인할 내용 |
|---|---|
| 코드 | 고정 source, 변경분, 환경 wrapper, action scale, critic loss, optimizer와 clipping |
| protocol | HER indexing, reward, terminal/timeout, UTD, warmup, exploration, network |
| 실행 | training seed, 2M 완료, config, fresh run 여부, infrastructure 중단 기록 |
| 평가 | 정확한 200개 checkpoint, reset seed 규칙, episode 수, 성공률 정의 |
| 진단 | 추가 logging이 replay·exploration·HER RNG에 영향을 주지 않음 |

재사용은 `reuse_catalog.json`에 원 run ID와 hash를 남긴다. 재사용된 run은 새 독립 반복으로 세지 않는다. parity가 입증되지 않은 데이터는 별도 historical 결과로 제시하며 새 arm과 조용히 합치지 않는다.

Lambda=0은 HER의 성능 기준으로 사용할 수 있지만, support query를 계속 수행하면 계산 비용까지 HER와 같지는 않다. 런타임 순위는 별도로 동일 계측량과 부하를 맞춘 경우에만 논한다.

## 7. 평가 지표와 통계

### 7.1 성능

| 지표 | 정의 |
|---|---|
| Primary AUC | 10k, 20k, ..., 2M의 200개 평가점에서 trapezoidal integration / 1,990,000 |
| FQ-AUC | 1.5M–2M 적분 / 500,000 |
| Final | 정확히 2M 평가 성공률 |
| Hit90 | 최초 성공률 0.9 이상 step; 도달 수와 조건부 중앙값을 함께 제시 |
| zero-evaluation | 200개 예정 평가점이 모두 0인 run의 수 |

Checkpoint 누락이나 중복을 조용히 교집합으로 처리하지 않는다. 수치가 유한한 zero-evaluation 실행은 유효한 결과로 포함한다. 학습 중 성공 경험 유무와 평가 성공률 0을 구분한다.

### 7.2 분석 방향

기본 비교 방향은 각 변형의 `AUC_variant - AUC_n4_l1`이다. default 자체의 유용성은 `AUC_n4_l1 - AUC_n4_l0`로 별도 표시하되, 같은 pair를 독립 증거로 두 번 세지 않는다.

- seed별 지표와 paired 차이를 모두 공개한다. 평균·표본 SD·paired CI를 제시한다.
- 10,000회 paired bootstrap, analysis seed `20261009`. 모든 변형에 같은 seed index 재표본을 사용한다.
- 기본 민감도 표의 95% CI는 기술적 불확실성 표시다. 다섯 variant 대 default 전체의 우열을 동시에 주장하려면 사전에 정한 nominal Bonferroni 99% CI도 함께 제시한다.
- n=5 또는 n=10의 작은 표본과 percentile bootstrap 한계를 명시한다.
- CI가 0을 포함한다는 이유만으로 설정들이 동등하거나 강건하다고 결론내리지 않는다. 동등성/비열등성을 주장하려면 의미 있는 효과 크기 기준과 적절한 정밀도를 별도로 설계한다.
- 최고 평균의 설정으로 원래 기본점을 교체하지 않는다. 새 기본값을 개발하려면 개발 결과와 독립 평가를 분리한다.

### 7.3 support와 최적화 진단

같은 1,000-update window에서 valid update 수를 이용해 집계한 뒤 seed를 동일 가중치로 평균한다. 0–500k, 500k–1M, 1M–2M 구간도 함께 제시한다.

- eligible sample 비율, support-positive 비율, k=0 비율
- `k=|E|>0`인 배정 포화와 `k=B`인 완전 uniform 상황을 구분
- informative update 비율 `0<k<|E|`
- 실제 weight 평균·분산, gradient clipping 빈도와 parameter-update 크기 표본
- coarse 후보 수, 실제 scan 수, early-exit 비율
- 훈련 성공 경험과 평가 성공률의 시간적 변화

Early exit로 얻은 count가 threshold 도달의 하한이면 정확한 전체 local population으로 기록하지 않는다. 정확한 count가 필요한 소규모 사후 진단은 같은 후보 집합 전체를 다시 검사한다.

## 8. 실패 처리와 구현 검증

- Non-finite loss·TD·gradient는 0으로 치환하지 않고 실행 오류로 기록한다.
- seed나 설정을 성능이 낮다는 이유로 제거하거나 다시 시작하지 않는다.
- 본학습은 fresh run을 기본으로 하고 checkpoint continuation으로 다른 trajectory를 이어 붙이지 않는다.
- Infrastructure 중단은 로그로 원인을 구분하고 attempt를 남긴다. 재실행이 필요하면 같은 seed를 처음부터 실행한 별도 attempt로 관리한다.
- 실제 실행 직전 `requested_config`와 `effective_config`를 비교하고, arm별로 허용된 한 항목 외의 학습 차이는 오류로 처리한다.
- NumPy reference와 Numba/캐시 경로의 판정 parity를 threshold별로 확인한다. 캐시가 이전 설정의 mask를 재사용하지 않도록 config별로 분리한다.

## 9. 후순위 확장과 범위

다음은 기본 6조건 예산에 포함하지 않는다.

| 확장 후보 | 구분해야 할 점 |
|---|---|
| N_scan 128 / 512 / 2,048 | 같은 Tail 규칙에서 후보 수를 바꾸는 실험. Tail-vs-Random-512와 다른 질문 |
| Context discretization을 성기게/촘촘하게 변경 | 어떤 경계만 변경하는지 고정. phase 정의나 feature까지 동시에 바꾸지 않음 |
| N_local과 lambda의 제한된 교차 실험 | 기본 one-factor 설계가 다루지 못한 상호작용 |
| 다른 task/backbone의 대표 조건 | DrawerOpen–DDPG에서 관측한 민감도의 외적 타당성 |

확장 조건은 필요한 질문과 총 예산을 별도 문서에 고정한다. 추가 SAC 성능표가 이 민감도 실험을 대신했다고 보지는 않는다.

## 10. 산출물과 해석

제안 산출물은 `protocol.json`, `arm_matrix.csv`, `reuse_catalog.json`, `source_manifest.json`, arm별 requested/effective config, `seed_metrics.csv`, `support_windows.csv`, `paired_differences.csv`, `report.json`, `REPORT.md`다.

그림은 N_local에 따른 AUC와 lambda에 따른 AUC를 별도 패널로 배치한다. seed 점과 불확실성을 표시하고, 같은 축 아래에 support 활성률 또는 weight 분산을 함께 배치하면 성능과 신호 변화의 관계를 읽기 쉽다.

| 결과 | 해석 |
|---|---|
| 여러 설정에서 평균 이득이 관측되고 추정도 비교적 정밀함 | 시험한 설정 범위에서 효과 유지의 근거 |
| 일부 설정에서 성능이 크게 악화 | 민감한 조건과 사용 범위를 명시 |
| N_local 변화에도 signal과 성능이 거의 같음 | population 조건이 해당 데이터에서 대부분 충족되는지 확인 |
| signal은 크게 달라지지만 성능 CI가 넓음 | 성능 민감도가 없다는 결론 대신 불확실성을 보고 |
| lambda=0과 HER가 다름 | 우선 config·loss scale·RNG·actor/target parity를 점검 |

실행 순서는 **고정 source 확보 → effective config 감사 → 학습 없는 판정 점검 → smoke → 고정한 seed 규모의 본 비교**다. 기존 실험 폴더와 결과를 덮어쓰지 않는다.

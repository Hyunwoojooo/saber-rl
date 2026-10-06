# SABER: AAAI 리뷰 이후 실험 현황 및 결과 보고서

- 작성일: 2026-10-06.
- 진행 상황 기준 시각: **2026-10-06 04:51:13 UTC / 13:51:13 KST**.
- 범위: 2026-09-25 AAAI 결정 이후 이 작업공간에서 수행한 추가 학습, 사후 진단, 검증 및 예약 작업.
- 자료: 실험별 고정 protocol, 실행 상태, 원시 `metrics.csv`, 진단 CSV, 분석 보고서, 검증 기록, source manifest.
- 이 문서는 해당 시점의 정적 기록이다. 진행 중인 실험의 이후 결과는 자동 반영되지 않는다.
- 이미 제출한 Full Paper와 Technical Supplement 전체를 이번에 다시 감사한 보고서는 아니다. 리뷰 연결은 전달받은 리뷰 내용에 근거한다.

## 1. 전체 현황

**추가 본학습은 40회 완료됐고, 현재 Random-512 retrieval 5회가 실행 중이며 5회가 대기 중이다.** 완료된 SABER 10회는 이후 PAL 비교와 Tail 기준군에 재사용하므로 중복 집계하지 않는다.

| 작업 | 새 학습 규모 | 완료 | 실행 중 | 대기 | 현재 결과 |
|---|---:|---:|---:|---:|---|
| SABER / Random assignment / TD-error Top-k | 3방법 x 10 seeds x 2M | 30 | 0 | 0 | 최종 지표와 paired CI 확보 |
| HER + PAL | 10 seeds x 2M | 10 | 0 | 0 | 최종 지표와 SABER 대비 paired CI 확보 |
| Tail / Random / Full 오프라인 조회 진단 | 기존 SABER 최종 checkpoint 10개 | 분석 완료 | 0 | 0 | 조회 누락률, 불일치율, 참조 구현 시간 확보 |
| 실패 및 support 포화 분석 | 기존 배정 실험 30회 | 분석 완료 | 0 | 0 | zero-evaluation 결과와 포화의 동반 관계 확보 |
| Tail-512 / Random-512 online 비교 | 새 Random-512 10 seeds x 2M | 0 | 5 | 5 | 중간 결과만 존재 |
| Online 조회 비교 종료 후 분석 | 위 실험 결과 및 양쪽 checkpoint | 0 | 0 | 예약됨 | 최종 CI 및 추가 정적 조회 진단 |

### 1.1 진행률 계산

| 기준 | 값 |
|---|---:|
| 완료된 고유 본학습 run | 40 / 50 = 80.0% |
| 완료된 run의 환경 step | 80,000,000 |
| 현재 Random-512 cohort의 누적 환경 step | 7,206,000 / 20,000,000 = 36.03% |
| 전체 승인된 추가 본학습의 누적 환경 step | 87,206,000 / 100,000,000 = 87.206% |
| 아직 남은 환경 step | 12,794,000 |

위 분모는 지금까지 실행이 승인된 세 학습 cohort에 한정한다. Smoke test와 오프라인 query 수는 환경 step 예산에 넣지 않았다. 앞으로 논의만 한 민감도, 전이, critic 분기 실험도 포함하지 않았다.

### 1.2 핵심 판단

1. 완료된 비교에서 SABER의 평균 AUC는 Random assignment, TD-error Top-k, PAL보다 높다.
2. 그러나 세 SABER-minus-control 비교의 95% CI가 모두 0을 포함한다. 현재 자료로 평균 우위를 확정적인 통계적 우월성으로 표현하면 안 된다.
3. 새로운 비교는 기존의 연속 TD-error weighting과 달리 가중치 형태와 선택 개수를 통제했고, PAL이라는 구체적인 선행방법도 추가했다.
4. 정적 조회 진단에서는 Random-512가 Tail-512보다 Full 기준의 support를 덜 놓쳤다. 이것만으로 학습 성능까지 좋아진다고 결론 낼 수는 없다.
5. 진행 중인 online 조회 비교의 seeds 1-5, 공통 1.4M 구간에서는 Tail의 평균 AUC가 높다. 아직 10-seed 최종 결과가 아니다.
6. 현재 새 실험의 적용 범위는 모두 **DrawerOpen + DDPG**다. FPP, DrawerClose, TD3, SAC 전반으로 확대해서 해석하지 않는다.

## 2. 리뷰 지적과 실험의 연결

| 리뷰에서 제기된 문제 | 대응 작업 | 현재까지 답한 부분 | 아직 답하지 못한 부분 |
|---|---|---|---|
| wpgC: 일반적인 loss reweighting 자체의 신규성이 약함 | HER + PAL 비교 | 구체적인 기존 loss-side 방법과 직접 비교 | 성능 비교만으로 신규성이 확보되지는 않음 |
| wpgC: support 판정 규칙의 정당성이 부족함 | 동일 가중치 분포의 Support / Random / TD 배정 비교 | 선택 위치의 추가 가치를 더 통제된 조건에서 평가 | CI가 넓고, policy-action 위치의 실제 Q 정확도는 측정하지 않음 |
| mK8N: 왜 tail 후보를 사용하는가 | 정적 조회 진단 및 online Random-512 비교 | 누락률과 순서 의존성을 정량화; 학습 영향은 검증 중 | 어느 환경에서도 Tail이 좋다는 결론은 불가능 |
| mK8N: 수작업 context와 여러 hyperparameter의 민감도 | 후속 민감도 및 공통 context 전이 구상 | 질문과 후보 실험을 구분해 정리 | 새로운 학습은 아직 수행하지 않음 |
| AI: 적은 seed, 변동성, mechanism CI의 불확실성 | 10-seed 추가 cohort, 실패 및 포화 분석 | 모든 seed를 포함한 지표와 실패 양상 확보 | 다른 task/backbone의 불확실성까지 해소하지는 않음 |
| AI: 동일 UTD와 동일 계산 비용은 다름 | 실행 시간 기록, 공정한 wall-clock 비교 구상 | 현재 실행의 측정 시간은 보존 | 계측량과 부하를 맞춘 알고리즘 비용 비교는 미완료 |
| 두 사람 리뷰어: Method 및 diagnostic 설명이 어려움 | 본문 재구성 계획 | 옮겨 넣을 수 있는 비교표와 분석 자료 확보 | 추가 실험만으로 가독성 문제가 해결되지는 않음 |

리뷰 점수는 wpgC 4, mK8N 6이었고, 두 리뷰어 모두 설명의 복잡성을 지적했다. 이 기록만으로 어느 리뷰 또는 AI 리뷰가 최종 reject를 결정했다고 판단하지 않는다. 후속 작업의 목적은 다음 투고를 위한 근거와 설명의 보강이다.

## 3. 실행 이력과 독립 관리

### 3.1 실제 실행 시간

폴더 이름은 코드 준비 날짜일 수 있으므로, 본학습 시작과 종료는 실행 상태 파일을 기준으로 구분했다.

| 사건 | UTC | KST |
|---|---|---|
| 배정 비교 첫 본학습 시작 | 09-25 15:04:05 | 09-26 00:04:05 |
| 배정 비교 마지막 본학습 종료 | 10-02 04:49:26 | 10-02 13:49:26 |
| PAL v2 본학습 시작 | 10-02 04:52:45 | 10-02 13:52:45 |
| PAL 마지막 본학습 종료 | 10-03 20:26:45 | 10-04 05:26:45 |
| PAL 최종 분석 및 자동화 완료 | 10-03 20:26:51 | 10-04 05:26:51 |
| 사후 분석 복구판 v2 실행 | 10-04 12:45:44 | 10-04 21:45:44 |
| 사후 분석 복구판 v2 완료 | 10-04 12:51:39 | 10-04 21:51:39 |
| Online retrieval 사전검증 완료 | 10-04 13:49:03 | 10-04 22:49:03 |
| Random-512 본학습 시작 | 10-04 13:50:48 | 10-04 22:50:48 |

따라서 이 보고서의 실험은 UTC 기준 9월 25일 이후의 작업이다. 최초 본학습 시작은 한국 시간으로 9월 26일이다.

### 3.2 독립 실험 폴더

| 폴더 | 역할 | 집계 규칙 |
|---|---|---|
| `2026-09-25_draweropen_ddpg_assignment` | 배정 비교의 고정 코드와 30회 결과 | 새 본학습 30회 |
| `2026-09-30_draweropen_ddpg_pal` | PAL 초기 준비판 | 별도 완료 cohort로 집계하지 않음 |
| `2026-10-01_draweropen_ddpg_pal_v2` | PAL 최종 실행 코드와 10회 결과 | 새 본학습 10회 |
| `2026-10-02_draweropen_retrieval_failure_diagnostics` | 최초 사후 분석판, loader 단계에서 중단 | 중단 기록 보존 |
| `2026-10-04_draweropen_retrieval_failure_diagnostics_v2` | 독립 복구판, 전체 분석 완료 | 추가 학습 없음 |
| `2026-10-04_draweropen_ddpg_retrieval_v1` | Online Random-512 및 Tail 기준 비교 | 새 본학습 10회 예정 |

과거 논문 코드와 결과에 덮어쓰지 않는다는 [독립 실험 관리 규칙](../experiments_post_review/AGENTS.md)을 적용했다. 본학습 코드, protocol, tests, launcher, 분석기와 manifest가 각 실험 폴더에 묶여 있다. 배정 실험은 과거 논문의 seeds 1-5를 합친 것이 아니라 **30회 모두 새 실행**이다.

## 4. 공통 학습 및 통계 규칙

### 4.1 학습 설정

| 항목 | 설정 |
|---|---|
| 환경 | DrawerOpen-v3, MetaWorld reward-function v2, sparse reward |
| 알고리즘 | DDPG + HER |
| Seeds / 실행당 학습량 | 1-10 / 2,000,000 환경 steps |
| UTD / batch / replay capacity | 1 / 256 / 300,000 |
| Horizon / warmup / 초기 random actions | 200 / 5,000 / 10,000 |
| Discount / target update | gamma=0.98 / tau=0.005 |
| Actor / critic learning rate | 각각 0.0003 |
| Hidden size / exploration sigma | 256 / 0.2 |
| HER | nominal probability 0.8, strict future stored-transition indexing |
| 종료 처리 | success로 조기 종료하지 않음, timeout에서 bootstrap |
| 평가 | 매 10,000 steps, deterministic action, 20 episodes |
| 평가 reset seed | `10000 + training_seed + 10 * checkpoint_step + episode_index` |
| Support 설정 | K=5, N_scan=512, N_local=4, M_Q_min=4, M_B=16 |
| SABER weight | lambda_support=1, critic-only, minibatch 평균 1 정규화 |

HER는 현재 transition보다 뒤의 stored transition을 선택하고 그 transition의 `next_achieved_goal`을 사용한다. 적용할 future transition이 없으면 원래 goal을 유지한다. 따라서 nominal 0.8은 모든 시점에서 동일한 effective relabeling 확률을 뜻하지 않는다.

위 표는 공통 핵심 설정의 요약이다. 전체 effective config는 각 attempt의 `requested_config.json`과 `run/config.json`에 보존된다. PAL에는 support 설정을 적용하지 않고 별도의 loss 설정을 사용한다.

### 4.2 지표

- **AUC:** 10k, 20k, ..., 2M의 정확한 200개 평가 checkpoint에서 trapezoidal integration. 분모는 `2,000,000 - 10,000 = 1,990,000`.
- **FQ-AUC:** 1.5M-2M 구간을 적분하고 500,000으로 나눈 값.
- **Final success:** 2M checkpoint의 평가 성공률.
- **Hit90:** 평가 성공률이 처음 0.9 이상이 된 checkpoint. 도달 seed 수와 도달한 seed에 한정한 중앙값을 함께 보고.
- **평균 및 SD:** seed별 지표의 평균과 표본 표준편차. Bootstrap 반복 수를 독립 실험 수로 세지 않음.
- **Paired difference:** 같은 seed 번호의 두 방법 지표 차이. 동일 seed가 서로 다른 방법의 replay trajectory까지 같음을 뜻하지는 않음.

최종 분석은 checkpoint 누락이나 중복을 완료 검증에서 잡는다. 누락된 지점을 교집합으로 조용히 제거해서 AUC 구간을 줄이지 않는다. 이 문서의 반올림 표시는 읽기 위한 것이며, 차이와 CI는 반올림 전 원자료에 근거한다.

### 4.3 실패 처리

본학습은 fresh run이고 production checkpoint continuation은 금지한다. Non-finite loss, TD error, gradient 등은 0으로 치환하지 않고 실행 실패로 기록한다. 반면 모든 값이 유한하지만 평가 성공률이 0인 실행은 **유효한 성능 결과**로 남긴다.

장비 문제에 의한 중단과 학습 실패는 구분한다. 근거가 있는 infrastructure 중단은 같은 seed를 새 attempt에서 처음부터 다시 실행하는 절차만 허용하며 자동 재시도나 trajectory 이어 붙이기는 하지 않는다. 이번에 완료된 40회는 모두 `attempt_01`, `complete`, `checkpoint_continuation=false`다.

## 5. 완료: Support / Random / TD-error Top-k 배정 비교

근거: [고정 protocol][A-PROTOCOL], [최종 분석][A-RESULT], [상세 집계][A-DETAIL].

### 5.1 검증 질문

> Context eligibility와 SABER에서 유도한 high-weight 개수를 유지할 때, 현재-policy support에 따라 가중치를 배정하는 것이 무작위 배정이나 TD-error 순위 배정보다 유리한가?

세 방법 모두 SABER가 계산한 eligibility와 high-weight 개수를 사용한다. 이 비교는 support 계산 전체를 없애도 되는지, 실제 Q 오차가 줄어드는지, actor gradient가 더 정확해지는지를 직접 검증하지 않는다.

### 5.2 정확히 무엇을 바꾸었는가

각 minibatch에서 E는 context-population 조건을 통과한 위치 집합, M은 기존 policy-action support-positive 위치 집합이다. `M subset E`, `m=|E|`, `k=|M|`이다.

| 방법 / 코드 이름 | 큰 가중치를 줄 위치 |
|---|---|
| SABER / `support` | M 자체 |
| Context-Eligible Random / `random` | E에서 정확히 k개를 균일 비복원 추출 |
| TD-error Top-k / `td_topk` | E 안에서 업데이트 전 절대 TD error가 큰 k개; 경계 동률만 균일 무작위 처리 |

여기서 TD-error Top-k는 **TD3 알고리즘이 아니다.** 세 방법의 backbone은 모두 DDPG다.

선택 mask를 z라 하면 `w_raw = 1 + z`, `Omega = w_raw / mean(w_raw)`다. 두 집단이 모두 존재할 때 큰 가중치와 작은 가중치의 비율은 2:1이고, minibatch 평균은 1이다. Actor weight는 1로 유지한다.

TD selector에는 실제 critic loss를 계산하는 동일 forward pass의 `abs(delta.detach())`를 전달한다. 다른 target을 다시 만들어 순위를 계산하지 않는다. 학습하는 residual은 replay action에서의 residual이고, current-policy action은 support 조회에만 사용한다.

각 online 실행은 자신의 replay, policy, HER minibatch로 E, M, k를 계산한다. **서로 다른 실행 사이에서 실제 k나 버퍼가 끝까지 같다는 실험은 아니다.** 배치 안에 같은 transition이 중복 추출되면 별도 batch 위치로 취급한다.

### 5.3 RNG 및 경계 조건

| 항목 | 규칙 |
|---|---|
| Replay / HER RNG | seed + 17 |
| Legacy weight RNG | 20000 + seed |
| Random assignment RNG | 30000 + seed |
| TD boundary-tie RNG | 40000 + seed |
| k=0 | 세 방법 모두 uniform weight; overlap/k는 NA |
| k=m>0 | 세 방법 모두 E를 선택; E가 batch 전체일 때만 전체 uniform |
| 0<k<m | 배정 규칙이 실제로 달라질 수 있는 informative update |

모든 arm에서 가상의 Random/TD mask도 계산해 진단한다. 전용 RNG를 사용하므로 logging이 replay/exploration stream을 소비하지 않는다. 원래 inherited live-target post-update residual 진단은 비활성화했다. 이 실험 로그로 fixed-target residual 또는 true Q accuracy를 주장하지 않는다.

Random 선택의 기대 교집합은 `k^2/m`, k>0에서 기대 overlap fraction은 `k/m`이다. 이 기대 교집합을 Jaccard 공식에 대입한 값을 Jaccard 기대값으로 사용하지 않는다.

### 5.4 최종 성능: 각 10 seeds, 모두 2M 완료

| 방법 | AUC mean +/- SD | FQ-AUC mean +/- SD | Final mean +/- SD | Hit90 | 도달 seed의 Hit90 중앙값 |
|---|---:|---:|---:|---:|---:|
| SABER | 0.652613 +/- 0.248402 | 0.892900 +/- 0.313825 | 0.890 +/- 0.313404 | 9/10 | 470k |
| Random assignment | 0.515678 +/- 0.359443 | 0.697700 +/- 0.481466 | 0.700 +/- 0.483046 | 7/10 | 550k |
| TD-error Top-k | 0.534623 +/- 0.380191 | 0.693600 +/- 0.478725 | 0.700 +/- 0.483046 | 7/10 | 400k |

TD의 조건부 Hit90 중앙값 400k가 작다는 것은, 도달한 7개 seed 안에서의 값이다. SABER의 9/10 도달률과 함께 읽어야 하며, 서로 다른 도달 집합의 중앙값만으로 전체 학습 속도를 순위화하지 않는다.

### 5.5 Paired AUC 결과

| 고정 비교 방향 | 평균 차이 | 95% CI | 97.5% CI | 양수 / 음수 / 동률 |
|---|---:|---|---|---:|
| SABER - Random | +0.136935 | [-0.127451, +0.384334] | [-0.162888, +0.414626] | 7 / 3 / 0 |
| SABER - TD-error Top-k | +0.117990 | [-0.054866, +0.324917] | [-0.071018, +0.360704] | 4 / 5 / 1 |

Bootstrap은 10,000회, analysis seed는 20260925다. 매 재표본에서 같은 seed 행 인덱스를 두 비교에 함께 적용했다. 95% 구간은 2.5%, 97.5% 분위수, 97.5% 구간은 1.25%, 98.75% 분위수다. 후자는 두 primary 비교에 대한 nominal Bonferroni 보정이지만, 10-seed percentile bootstrap의 정확한 유한표본 동시 coverage를 보장하지 않는다.

**해석:** 평균은 SABER가 높지만 우월성의 불확실성이 크다. 특히 TD 대비는 평균 차이가 양수여도 개별 seed에서는 4개 개선, 5개 악화, 1개 동률이다. CI가 0을 포함한다는 것은 우위를 확정하지 못했다는 뜻이지, 효과가 없거나 두 방법이 동등하다는 증명이 아니다.

### 5.6 배정 및 support 진단

각 seed 안에서는 유효 update 수로 가중 집계하고, 그다음 seed를 동일한 통계 단위로 평균했다.

| 방법 | Eligible sample 비율 | Support-positive 비율 | 포화 update 비율 | Informative update 비율 |
|---|---:|---:|---:|---:|
| SABER | 96.6158% | 42.9368% | 2.7987% | 97.2010% |
| Random assignment | 96.9304% | 50.4654% | 11.3761% | 88.6239% |
| TD-error Top-k | 96.7359% | 50.8080% | 13.4219% | 86.5781% |

SABER arm의 informative update에 한정한 overlap fraction은 다음과 같다.

| 비교 | 평균 overlap fraction |
|---|---:|
| Support와 TD top-k의 교집합 / k | 0.504095 |
| Support와 Random의 실제 교집합 / k | 0.427443 |
| Random 기준 기대값 k/m | 0.427438 |

Random의 관측 overlap과 기대값은 일치한다. Support와 TD 선택도 상당 부분 겹치지만 동일한 mask는 아니다. 이 수치 자체는 어느 선택이 학습에 더 유용한지 입증하지 않는다.

### 5.7 이 실험에서 얻은 의미

- 기존 연속 TD-error weighting과 달리 가중치 비율, 개수, normalization의 차이를 제거한 대조군을 확보했다.
- 동일한 평가 grid에서 10-seed 분포, 실패 사례, 선택 overlap을 함께 보고할 수 있다.
- SABER의 평균 AUC와 도달률이 높은 것은 관측된 사실이다. 하지만 정확한 support-positive 위치 배정이 필수적이라고 입증한 상태는 아니다.
- Random과 TD 대조군도 E와 k를 통해 support 정보를 사용한다. 결과가 비슷해도 support 계산 전체가 불필요하다고 결론 낼 수 없다.

## 6. 완료: HER + PAL 비교

근거: [PAL protocol][P-PROTOCOL], [최종 결과][P-RESULT], [구현 및 선행 코드 고지][P-NOTICES].

### 6.1 역할과 구현 범위

PAL은 uniform replay를 유지하면서 critic loss를 바꾸는 선행방법에 대한 비교다. 현재 TD top-k가 선택 신호를 통제하는 mechanism control이라면, PAL은 구체적인 기존 방법에 대한 성능 baseline이다. PAL을 PER처럼 높은 TD-error 경험을 더 자주 sampling하는 방법으로 설명하면 안 된다.

이번 구현은 저장된 원 저자 PAL 구현의 loss와 normalization을 단일 critic DDPG에 적용했다. 기본 설정은 `alpha=0.4`, `min_priority=1`이다. 이 설정에서 residual delta에 대한 piecewise loss는 다음과 같다.

```text
ell(delta) = 0.5 * delta^2                         if abs(delta) < 1
             abs(delta)^(1 + alpha) / (1 + alpha) otherwise

Z = stop_gradient(mean(max(abs(delta), 1)^alpha))
L_PAL = mean(ell(delta)) / Z
```

원 구현의 구간별 상수를 임의로 바꾸지 않았고, SABER식 cap=2나 mean-one sample weighting을 추가하지 않았다. Loss에 임의의 2배 보정이나 learning-rate 보상도 하지 않았다. 원 TD3 구현의 두 critic을 사용하는 normalization을 단일 critic 조건으로 축소한 것이다.

이는 **HER + PAL의 DDPG 적용 비교**이지, 원 PAL 논문의 TD3 benchmark를 그대로 재현한 실험은 아니다. 현재 minibatch normalization을 쓴다고 HER 환경에서 전체 replay에 대한 이론적 기대 gradient의 정확한 등가성까지 입증한 것도 아니다.

PAL에서는 support 조회와 가상 배정 mask 계산을 제거했다. Replay sampling, HER, target, actor update, UTD 및 나머지 핵심 학습 조건은 비교 protocol에 맞췄다. Alpha와 threshold를 결과에 맞춰 재탐색하지 않았다.

### 6.2 비교 cohort

- PAL seeds 1-10을 모두 처음부터 2M까지 새로 실행했다.
- SABER 기준군은 앞선 배정 실험의 `support` 10회를 재사용했다. 별도 SABER 10회를 더 실행한 것이 아니다.
- Reference의 설정, 평가 grid 및 실행 조건 재사용 감사가 통과했다.
- PAL 설계는 배정 실험의 중간 결과를 본 뒤 정했다. PAL 실행 전 설정은 고정했지만, 모든 선행 자료와 독립인 미관측 확인 실험이라고 부르지는 않는다.

### 6.3 최종 결과

| 방법 | AUC mean +/- SD | FQ-AUC mean +/- SD | Final mean +/- SD | Hit90 | 도달 seed의 Hit90 중앙값 |
|---|---:|---:|---:|---:|---:|
| SABER, 재사용 기준군 | 0.652613 +/- 0.248402 | 0.892900 +/- 0.313825 | 0.890 +/- 0.313404 | 9/10 | 470k |
| HER + PAL | 0.545352 +/- 0.302841 | 0.792600 +/- 0.417888 | 0.800 +/- 0.421637 | 8/10 | 620k |

| 고정 비교 방향 | 평균 차이 | Paired difference 중앙값 | 95% CI | 양수 / 음수 / 동률 |
|---|---:|---:|---|---:|
| SABER - PAL, AUC | +0.107261 | +0.068907 | [-0.153141, +0.357651] | 7 / 2 / 1 |

Bootstrap은 10,000회, analysis seed는 20260930이다. PAL 비교의 사전 고정 primary contrast는 하나이며, 앞선 두 배정 비교의 Bonferroni 구간을 그대로 가져오지 않는다.

SABER의 평균 AUC가 높고 7/10 pair에서 양수다. 그러나 구간이 넓고 0을 포함한다. 현재 표현은 **"이 DrawerOpen-DDPG cohort에서 더 높은 평균 AUC를 관측했으나 paired uncertainty는 남아 있다"**가 적절하다.

### 6.4 시간 기록과 비교의 한계

| 실행 | 평균 steps/s +/- SD | 실행당 평균 시간 +/- SD |
|---|---:|---:|
| SABER, 배정 진단 포함 | 21.604 +/- 1.241 | 25.795 +/- 1.546 h |
| Random assignment, 배정 진단 포함 | 21.775 +/- 1.442 | 25.621 +/- 1.811 h |
| TD-error Top-k, 배정 진단 포함 | 21.481 +/- 1.409 | 25.964 +/- 1.712 h |
| PAL, support/배정 진단 없음 | 28.096 +/- 0.317 | 19.775 +/- 0.223 h |

이 값은 실제 기록된 각 실행의 속도와 시간이다. **알고리즘 자체의 공정한 비용 비교 표는 아니다.** SABER 쪽에는 추가 배정 계측이 있고 실행 시기와 부하도 같지 않다. PAL 분석기 역시 `runtime_comparison_valid=false`로 표시한다. 동일한 계측 조건과 비교 가능한 부하의 성능-시간 곡선은 별도 평가가 필요하다.

## 7. 완료: 오프라인 조회 방식 진단

근거: [진단 protocol][D-PROTOCOL], [최종 보고서][D-FINAL], [조회 집계][D-RETRIEVAL].

### 7.1 질문과 고정 대상

> 같은 policy, replay, index, HER query에서 Tail-512, Random-512, Full 검색은 support 판정을 얼마나 다르게 만드는가?

대상은 앞서 학습한 SABER seeds 1-10의 **최종 2M checkpoint**다. Best checkpoint를 결과에 맞춰 고르지 않았다. Seed당 1,024개 query, 총 10,240개이며 Random은 query당 5회 반복했다. Tail 1회, Random 5회, Full 1회로 총 71,680개 방식별 query 측정이다.

| 방식 | 후보 선택 |
|---|---|
| Tail-512 | 같은 coarse-key bucket에서 유지된 index 순서의 끝부분 최대 512개 |
| Random-512 | 같은 bucket의 유효 후보 중 최대 512개를 균일 비복원 선택 |
| Full | 같은 bucket의 모든 유효 후보를 검사하고 full-context 조건 적용 |

Full은 **현재 이산 context 정의 안에서의 기준**이다. 연속 공간의 참된 support를 측정한 oracle은 아니다. Full-context가 retrieval key 성분을 포함하는 구조를 사용했고, seed당 16개 query, 총 160개를 전체 replay 직접 검색과 추가 대조했다.

Tail은 support-positive 샘플 중 마지막 512개를 고르는 것이 아니다. Support 판정 전에 coarse bucket 후보를 선택한다. 또한 유지된 index 순서를 시간적 최신성으로 해석하지 않는다.

### 7.2 최종 집계

아래 값은 seed별 query 및 반복 집계를 한 뒤 seed를 동일 가중치로 평균한 값이다.

| 방식 | Support-positive 비율 | Eligible 비율 | Full-positive 중 놓친 비율 | CPU 참조 구현 ms/query |
|---|---:|---:|---:|---:|
| Tail-512 | 53.7109% | 97.8809% | 41.0458% | 3.9524 |
| Random-512 | 62.5352% | 99.7305% | 31.1727% | 3.9910 |
| Full | 90.2734% | 99.9805% | 0.0000% | 6.3218 |

- Tail과 Random의 support mask 불일치율: **22.7266%**.
- Random의 conditional miss rate는 Tail보다 평균 **9.8731 percentage points** 낮았다.
- Conditional miss는 Full이 positive인 query에 대한 누락률이다. 전체 query의 오류율이나 학습 실패율이 아니다.
- 위 누락률은 seed별 비율의 평균이므로, 표의 평균 positive 비율끼리 나눈 값과 정확히 같을 필요가 없다.

### 7.3 의미와 한계

같은 조회량에서도 후보 선택에 따라 support 신호가 달라진다. 따라서 "512 제한과 index 순서가 결과에 영향을 주지 않는다"고 말할 수 없다. `support=0` 역시 전체 replay에 support가 없다는 뜻보다 **검사한 후보 안에서 확인되지 않았다**는 뜻으로 해석해야 한다.

반대로 Full 기준의 positive를 더 많이 찾는 것이 반드시 더 좋은 가중치 배정이나 학습 성능을 뜻하지는 않는다. 바로 이 차이를 확인하기 위해 다음 online 조회 실험을 진행 중이다.

이 진단은 최종 checkpoint만 본 것이다. 초기 학습에서의 누락률까지 측정했다고 할 수 없다. Query 10,240개를 독립된 학습 seed 10,240개처럼 취급하지도 않는다.

시간은 CPU 단일 thread의 공통 참조 구현에서 측정했다. 후보 열거와 선택은 포함하지만 actor 실행, JIT 준비, 전체 index 유지 및 학습 loop 비용을 대표하지 않는다. Production Numba 구현의 속도나 전체 학습 runtime으로 바꿔 해석하면 안 된다.

## 8. 완료: 실패 및 support 포화 분석

근거: [실행별 요약][D-RUNS], [사후 그룹 집계][D-GROUPS], [최종 보고서][D-FINAL].

### 8.1 분석 대상과 실패 정의

배정 실험 30회를 대상으로 했다. 여기서 **zero-evaluation run**은 예정된 200개 평가 checkpoint의 성공률이 전부 0인 실행이다. 수치 발산이나 프로그램 오류와 다르며, 학습 중 단 한 번도 성공 경험을 얻지 못했다는 뜻도 아니다.

| 방법 | Zero-evaluation seeds | 수 | Hit90 도달 |
|---|---|---:|---:|
| SABER | 7 | 1/10 | 9/10 |
| Random assignment | 3, 8, 9 | 3/10 | 7/10 |
| TD-error Top-k | 6, 7, 8 | 3/10 | 7/10 |

PAL의 zero-evaluation seeds는 별도로 2, 3이다. PAL은 support를 계산하지 않으므로 아래 support 포화 그룹 분석에는 넣지 않았다.

Random assignment seed 9와 TD-error Top-k seed 7은 학습 로그에 각각 positive episode 1개가 있었다. 따라서 이들을 "어떤 성공도 한 번도 없던 실행"으로 적으면 부정확하다.

### 8.2 포화의 의미와 관측

포화 update는 `k=|E|>0`인 경우다. 세 배정 방법이 모두 E를 선택하므로 배정 규칙의 차이가 사라진다. E가 minibatch 전체가 아니라면 weight가 완전히 uniform해지는 것은 아니다.

- Zero-evaluation 7회 모두에서 포화 update가 관측됐다.
- 나머지 23회에서는 포화 update가 관측되지 않았다.
- 하지만 처음 500k에서는 30회 모두 포화가 없었다. 이 자료만으로 학습 초기에 실패를 예측하는 지표라고 할 수 없다.

| 방법 | Zero 그룹 평균 support-positive | 나머지 그룹 평균 support-positive | Zero 그룹 평균 포화 update | 나머지 그룹 평균 포화 update |
|---|---:|---:|---:|---:|
| SABER | 55.9569% | 41.4901% | 27.9873% | 0% |
| Random assignment | 64.3849% | 44.4999% | 37.9203% | 0% |
| TD-error Top-k | 70.2935% | 42.4571% | 44.7395% | 0% |

처음 포화가 포함된 로그 window의 끝 step은 다음과 같다. 이는 정확한 최초 발생 step이 아니라 window 기반 위치다.

| 방법 | Seed | 최초 포화 포함 window 끝 step |
|---|---:|---:|
| SABER | 7 | 1,104,999 |
| Random assignment | 3 | 693,999 |
| Random assignment | 8 | 586,999 |
| Random assignment | 9 | 537,999 |
| TD-error Top-k | 6 | 1,049,999 |
| TD-error Top-k | 7 | 706,999 |
| TD-error Top-k | 8 | 965,999 |

### 8.3 해석

현재 결과는 **성과가 나오지 않은 실행에서 support가 높고 배정이 포화되는 현상이 함께 나타났다**는 사후 관찰이다. 포화가 실패를 일으켰는지, 실패한 policy/replay가 포화를 만들었는지, 제3의 요인이 있는지는 분리하지 못했다.

SABER의 높은 전체 평균에는 zero-evaluation run 수가 더 적다는 점이 반영되어 있다. 다만 1/10 대 3/10만으로 실패 확률의 확정적 감소를 입증한 것은 아니다.

참고로 zero-evaluation을 제외한 각 방법의 AUC 평균은 SABER 0.725126, Random 0.736683, TD 0.763747이다. **결과를 본 뒤 서로 다른 seed 집합을 골랐으므로 공정한 성능 순위가 아니다.** 실패 seed를 제외하거나 이 조건부 평균으로 primary 결과를 바꾸면 안 된다.

또한 정적 조회 진단에서 SABER seed 7은 Full support-positive가 100%였지만 AUC는 0이었다. 높은 support 활성화 자체가 유능한 policy나 높은 Q 정확도의 지표는 아니라는 사례다.

## 9. 사후 분석 중단 원인과 복구

근거: [복구 설명][D-RECOVERY], [복구 검증][D-VERIFY].

최초 사후 분석은 PAL 종료 이후 실행되었으나 checkpoint pilot loader에서 멈췄다. 저장된 config의 **`thread_hint`**를 알려진 runtime metadata로 처리하지 않고 미등록 학습 설정으로 거부한 것이 원인이었다.

이는 PAL 학습이 중단되거나 결과가 손상된 사건이 아니다. 최초판은 synthetic tests 41개와 배정 30회의 실패 분석을 통과한 뒤, checkpoint 조회 진단에 진입하는 과정에서 중단됐다.

복구는 원 폴더를 덮어쓰지 않고 독립 v2에서 수행했다. 알려진 runtime metadata만 허용하고, 미등록 설정이나 잘못된 값, 누락된 필수 학습 설정은 계속 거부하도록 했다. 전체 저장 config를 비교하는 검증도 유지했다.

| 복구 검증 | 결과 |
|---|---|
| 회귀 테스트 | 48개 통과 |
| 실패 분석 6개 산출물의 이전 결과 대조 | Byte-identical |
| 입력 및 출력 무결성 재검증 | 당시 218개 고유 파일 확인 |
| 정적 조회 | 10,240 query 전체 완료 |
| 직접 전체 replay cross-check | 160건 |
| Tail mask parity mismatch | 0 |
| 추가 학습 / 원 학습 결과 변경 | 없음 |

Byte-identical 확인 대상은 `evaluation_curves.csv`, `posthoc_groups.csv`, `report.json`, `run_summary.csv`, `time_bands.csv`, `window_trajectories.csv`다. 최종 복구판은 10월 4일 21:51 KST에 완료됐다.

## 10. 진행 중: Tail-512 vs Random-512 online 학습

근거: [고정 protocol][R-PROTOCOL], [현재 자동화 상태][R-STATUS], [historical reuse 검증][R-REUSE].

### 10.1 검증 질문과 앞선 Random의 차이

> SABER의 support 판정을 위해 같은 coarse bucket에서 최대 512개 후보를 검사할 때, Tail 대신 Random을 사용하는 것이 실제 학습에 어떤 영향을 주는가?

| 이름 | 무작위로 선택하는 대상 | Weight 계산 |
|---|---|---|
| 앞선 Random assignment | Minibatch에서 큰 weight를 받을 k개 위치 | E와 k를 고정하고 위치만 random |
| 현재 Random-512 retrieval | Support 판정에 사용할 replay 후보 | 조회 결과로 SABER support와 weight를 계산 |

현재는 두 arm 모두 SABER다. 후보 선택 방식만 바뀌며, 조회 결과에 따라 eligibility, support 및 k 자체가 달라질 수 있다. 앞선 실험과 달리 high-weight 개수를 두 조회 방법 사이에서 같게 맞추는 실험은 아니다.

### 10.2 구현 규칙

- 같은 coarse-key bucket의 유효하고 중복되지 않은 후보를 기준으로 최대 `min(512, bucket_size)`개를 선택한다.
- Tail은 유지된 index 순서의 끝부분을 사용한다. Temporal recency라는 의미를 부여하지 않는다.
- Random은 bucket에서 균일 비복원 추출한다. 전용 RNG는 `50000 + seed`다.
- Random subset은 같은 critic update의 동일 coarse key query 사이에서 공유하고 다음 update에서 다시 추출한다.
- 후보를 선택한 **뒤에** full-context matching과 action-bin support 검사를 적용한다. 정확히 일치하는 후보가 512개가 될 때까지 추가로 채우지 않는다.
- Bucket이 512개 이하이면 두 방식 모두 전체 후보 집합을 사용한다.
- Cache나 early exit로 실제 검사량은 상한보다 작을 수 있다. 동일 상한이 동일 실행시간을 보장하지 않는다.
- 기존 배정 실험의 support arm과 같은 counterfactual assignment 계측을 유지한다.

Tail 기준은 배정 실험의 SABER 10회 전체를 재사용하며 seed 7의 0 결과도 포함한다. 새로운 Random-512 10회는 fresh run이다. Historical reuse와 코드 parity 검증을 거쳤지만, 기존 결과를 본 뒤 설계한 후속 비교라는 점은 남는다.

### 10.3 현재 진행 상황

**아래 표는 2026-10-06 13:51:13 KST 기준이며 최종 결과가 아니다.**

| Seed | 마지막 로그 step | 2M 대비 | 마지막 평가 step | 최근 성공률 | 최고 성공률 | Hit90 | 최근 50k steps/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,407,000 | 70.35% | 1,400,000 | 1.00 | 1.00 | 420k | 8.509 |
| 2 | 1,447,000 | 72.35% | 1,440,000 | 0.00 | 0.00 | 미도달 | 9.717 |
| 3 | 1,457,800 | 72.89% | 1,450,000 | 0.00 | 0.00 | 미도달 | 9.744 |
| 4 | 1,453,400 | 72.67% | 1,450,000 | 0.95 | 1.00 | 670k | 9.004 |
| 5 | 1,440,800 | 72.04% | 1,440,000 | 1.00 | 1.00 | 870k | 8.902 |
| 6-10 | 0 | 0% | 없음 | 없음 | 없음 | 대기 | 없음 |

5개 worker가 살아 있고 로그가 계속 진행하고 있었다. 확인한 production 및 automation 로그에서 traceback, CUDA error, OOM, non-finite 실패 등 실행 오류 패턴은 발견하지 못했다. Seeds 2, 3의 현재 0 성공률은 프로그램 중단을 뜻하지 않는다.

### 10.4 정확히 같은 구간의 중간 AUC

Seeds 1-5의 비교 구간을 **10k-1.4M**으로 고정했다. 140개 checkpoint, 분모는 1,390,000이다. Tail 역시 같은 seed와 같은 구간으로 다시 계산했다.

| Seed | Tail-512 AUC | Random-512 AUC |
|---|---:|---:|
| 1 | 0.680216 | 0.687050 |
| 2 | 0.640288 | 0.000000 |
| 3 | 0.702518 | 0.000000 |
| 4 | 0.744604 | 0.508993 |
| 5 | 0.494245 | 0.402518 |
| 평균 | **0.652374** | **0.319712** |

현재 평균 `Random-512 - Tail-512 = -0.332662`이며 5개 중 4개 seed에서 Tail의 AUC가 높다. 이는 현재 상태를 기술하는 값이다. **전체 10-seed, 2M 최종 CI는 아직 계산할 수 없다.** 이 중간 표를 전체 10-seed SABER 최종 AUC 0.652613과 혼동하면 안 된다.

최종 분석의 사전 고정 contrast는 **Random-512 minus Tail**이다. 현재 Tail에 유리하다고 primary 방향이나 seed 수를 바꾸지 않는다. 최종 bootstrap은 10,000회, seed 20261004, 단일 primary contrast의 95% percentile CI다.

### 10.5 최근 support 포화

각 실행의 최근 약 50k update 구간을 유효 update 수로 가중 집계했다. 실행별 마지막 step이 조금씩 달라 서로 완전히 같은 시간 구간은 아니다.

| Seed | Window 끝 step | Eligible 비율 | Support-positive 비율 | 포화 update 비율 |
|---|---:|---:|---:|---:|
| 1 | 1,406,999 | 99.5127% | 65.5052% | 0.0000% |
| 2 | 1,446,999 | 100.0000% | 99.8719% | 84.4140% |
| 3 | 1,456,999 | 100.0000% | 100.0000% | 100.0000% |
| 4 | 1,452,999 | 99.4875% | 62.2517% | 0.0000% |
| 5 | 1,439,999 | 99.5299% | 64.7485% | 0.0000% |

처음 포화가 포함된 window의 끝은 seed 2에서 1,186,999, seed 3에서 841,999였다. 현재 낮은 성과와 높은 포화가 함께 보이지만, 완료된 사후 분석과 마찬가지로 인과관계는 확정하지 않는다.

### 10.6 예상 종료 시점

현재 5개 worker의 최근 50k 구간 평균은 약 **9.175 steps/s**다. 5병렬을 유지하고 빈 worker에 다음 seed를 넣으며 이후 속도가 비슷하다는 가정으로 계산했다.

| 대상 | 기준 시각에서 예상 잔여 시간 | 예상 KST |
|---|---:|---|
| 현재 seeds 1-5가 모두 완료 | 약 19.4 h | 10월 7일 오전 9시 전후 |
| Seeds 1-10 본학습 전체 완료 | 약 79.9 h | 10월 9일 밤 9-10시 전후 |

최근 100k 또는 200k 구간 속도로 계산해도 전체 종료 추정은 10월 9일 밤 9시대였다. 이는 통계적 예측구간이 아니라 현재 속도 기반 외삽이다. 시스템 부하, buffer 상태, checkpoint 저장 등에 따라 달라지며, 종료 후 분석 시간은 포함하지 않았다.

### 10.7 종료 후 이미 예약된 작업

실제 [자동화 코드][R-AFTER]에 연결된 순서는 다음과 같다.

1. Random-512 10회 학습 완료.
2. 완료 상태, 설정, reference 및 전체 평가 grid 검증.
3. 최종 AUC, FQ-AUC, Final, Hit90, seed별 결과 및 paired CI 생성.
4. Tail 10개와 Random-512 10개, 총 20개 최종 checkpoint에서 추가 CPU 조회 진단.
5. 조회 진단 report와 manifest 생성.

추가 조회 진단은 checkpoint당 1,024개 query, 총 20,480개이며 Random 조회 5회 반복과 seed/arm별 16개 직접 전체 replay 대조가 계획돼 있다. 이 분석은 양쪽 학습으로 달라진 replay/policy 분포를 함께 보는 작업이며, 먼저 완료한 Tail checkpoint만의 진단과 구분한다.

실패하면 blocked 상태를 남겨 원인을 확인하도록 되어 있다. 새 online 실험이나 critic 분기 학습, commit/push는 이 자동화에 예약되어 있지 않다.

## 11. 코드 및 자료 검증 상태

### 11.1 본학습 전에 남긴 검증 기록

아래 테스트는 기존 gate 산출물을 확인한 것이다. 이번 문서 작성 중 GPU smoke나 전체 테스트를 새로 실행하지는 않았다.

| 실험 | 기존 검증 근거 | 결과 |
|---|---|---|
| 배정 비교 | 회귀 테스트, 각 arm 6k smoke, checkpoint audit | 33 tests, 각 1,001 critic updates, gate 통과 |
| PAL v2 | CPU 테스트, CUDA loss/gradient 대조, 6k smoke, reference 감사 | 39 tests; CUDA 26개 residual 사례에서 loss/gradient 최대 차이 0; gate 통과 |
| 사후 진단 v2 | Metadata 처리와 조회 회귀 테스트, 원 분석 대조 | 48 tests; 실패 분석 산출물 동일 |
| Online retrieval | 회귀 테스트, Tail parity, Tail/Random smoke, historical reuse | 40 tests; gate 통과 |

Online retrieval의 Tail parity는 6k 실행에서 모델, optimizer 관련 상태, replay, 평가 이력, RNG 상태, update 수 및 assignment trace 등 비교 항목이 일치하는지 확인한 것이다. Metadata가 다른 checkpoint 파일 전체의 byte-identical을 주장하는 것은 아니다.

PAL의 사전 준비 파일에 남아 있는 `production_started=false`나 초기 README의 pending 표현은 작성 당시 상태다. 현재 완료 여부는 `outputs/automation/status.json`과 최종 run 결과를 기준으로 판단한다.

### 11.2 이번 보고서 작성 과정의 읽기 전용 재검증

- 완료된 40회 모두 single attempt, `complete`, `checkpoint_continuation=false` 확인.
- Run마다 200개, 총 8,000개 예정 평가 checkpoint의 존재, 중복 없음, 마지막 2M step 확인.
- 원시 평가 CSV에서 AUC, FQ-AUC, Final을 재계산하여 저장된 최종 분석과 `1e-12` 이내 일치 확인.
- Warmup 이후 기록된 핵심 loss/Q/TD 열에서 non-finite 값이 없는지 확인.
- 네 canonical source manifest의 **250개 file entry**를 SHA256으로 재검증, 누락 및 불일치 0. 서로 다른 manifest 사이의 중복 파일을 제거한 고유 파일 수라는 뜻은 아니다.
- 진행 중인 5개 worker의 상태와 로그 진행을 확인.
- CI는 고정된 최종 분석 산출물의 값을 사용했다. 문서 작성 과정에서 새 bootstrap이나 다른 통계 절차로 교체하지 않았다.

| Manifest 대상 | File entry 수 | Manifest 파일 SHA256 |
|---|---:|---|
| Assignment | 60 | `d281263084807ec6010986dc762301dce420b004e022c88985e51dcc50ccf496` |
| PAL v2 | 76 | `05e08b06dd1938a839146cdd5ad0ccde2c09c309a017c3c77d8fe524061c8860` |
| Offline diagnostics v2 | 42 | `4d660f281ff0f2a63d16858ca7e2cf51756d9790b63f7514d03b27e88a8f204f` |
| Online retrieval | 72 | `762600452bf67e48481aecb781c617afd9faeea4b965e6bb05085c0735d44d5a` |

각 attempt에는 전체 config, environment 정보, pip freeze, status와 로그가 보존된다. `run/env_snapshot.json`은 task 환경 상태 기록이며 전체 패키지 버전 기록을 대신하지 않는다.

## 12. 현재 가능한 주장과 불가능한 주장

| 현재 자료가 지지하는 표현 | 피해야 하는 확대 해석 |
|---|---|
| DrawerOpen-DDPG의 추가 10-seed 비교에서 SABER가 세 대조군보다 높은 평균 AUC를 보였다 | 모든 비교에서 통계적으로 확정적인 우위를 보였다 |
| 배정 실험이 weight 형태와 개수를 통제한 새로운 mechanism evidence를 제공한다 | Exact action-bin support만이 개선의 직접적이고 유일한 원인이다 |
| PAL이라는 구체적인 loss-side 선행방법과 비교했다 | PAL보다 모든 환경에서 우월하므로 신규성이 입증됐다 |
| 최종 checkpoint의 Tail-512는 현재 context의 Full-positive 일부를 놓친다 | Tail이 전체 replay support를 정확히 판정한다 |
| Random-512가 정적 Full 기준 누락률을 낮췄다 | Random 조회가 더 좋은 학습 성능을 보장한다 |
| 진행 중인 5-seed 공통 구간에서는 Tail의 AUC가 높다 | Tail이 최적이고 Random보다 확실히 우월하다 |
| Zero-evaluation run과 support 포화가 함께 관측됐다 | Support 포화가 실패의 원인임을 증명했다 |
| 계측을 포함한 실제 실행 시간이 기록됐다 | 현재 속도 로그만으로 공정한 알고리즘 효율 순위를 정할 수 있다 |

새 cohort에는 별도 fresh HER-only 10-seed arm이 없다. 따라서 이 40회만으로 새로운 10-seed SABER-minus-HER 개선량을 계산할 수 없다. 기존 논문의 HER 비교와 후속 cohort를 구분해 보고해야 한다.

여러 후속 비교에서 같은 SABER 10회를 재사용했다. 이를 서로 독립적인 SABER 반복 재현 세 건처럼 세거나 증거량을 합산해서는 안 된다. 완료 결과가 확정한 것은 각 비교의 관측 분포이며, CI의 불확실성은 남아 있다.

## 13. 아직 실행하지 않았거나 예약하지 않은 작업

아래는 대화에서 논의한 후보 목록이다. 이 문서 작성이 실행 승인이나 예약을 뜻하지 않는다.

| 후보 | 목적 | 현재 상태 / 다음 설계 조건 |
|---|---|---|
| 동일 checkpoint critic 분기 진단 | Replay-action 가중 업데이트가 policy-action 위치의 예측에 도움이 되는지 | 미실행. Critic/optimizer 복제, actor/target/batch 고정, 별도 평가 필요 |
| N_local, N_scan, lambda, context 구간 민감도 | 수동 설정 의존성 평가 | 미실행. 값과 seed 예산을 실행 전에 고정해야 함 |
| 공통 context 단순화 및 새 task 전이 | Task-specific engineering 부담 평가 | 미실행. 전이 전에 규칙 고정, task별 재튜닝과 구분 필요 |
| PAL / mechanism의 다른 task 또는 backbone 확장 | DrawerOpen-DDPG 범위 밖의 외적 타당성 | 미실행. 평가 대상과 예산 미확정 |
| DrawerClose SAC | 기존 benchmark/backbone 범위 확장 | 후보 논의만 있었음; 이 작업공간의 해당 추가 cohort 없음 |
| 동일 계측 조건의 HER / SABER / PAL wall-clock 비교 | 실제 시간 대비 학습 이득 평가 | 미실행. Hardware, 병렬 부하, 진단량을 맞춰야 함 |
| LAP 비교 | Sampling-side 관련 선행방법 범위 보강 | 논의만 있었음. 현재 PAL과 별도 실험 |
| 추가 seed 확장 | 중요한 효과 추정의 정밀도 개선 | 미예약. 유리한 CI가 나올 때까지 늘리는 방식은 피해야 함 |

Policy-action 진단에서 replay action으로 얻은 reward와 next state를 다른 policy action의 정답으로 그대로 사용해서는 안 된다. 실제 policy action의 transition이나 고정 정책 rollout을 얻어 평가 기준을 구성해야 한다. Fixed-target Bellman residual만 줄었다면 true Q error 감소와 구분한다.

기존 제출 자료의 K=3/5/7 민감도, FPP, DrawerClose 등의 실험은 별도 선행 자료다. 위 표에서 "미실행"은 리뷰 이후 이 보고서가 다루는 새로운 후속 작업을 뜻하며, 과거 모든 실험이 없다는 뜻이 아니다.

### 13.1 학습 없이 병행해야 할 원고 작업

- Method를 예시, support 조건, normalized objective, 구현 세부 순서로 재구성.
- Current-policy action 조회와 replay-action residual 회귀를 그림에서 구분.
- Context 구성표와 짧은 pseudocode를 본문에 배치.
- Diagnostic 수치 나열을 compact 표 또는 그림으로 전환.
- 신규성의 중심을 일반적인 reweighting 자체보다 구체적인 support 신호와 검증 질문으로 한정.
- PAL 및 actor-aware replay와의 관계를 정확히 설명.
- 후속 10-seed cohort, 원 제출 cohort, 완료 결과, 중간 결과를 혼합하지 않음.

현재 자료는 추가 근거와 실패 사례를 모두 제공한다. 다음 원고는 "모든 실험에서 우월하다"보다 **무엇을 통제해 비교했고, 어느 범위에서 어떤 평균 이득과 불확실성을 관측했는가**를 더 선명하게 전달해야 한다.

## 14. 완료된 40회 학습의 seed별 지표

모든 행은 2M 완료 결과다. AUC는 10k-2M, FQ-AUC는 1.5M-2M이며, Hit90의 `NA`는 미도달을 뜻한다. PAL 비교 및 Tail 기준으로 재사용된 SABER는 아래 첫 10행과 같은 실행이다.

| 방법 | Seed | AUC | FQ-AUC | Final | Hit90 |
|---|---:|---:|---:|---:|---:|
| SABER | 1 | 0.776633 | 1.0000 | 1.00 | 460k |
| SABER | 2 | 0.744975 | 0.9850 | 1.00 | 520k |
| SABER | 3 | 0.789070 | 0.9875 | 0.95 | 450k |
| SABER | 4 | 0.818342 | 0.9920 | 1.00 | 350k |
| SABER | 5 | 0.646734 | 1.0000 | 1.00 | 720k |
| SABER | 6 | 0.765578 | 0.9890 | 1.00 | 470k |
| SABER | 7 | 0.000000 | 0.0000 | 0.00 | NA |
| SABER | 8 | 0.576382 | 0.9980 | 1.00 | 830k |
| SABER | 9 | 0.570980 | 0.9775 | 0.95 | 850k |
| SABER | 10 | 0.837437 | 1.0000 | 1.00 | 340k |
| Random assignment | 1 | 0.869347 | 1.0000 | 1.00 | 250k |
| Random assignment | 2 | 0.739447 | 0.9990 | 1.00 | 530k |
| Random assignment | 3 | 0.000000 | 0.0000 | 0.00 | NA |
| Random assignment | 4 | 0.690704 | 1.0000 | 1.00 | 600k |
| Random assignment | 5 | 0.725879 | 0.9940 | 1.00 | 500k |
| Random assignment | 6 | 0.714824 | 0.9930 | 1.00 | 590k |
| Random assignment | 7 | 0.732663 | 0.9930 | 1.00 | 550k |
| Random assignment | 8 | 0.000000 | 0.0000 | 0.00 | NA |
| Random assignment | 9 | 0.000000 | 0.0000 | 0.00 | NA |
| Random assignment | 10 | 0.683920 | 0.9980 | 1.00 | 660k |
| TD-error Top-k | 1 | 0.807538 | 0.9990 | 1.00 | 390k |
| TD-error Top-k | 2 | 0.792965 | 0.9970 | 1.00 | 420k |
| TD-error Top-k | 3 | 0.512814 | 0.9880 | 1.00 | 920k |
| TD-error Top-k | 4 | 0.809799 | 0.9980 | 1.00 | 400k |
| TD-error Top-k | 5 | 0.773367 | 0.9910 | 1.00 | 460k |
| TD-error Top-k | 6 | 0.000000 | 0.0000 | 0.00 | NA |
| TD-error Top-k | 7 | 0.000000 | 0.0000 | 0.00 | NA |
| TD-error Top-k | 8 | 0.000000 | 0.0000 | 0.00 | NA |
| TD-error Top-k | 9 | 0.808543 | 0.9970 | 1.00 | 400k |
| TD-error Top-k | 10 | 0.841206 | 0.9660 | 1.00 | 330k |
| PAL | 1 | 0.715075 | 0.9980 | 1.00 | 590k |
| PAL | 2 | 0.000000 | 0.0000 | 0.00 | NA |
| PAL | 3 | 0.000000 | 0.0000 | 0.00 | NA |
| PAL | 4 | 0.818342 | 0.9990 | 1.00 | 400k |
| PAL | 5 | 0.557035 | 0.9830 | 1.00 | 850k |
| PAL | 6 | 0.689698 | 0.9980 | 1.00 | 640k |
| PAL | 7 | 0.809296 | 0.9950 | 1.00 | 410k |
| PAL | 8 | 0.691206 | 0.9620 | 1.00 | 600k |
| PAL | 9 | 0.509045 | 0.9930 | 1.00 | 930k |
| PAL | 10 | 0.663819 | 0.9980 | 1.00 | 650k |

### 14.1 Seed별 paired AUC 차이

| Seed | SABER - Random | SABER - TD top-k | SABER - PAL |
|---|---:|---:|---:|
| 1 | -0.092714 | -0.030905 | +0.061558 |
| 2 | +0.005528 | -0.047990 | +0.744975 |
| 3 | +0.789070 | +0.276256 | +0.789070 |
| 4 | +0.127638 | +0.008543 | +0.000000 |
| 5 | -0.079146 | -0.126633 | +0.089698 |
| 6 | +0.050754 | +0.765578 | +0.075879 |
| 7 | -0.732663 | +0.000000 | -0.809296 |
| 8 | +0.576382 | +0.576382 | -0.114824 |
| 9 | +0.570980 | -0.237563 | +0.061935 |
| 10 | +0.153518 | -0.003769 | +0.173618 |

## 15. 자료 위치와 재확인 방법

이 문서는 `ICML_27/`에서 관리한다. 아래 상대 링크는 같은 로컬 저장소의 `../experiments_post_review/`에 보존된 실험 자료를 가리킨다. 이번 커밋에는 보고서만 포함하므로, 해당 자료를 별도로 게시하기 전까지 GitHub 웹에서는 이 로컬 자료 링크가 열리지 않는다. 익명 제출용 package를 새로 생성하거나 공개한 문서는 아니다.

| 자료 | 위치 |
|---|---|
| 배정 비교 설계 | [PROTOCOL.md][A-PROTOCOL] |
| 배정 비교 최종 결과 | [report.json][A-RESULT] |
| 배정 비교 seed별 CSV | [seed_metrics.csv][A-SEEDS] |
| 배정 상세 결과 및 진단 | [final_details_20261002][A-DETAIL] |
| PAL 설계 | [PROTOCOL.md][P-PROTOCOL] |
| PAL 최종 결과 | [report.json][P-RESULT] |
| PAL seed별 CSV | [seed_metrics.csv][P-SEEDS] |
| PAL 검증 기록 | [validation][P-VALIDATION] |
| 사후 분석 최종 설명 | [REPORT.md][D-FINAL] |
| 사후 분석 복구 근거 | [RECOVERY.md][D-RECOVERY] |
| 정적 조회 요약 | [aggregate_summary.csv][D-RETRIEVAL] |
| 실패 분석 실행별 요약 | [run_summary.csv][D-RUNS] |
| 실패 분석 그룹 집계 | [posthoc_groups.csv][D-GROUPS] |
| Online 조회 실험 설계 | [PROTOCOL.md][R-PROTOCOL] |
| Online 조회 실험 상태 | [status.json][R-STATUS] |
| Tail 재사용 감사 | [reuse_catalog.json][R-REUSE] |
| 종료 후 자동화 | [after_training.py][R-AFTER] |
| 진행 중 seed 1 원시 학습 로그 | [metrics.csv][R-METRICS] |

저장소 root에서 아래는 상태와 기존 결과를 읽는 명령이다. 새 학습이나 분석 재실행을 시작하지 않는다.

```bash
cat experiments_post_review/2026-10-04_draweropen_ddpg_retrieval_v1/outputs/automation/status.json
tail -n 5 experiments_post_review/2026-10-04_draweropen_ddpg_retrieval_v1/outputs/production/random512/seed_01/attempt_01/run/metrics.csv
cat experiments_post_review/2026-10-04_draweropen_retrieval_failure_diagnostics_v2/outputs/final/REPORT.md
jq '{arm_summary, inference, problems, runtime_comparison_valid}' experiments_post_review/2026-10-01_draweropen_ddpg_pal_v2/outputs/analysis/report.json
```

**최종 요약:** 리뷰 이후 비교의 통제 수준, 선행방법 coverage, 조회 방식 및 실패 양상에 관한 자료는 늘었다. SABER의 평균 성능 이득은 관측되지만 support 배정의 독자적 필요성과 보편적인 우월성은 아직 확정하지 못했다. 현재 Random-512 10회와 예약 분석을 끝까지 완료하고, 그 결과를 포함해 주장 범위를 정하는 것이 다음 단계다.

[A-PROTOCOL]: ../experiments_post_review/2026-09-25_draweropen_ddpg_assignment/PROTOCOL.md
[A-RESULT]: ../experiments_post_review/2026-09-25_draweropen_ddpg_assignment/outputs/analysis/report.json
[A-SEEDS]: ../experiments_post_review/2026-09-25_draweropen_ddpg_assignment/outputs/analysis/seed_metrics.csv
[A-DETAIL]: ../experiments_post_review/2026-09-25_draweropen_ddpg_assignment/outputs/analysis/final_details_20261002
[P-PROTOCOL]: ../experiments_post_review/2026-10-01_draweropen_ddpg_pal_v2/PROTOCOL.md
[P-RESULT]: ../experiments_post_review/2026-10-01_draweropen_ddpg_pal_v2/outputs/analysis/report.json
[P-SEEDS]: ../experiments_post_review/2026-10-01_draweropen_ddpg_pal_v2/outputs/analysis/seed_metrics.csv
[P-NOTICES]: ../experiments_post_review/2026-10-01_draweropen_ddpg_pal_v2/THIRD_PARTY_NOTICES.md
[P-VALIDATION]: ../experiments_post_review/2026-10-01_draweropen_ddpg_pal_v2/validation
[D-PROTOCOL]: ../experiments_post_review/2026-10-04_draweropen_retrieval_failure_diagnostics_v2/PROTOCOL.md
[D-FINAL]: ../experiments_post_review/2026-10-04_draweropen_retrieval_failure_diagnostics_v2/outputs/final/REPORT.md
[D-RETRIEVAL]: ../experiments_post_review/2026-10-04_draweropen_retrieval_failure_diagnostics_v2/outputs/retrieval/aggregate_summary.csv
[D-RUNS]: ../experiments_post_review/2026-10-04_draweropen_retrieval_failure_diagnostics_v2/outputs/failure_analysis/run_summary.csv
[D-GROUPS]: ../experiments_post_review/2026-10-04_draweropen_retrieval_failure_diagnostics_v2/outputs/failure_analysis/posthoc_groups.csv
[D-RECOVERY]: ../experiments_post_review/2026-10-04_draweropen_retrieval_failure_diagnostics_v2/RECOVERY.md
[D-VERIFY]: ../experiments_post_review/2026-10-04_draweropen_retrieval_failure_diagnostics_v2/outputs/recovery_verification/report.json
[R-PROTOCOL]: ../experiments_post_review/2026-10-04_draweropen_ddpg_retrieval_v1/PROTOCOL.md
[R-STATUS]: ../experiments_post_review/2026-10-04_draweropen_ddpg_retrieval_v1/outputs/automation/status.json
[R-REUSE]: ../experiments_post_review/2026-10-04_draweropen_ddpg_retrieval_v1/validation/reuse_catalog.json
[R-AFTER]: ../experiments_post_review/2026-10-04_draweropen_ddpg_retrieval_v1/after_training.py
[R-METRICS]: ../experiments_post_review/2026-10-04_draweropen_ddpg_retrieval_v1/outputs/production/random512/seed_01/attempt_01/run/metrics.csv

# DrawerOpen DDPG-HER-B3 Critic-Only 10-Seed 진단 보고서

작성일: 2026-07-13 UTC
대상 태스크: MetaWorld `drawer-open-v3`
대상 방법: `DDPG + HER + SABER/B3 support weighting (critic only)`

## 1. 목적과 결론 요약

이 문서는 `DDPG-HER-B3-Critic-Only-2M`의 온라인 diagnostic logging으로 다음 질문을 검증한 팀 공유용 보고서다.

1. B3 support predicate가 critic이 어려워하는 샘플을 선별하는가?
2. Support-positive 샘플은 한 번의 업데이트 후 더 크게 교정되는가?
3. 진단 지표가 AUC와 학습 속도에 연결되는가?
4. 성공 seed와 실패 seed는 언제, 어떤 신호로 갈라지는가?
5. Seed 7 실패는 수치 불안정, 탐색 실패, support 포화 중 무엇으로 설명되는가?
6. 현재 증거로 주장 가능한 범위와 추가 검증이 필요한 범위는 어디까지인가?

핵심 결론은 다음과 같다.

- 최신 diagnostic seeds 1–10 중 9개가 성공했다.
- Support-positive 샘플은 support-negative보다 업데이트 전 TD error가 평균 17% 높았다.
- 같은 weighted minibatch 업데이트에서 residual 감소 폭은 support-negative보다 약 2.1배 컸다.
- B3는 풍부한 HER-open transition보다 희소한 approach/contact/opening transition을 훨씬 자주 강조했다.
- Residual 우위가 큰 seed일수록 AUC가 높고 Hit90이 빠른 탐색적 상관이 있었다.
- Seed 7은 수치 폭발이 아니라 400–800k 구간의 접촉 탐색 실패 후 critic 신호가 사라지고 support가 비구분적으로 포화된 failure mode다.
- 현재 residual 진단은 동일 weighted minibatch의 in-sample 측정이므로 일반화 향상의 인과 증거로는 부족하다.

가장 적절한 메커니즘 설명은 다음과 같다.

> SABER/B3는 HER replay에서 풍부하고 쉽게 달성되는 relabeled transition보다 희소한 상호작용 진행 transition에 critic 학습 용량을 재배분한다. 이 재배분은 높은 TD error와 더 큰 로컬 residual correction으로 관측된다. 다만 의미 있는 접촉 경험이 생성되지 않으면 B3는 탐색 실패를 복구하지 못하고 support 신호가 포화될 수 있다.

## 2. 분석 대상과 실행 조건

### 2.1 사용한 run

| Seed | 실행 | 결과 위치 | 실행 시간 |
| --- | --- | --- | --- |
| 1–5 | diagnostic logging 재실행 | `results_2m_diag/DDPG-HER-B3-Critic-Only-2M-diagnostic/` | 2026-07-11 06:21–2026-07-12 14:30 UTC |
| 6–10 | 추가 seed diagnostic run | `results_2m/DDPG-HER-B3-Critic-Only-2M/` | 2026-07-09 23:14–2026-07-11 06:19 UTC |

공통 조건:

- Seed당 2,000,000 environment steps
- 평가 주기 10,000 step, 평가 episode 20
- Replay capacity 300,000
- Batch size 256
- HER relabel 설정 0.8, 실제 관측 평균 약 0.794
- Support raw weight: negative 1, positive 2
- Batch-mean normalization
- Critic-only weighting
- `local_search_max=512`

### 2.2 Diagnostic 정의

각 업데이트에서 다음을 기록한다.

- `support = psi_q_rank > 0.5`
- `td_before`: critic 업데이트 전 절대 TD error
- `td_after`: critic·actor 업데이트와 target soft update 후 같은 minibatch의 절대 TD error
- `residual_reduction = td_before - td_after`

```text
TD 차이
  = mean(td_before | support+) - mean(td_before | support-)

Residual 차이
  = mean(td_before - td_after | support+)
  - mean(td_before - td_after | support-)

Residual 개선률 차이
  = P(td_before - td_after > 0 | support+)
  - P(td_before - td_after > 0 | support-)
```

Diagnostic logging은 replay sampling, loss weight, optimizer step, policy update를 변경하지 않는다. 업데이트 후 `no_grad` forward를 추가하므로 실행 시간만 증가한다.

### 2.3 데이터 정리와 통계

`metrics.csv`에는 같은 step의 episode 행과 evaluation 행이 중복될 수 있다. 같은 마지막 minibatch를 가리키므로 다음 절차를 적용했다.

1. 동일 step diagnostic 행 중복 제거
2. Seed당 9,976개 고유 스냅샷 확보
3. 시간 스냅샷은 자기상관이 있으므로 독립 표본으로 사용하지 않음
4. Seed별 전체 구간 평균을 독립 통계 단위로 사용
5. Seed-level Wilcoxon signed-rank test와 t interval 사용
6. 성능 관계는 Spearman rank correlation 사용
7. 다중 비교 보정 전 상관 p-value는 탐색적으로 해석

## 3. 성능과 재현성

### 3.1 전체 결과

| 구간 | 성공 seed | 평균 Final | 평균 AUC | 평균 Hit90 |
| --- | ---: | ---: | ---: | ---: |
| 기존 B3 seeds 1–5 | 5/5 | 1.000 | 0.754 | 482k |
| Diagnostic 재실행 seeds 1–5 | 5/5 | 0.990 | 0.755 | 500k |
| 추가 seeds 6–10 | 4/5 | 0.790 | 0.550 | 623k* |
| 최신 diagnostic seeds 1–10 | 9/10 | 0.890 | 0.653 | 554k* |

`*` Hit90은 90%에 도달한 seed만 평균했다.

- 10-seed 성공률: `90%`
- 성공률 Wilson 95% CI: `[59.6%, 98.2%]`
- 10-seed AUC: `0.6526`, 95% CI `[0.4749, 0.8303]`
- 성공 9-seed AUC: `0.7251`, 95% CI `[0.6473, 0.8030]`

성공률은 높지만 seed 수가 10개이므로 신뢰구간은 아직 넓다.

### 3.2 동일 seeds 1–5 재실행

| Seed | 기존 AUC | Diagnostic AUC | 기존 Hit90 | Diagnostic Hit90 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.757 | 0.777 | 510k | 460k |
| 2 | 0.787 | 0.745 | 420k | 520k |
| 3 | 0.705 | 0.789 | 580k | 450k |
| 4 | 0.713 | 0.818 | 500k | 350k |
| 5 | 0.806 | 0.647 | 400k | 720k |
| 평균 | 0.754 | 0.755 | 482k | 500k |

Aggregate 성능은 거의 동일하게 재현됐지만 개별 trajectory는 달랐다. Diagnostic logging이 평균 성능을 체계적으로 바꿨다는 징후는 없지만 동일 seed가 동일 학습 경로를 보장한다고 가정해서는 안 된다.

### 3.3 최신 diagnostic run의 seed별 성능

| Seed | Final | AUC | Hit90 | 판정 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.00 | 0.777 | 460k | 성공 |
| 2 | 1.00 | 0.745 | 520k | 성공 |
| 3 | 0.95 | 0.789 | 450k | 성공 |
| 4 | 1.00 | 0.818 | 350k | 성공 |
| 5 | 1.00 | 0.647 | 720k | 성공 |
| 6 | 1.00 | 0.766 | 470k | 성공 |
| 7 | 0.00 | 0.000 | — | 실패 |
| 8 | 1.00 | 0.576 | 830k | 성공 |
| 9 | 0.95 | 0.571 | 850k | 성공 |
| 10 | 1.00 | 0.837 | 340k | 성공 |

## 4. Seed별 diagnostic

### 4.1 주요 효과

| Seed | 평균 support+ | 마지막 400k support+ | TD 차이 | Residual 차이 | Residual 개선률 차이 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3891 | 0.4537 | +0.0658 | +0.00406 | +0.01110 |
| 2 | 0.4160 | 0.5106 | +0.0660 | +0.00521 | +0.00877 |
| 3 | 0.4125 | 0.4665 | +0.0706 | +0.00561 | +0.01110 |
| 4 | 0.4220 | 0.4797 | +0.0226 | +0.00540 | +0.00958 |
| 5 | 0.4464 | 0.5201 | +0.00755 | +0.00270 | +0.00448 |
| 6 | 0.3850 | 0.4348 | +0.0608 | +0.00536 | +0.00951 |
| 7 | 0.5598 | 0.9928 | +0.00847** | +0.00014** | -0.00425** |
| 8 | 0.4054 | 0.4783 | +0.0376 | +0.00336 | +0.00635 |
| 9 | 0.4037 | 0.4879 | +0.0405 | +0.00358 | +0.00773 |
| 10 | 0.4529 | 0.5007 | +0.0472 | +0.00413 | +0.00830 |

`**` Seed 7은 후반 support-negative가 사라져 두 집단이 모두 존재한 86.7%의 스냅샷에서만 차이가 정의된다.

### 4.2 상관과 실행 지표

| Seed | Support–TD corr | Support–residual corr | Normalized ESS | Scan truncation |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1037 | 0.0187 | 0.8956 | 0.9772 |
| 2 | 0.1152 | 0.0266 | 0.9001 | 0.9766 |
| 3 | 0.1083 | 0.0259 | 0.8964 | 0.9767 |
| 4 | 0.0456 | 0.0180 | 0.8985 | 0.9805 |
| 5 | 0.0461 | 0.0123 | 0.9019 | 0.9832 |
| 6 | 0.1037 | 0.0186 | 0.8984 | 0.9798 |
| 7 | 0.1060** | -0.0003** | 0.9321 | 0.9893 |
| 8 | 0.0950 | 0.0159 | 0.8978 | 0.9820 |
| 9 | 0.1111 | 0.0149 | 0.8993 | 0.9781 |
| 10 | 0.0839 | 0.0204 | 0.8997 | 0.9832 |

성공 seed의 평균 support 비율은 `0.4148`, 마지막 400k에서는 `0.4814`였다. 평균 normalized ESS는 `0.8986`이었다.

## 5. Support가 높은 TD-error 샘플을 선별하는가?

성공한 9개 seed의 결과:

| 지표 | Support-positive | Support-negative | 차이/비율 |
| --- | ---: | ---: | ---: |
| 업데이트 전 절대 TD error | 0.3204 | 0.2739 | `+0.0465`, `1.170x` |

- Seed-level TD 차이: `+0.0465`
- 95% CI: `[+0.0299, +0.0631]`
- 9/9 seed에서 양수
- t-test: `p=0.000197`
- Wilcoxon: `p=0.003906`
- 평균 support–TD correlation: `0.0903`
- Support–TD correlation도 9/9 seed에서 양수, Wilcoxon `p=0.003906`

따라서 support predicate가 critic이 아직 잘 맞추지 못하는 샘플을 일관되게 선별한다는 증거는 강하다.

다만 TD 차이는 성능과 강하게 연결되지 않았다.

| 관계 | Spearman rho | p |
| --- | ---: | ---: |
| TD 차이 vs AUC | 0.267 | 0.488 |
| TD 차이 vs Hit90 | -0.267 | 0.488 |

어려운 샘플을 찾는 것만으로 최종 성능이나 학습 속도를 설명할 수는 없다.

## 6. Support-positive 샘플이 더 크게 교정되는가?

| 지표 | Support-positive | Support-negative | 차이 |
| --- | ---: | ---: | ---: |
| 평균 residual 감소 | 0.00843 | 0.00403 | `+0.00440` |
| Residual 감소 발생률 | 51.21% | 50.36% | `+0.85%p` |

- Seed-level residual 차이: `+0.00438`
- 95% CI: `[+0.00357, +0.00519]`
- 9/9 seed에서 양수
- Wilcoxon: `p=0.003906`
- Residual-positive-rate 차이: `+0.00854`
- Residual-positive-rate 차이 95% CI: `[+0.00689, +0.01020]`
- 평균 support–residual correlation: `0.0190`

평균 residual 감소는 약 2.1배지만 감소 발생 빈도 차이는 0.85%p다. 더 많은 샘플이 개선되기보다, 개선되는 일부 샘플의 감소 폭이 큰 형태다.

중요한 한계는 같은 weighted minibatch에서 업데이트 전후를 측정했다는 점이다. Support-positive 샘플은 이미 더 큰 critic weight를 받으므로 같은 샘플의 residual 감소가 일부 기계적으로 커질 수 있다. 이는 로컬 최적화 반응을 확인하지만 unseen transition 일반화를 직접 증명하지 않는다.

## 7. 진단 지표와 성능

Seed 7의 극단값을 제외하고 성공 9개 seed만 사용한 보수적 민감도 분석:

| 진단 지표 | AUC rho | AUC p | Hit90 rho | Hit90 p |
| --- | ---: | ---: | ---: | ---: |
| 평균 support | 0.383 | 0.309 | -0.383 | 0.309 |
| 마지막 400k support | -0.183 | 0.637 | 0.183 | 0.637 |
| TD 차이 | 0.267 | 0.488 | -0.267 | 0.488 |
| Residual 차이 | **0.667** | **0.0499** | **-0.667** | **0.0499** |
| Residual 개선률 차이 | 0.617 | 0.0769 | -0.617 | 0.0769 |
| Support–TD corr | -0.383 | 0.309 | 0.383 | 0.309 |
| Support–residual corr | 0.567 | 0.112 | -0.567 | 0.112 |

Residual 우위가 큰 seed일수록 AUC가 높고 Hit90이 빠른 경향이 가장 강했다. 그러나 n=9이고 다중 비교 보정 전 p-value가 경계에 있으므로 후속 검증 가설로 취급해야 한다.

## 8. Hit90 전후 변화

| 지표 | Hit90 이전 | Hit90 이후 | 변화 | 방향/검정 |
| --- | ---: | ---: | ---: | --- |
| Support 비율 | 0.2621 | 0.4700 | +0.2079 | 9/9 증가, `p=0.003906` |
| TD 차이 | 0.0401 | 0.0503 | +0.0102 | 5/9 증가, `p=0.164` |
| Residual 차이 | 0.00031 | 0.00586 | +0.00555 | 9/9 증가, `p=0.003906` |
| Residual 개선률 차이 | 0.00426 | 0.01025 | +0.00600 | 8/9 증가, `p=0.01172` |
| Support–TD corr | 0.1886 | 0.0562 | -0.1324 | 9/9 감소, `p=0.003906` |
| Support–residual corr | 0.0102 | 0.0225 | +0.0123 | 8/9 증가, `p=0.02734` |
| 손–손잡이 거리 | 0.4339 | 0.1094 | -0.3244 | 9/9 감소, `p=0.003906` |

초기에는 support가 높은 TD-error를 잘 구분하지만 즉각적인 residual 개선은 작다. 접촉과 성공 경험이 replay에 들어오면 support 집단이 확대되고 residual 개선이 커진다. 이는 B3가 학습을 돕는 동시에, 정책 진전이 다시 support 품질을 높이는 양방향 피드백을 시사한다. 현재 로그는 두 방향의 인과를 분리하지 못한다.

## 9. Phase별 메커니즘

### 9.1 성공 seed

| Phase | Batch 점유율 | Phase 내부 support+ |
| --- | ---: | ---: |
| Approach | 17.53% | 73.64% |
| Contact | 4.84% | 70.48% |
| Opening | 4.47% | 64.10% |
| Open | 73.16% | 30.46% |

Replay 대부분은 `open` phase지만 B3는 희소한 approach/contact/opening을 훨씬 자주 강조한다. 주요 기능은 단순 TD-error prioritization을 넘어 학습 진행에 필요한 희소 transition으로 critic 용량을 재배분하는 것으로 보인다.

### 9.2 `open` phase 주의

`phase_batch()`는 관측 handle 위치와 `desired_goal_train`의 거리를 사용한다. Batch의 약 79.4%는 HER relabeled goal이므로 `open`은 원래 환경 목표 성공과 동일하지 않다.

Seed 7은 실제 training/evaluation 성공이 모두 0이지만 batch의 약 79.6%가 `open`이었다. 따라서 다음 해석이 가장 타당하다.

> B3는 풍부하고 쉬운 HER relabel transition의 비중을 낮추고, 원래 조작 진행에 가까운 희소한 접근·접촉·열기 transition을 강조한다.

## 10. Seed 7 failure mode

Seed 7에는 NaN 폭발, CUDA 오류, OOM, 프로세스 실패가 없었다. 2M까지 정상 종료했지만 모든 training episode와 evaluation에서 성공률이 0이었다.

### 10.1 행동과 support의 시간 변화

| 구간 | 성공 support | S7 support | 성공 손–손잡이 거리 | S7 거리 | 성공 contact | S7 contact | 성공 opening | S7 opening |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5–400k | 0.239 | 0.277 | 0.444 | 0.452 | 0.25% | 0.12% | 0.06% | 0% |
| 400–800k | 0.392 | 0.286 | 0.265 | 0.557 | 3.48% | 0.03% | 2.93% | 0% |
| 800–1200k | 0.479 | 0.427 | 0.116 | 0.612 | 6.34% | 0% | 5.94% | 0% |
| 1200–1600k | 0.480 | 0.812 | 0.088 | 0.516 | 6.96% | 0% | 6.68% | 0% |
| 1600–2000k | 0.481 | 0.993 | 0.086 | 0.501 | 7.10% | 0% | 6.70% | 0% |

결정적 분기점은 400–800k다. 성공 seed는 이 구간부터 손–손잡이 거리가 감소하고 contact/opening 경험이 생긴다. Seed 7은 거리가 증가하고 접촉이 거의 없다.

### 10.2 Critic 신호의 시간 변화

| 구간 | 성공 TD error | S7 TD error | 성공 residual 차이 | S7 residual 차이 | 성공 critic loss | S7 critic loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5–400k | 0.1115 | 0.1017 | -0.00011 | +0.00089 | 0.0407 | 0.0303 |
| 400–800k | 0.2263 | 0.00653 | +0.00563 | -0.00001 | 0.2961 | 0.00042 |
| 800–1200k | 0.3623 | 0.00327 | +0.00553 | -0.00012 | 0.5284 | 0.00014 |
| 1200–1600k | 0.3661 | 0.00312 | +0.00511 | -0.00006 | 0.5391 | 0.00028 |
| 1600–2000k | 0.3910 | 0.00603 | +0.00567 | -0.00023 | 0.6108 | 0.00094 |

Seed 7의 critic loss와 TD error는 400k 이후 거의 0이 된다. 이는 발산이 아니라 진전 없는 replay 분포에 잘 맞는 상수형 해에 수렴했다는 뜻이다. Critic은 현재 데이터에는 정확하지만 정책은 손잡이에 접근하지 못한다.

### 10.3 Support 포화

- 1.2–1.6M 평균 support: `0.812`
- 1.6–2.0M 평균 support: `0.993`
- 마지막 400k에서 support >= 0.95인 스냅샷: `99.15%`
- 마지막 구간 approach support: `100%`
- 마지막 구간 open support: `99.1%`
- Contact/opening 데이터: 사실상 없음

모든 샘플이 support-positive면 raw weight가 모두 2가 되고 batch normalization 후 상대 가중치는 모두 1이 된다. B3의 우선순위가 사라져 균일 critic loss로 퇴화한다.

```text
초기 탐색 실패
  -> 손잡이 접촉 데이터 부재
  -> critic이 진전 없는 replay에 적합
  -> TD error와 학습 신호 소실
  -> replay/context 동질화
  -> support 판정 비구분화
  -> support 100% 포화
  -> B3 복구 신호 소실
```

Support 포화는 최초 원인보다 탐색 실패를 고착하는 후속 현상이다.

## 11. 구현과 실행 진단

### 11.1 Numba scanner

Launcher는 `--numba_support_scan`을 전달했지만 seeds 1–10 모두 다음을 기록했다.

```text
runtime/numba_scanner_warmed = 0
runtime/numba_support_scan_enabled = 0
runtime/audit_psi_A_policy_mismatch_rate = 0
```

기존 B3는 평균 23.33 steps/s, 이번 diagnostic run은 약 18.3 steps/s였다. 추가 diagnostic forward와 Numba fallback이 속도 저하 후보다.

Warmup/runtime 예외는 외부 로그 없이 scanner를 비활성화하므로 artifact만으로 원인을 확인할 수 없다. Audit mismatch 0은 Python fallback이 현재 reference와 같은 support 판정을 냈음을 뜻하므로 정확성 문제의 징후는 없다. 이는 과학적 결과보다 실행 효율과 관측 가능성 문제다.

### 11.2 Local scan truncation

- 성공 seed 평균 truncation: `97.97%`
- Seed 7 truncation: `98.93%`
- `local_search_max=512`

Audit mismatch 0은 같은 512-candidate 제한 아래의 구현 일치만 보장한다. 제한 없는 전체 replay 검색과의 동일성은 보장하지 않는다. 강한 정확성 주장을 위해 512/1024/full-scan label agreement가 필요하다.

## 12. 증거 수준

### 12.1 강하게 주장 가능한 내용

1. Support-positive는 support-negative보다 업데이트 전 TD error가 높다.
2. 방향은 성공 9개 seed 모두에서 일관된다.
3. B3는 HER-open보다 희소한 approach/contact/opening을 더 자주 강조한다.
4. Support-positive는 같은 weighted update 후 평균 residual 감소가 더 크다.
5. Seed 7은 초기 접촉 탐색 실패 후 critic 신호 소실과 support 포화로 실패했다.
6. B3는 유용한 접촉 경험이 전혀 없는 상황을 스스로 복구하는 탐색 알고리즘이 아니다.

### 12.2 현재 데이터만으로 확정하면 안 되는 내용

1. Same-batch residual 감소가 unseen transition 일반화를 증명한다.
2. Residual 차이가 AUC를 인과적으로 향상한다.
3. B3가 항상 90% seed 성공률을 보장한다.
4. Support 포화가 seed 7의 최초 원인이다.
5. 512-candidate support가 전체 replay support와 동일하다.
6. Numba fallback의 동등성을 모든 환경과 설정에 일반화한다.

## 13. 후속 검증 우선순위

동일 diagnostic 2M run을 반복할 필요는 없다. 다음 우선순위는 인과 분리와 실행 신뢰성이다.

### 1순위: Held-out residual

독립 audit batch에서 업데이트 전후 residual을 측정한다.

```text
batch A로 critic update
  -> 독립 batch B의 support+/support- residual 변화 측정
```

Same-batch의 기계적 in-sample 효과를 제거할 수 있다.

### 2순위: Seed 7 counterfactual

사전에 설정과 판정 기준을 고정하고 동일 예산으로 비교한다.

- B0
- B3 Critic-Only
- Random-Support
- Support 포화 시 uniform mixture 또는 상한을 적용한 B3

모두 실패하면 탐색 병목, B3만 실패하면 weighting 부작용, 포화 방지 B3만 복구하면 saturation lock-in 근거가 된다.

### 3순위: 원래 goal과 HER goal 분리

- Original/HER flag
- 두 집단별 phase
- Support-positive rate
- TD/residual enrichment

B3가 task-progress를 강조하는지 HER 구조만 구분하는지 직접 검증할 수 있다.

### 4순위: 동일 10-seed control

B0, Random-Support, PER-Style, 필요 시 Shuffled-Weight의 seeds 6–10을 추가해 B3와 동일 seed 범위로 비교한다.

### 5순위: Scanner 신뢰성

1. Numba warmup 예외를 로그에 남긴다.
2. 단일-process 10–50k smoke test에서 scanner warmup을 확인한다.
3. `numba_support_scan_enabled=1`과 audit mismatch 0을 함께 확인한다.
4. 일부 checkpoint에서 512/full-scan agreement를 측정한다.

## 14. 최종 판단

이번 diagnosis는 성능 결과와 내부 작동 원리가 서로 일치한다는 점에서 강하다. B3는 높은 TD error를 가진 희소한 진행 transition을 선별하고, 해당 샘플은 같은 업데이트에서 더 크게 교정된다. 대부분의 seed에서 이 과정은 빠르고 안정적인 학습과 함께 나타난다.

핵심 한계도 명확하다. 초기 exploration이 접촉 경험을 생성하지 못하면 critic이 진전 없는 replay에 적합되고, support가 포화되면서 B3의 구분력이 사라진다. 또한 same-batch residual은 일반화의 직접 증거가 아니다.

따라서 다음 단계는 동일 실험 반복보다 held-out residual, HER-origin 분리, 동일 10-seed control을 통해 인과성과 일반화를 강화하는 것이다.

## 15. 관련 파일

- 전체 요약: `metaworld_Drawer_Open_DDPG/README.md`
- Seeds 6–10 요약: `results_2m/DDPG-HER-B3-Critic-Only-2M/RESULTS_SUMMARY_S6_10_DIAGNOSTIC.md`
- Seeds 1–5: `results_2m_diag/DDPG-HER-B3-Critic-Only-2M-diagnostic/*/{summary.json,metrics.csv}`
- Seeds 6–10: `results_2m/DDPG-HER-B3-Critic-Only-2M/*_s{6..10}_*/{summary.json,metrics.csv}`
- Diagnostic 구현: `code/src/metacog_lite_v0/trainer.py`
- Support/weight 구현: `code/src/metacog_lite_v0/weights.py`
- Phase 정의: `code/src/metacog_lite_v0/task_drawer_open.py`

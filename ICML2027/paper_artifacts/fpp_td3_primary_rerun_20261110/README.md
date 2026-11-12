# FPP TD3 주 비교 재학습: 원 실행과의 비교

- 재학습 설정: [PROTOCOL_ko.md](../../experiments/fpp_td3_primary_checkpoint_rerun_20261110/PROTOCOL_ko.md)
- 실행: 2026-11-10 10:10–17:15, 10개 모두 완료, 실패 없음. HER는 실행당 약 3.1시간, SABER는 약 6.8시간.
- 원자료(`experiments/fpp_td3_primary_checkpoint_rerun_20261110/outputs/`)는 git에서 제외된다. 여기에는 요약만 둔다.

| 파일 | 내용 |
|---|---|
| `per_seed.csv` | 실행별 AUC, final-quarter AUC, 마지막 평가 성공률을 원 실행과 나란히 |
| `evaluation_curves.csv` | 10k마다의 평가 성공률. 재학습과 원 실행 |
| `launch_metadata.json` | 시작 시각, device, 코드 hash |

## 결과: 원 실행의 이득이 재현되지 않았다

같은 코드, 같은 명령, 같은 seed(1–5)다. 설정 파일은 device, 출력 경로, thread 키만 다르다.

| 지표 | 실행 | HER | SABER | SABER − HER | seed별 차이 | SABER가 높은 seed |
|---|---|---|---|---|---|---|
| AUC (10k–1M) | 원 실행 | 0.3716 | 0.4424 | **+0.0708** | +0.084, +0.084, +0.102, −0.004, +0.089 | 4 / 5 |
| | 재학습 | 0.3363 | 0.3406 | **+0.0043** | +0.034, +0.016, −0.018, −0.025, +0.014 | 3 / 5 |
| final-quarter AUC | 원 실행 | 0.6072 | 0.8040 | +0.1968 | | 4 / 5 |
| | 재학습 | 0.5268 | 0.5140 | −0.0128 | | 2 / 5 |
| 마지막 평가 성공률 | 원 실행 | 0.84 | 0.97 | +0.13 | | 3 / 5 |
| | 재학습 | 0.60 | 0.61 | +0.01 | | 2 / 5 |

- 재학습의 AUC 차이에 대한 짝 bootstrap 95% 구간은 [−0.015, +0.023]이다.
- 두 arm 모두 원 실행보다 낮다. HER는 5개 seed 모두 낮고(평균 −0.035), SABER는 5개 seed 모두 더 크게 낮다(평균 −0.102).

## 원 실행과 달랐던 것

| 항목 | 원 실행 | 재학습 |
|---|---|---|
| 기계 | 다른 기계(`/home/mimiclab/...`), 2026-06-02 ~ 03 | 이 PC |
| device | cuda | cpu, 실행당 1 thread |
| 패키지 버전 | 기록 없음 | torch 2.12.1, gymnasium 1.3.0, gymnasium-robotics 1.4.2, mujoco 3.3.0, numpy 2.3.5 |
| 환경 snapshot(`env_snapshot.json`) | – | 원 실행과 같음 |

기계와 device가 다르면 부동소수 결과가 달라져 궤적이 갈라진다. 그래서 같은 seed의 재실행도 사실상 새 표본이다. 이 차이가 우연인지, 기계·소프트웨어·device에서 오는 체계적 차이인지는 이 자료만으로 가를 수 없다.

## 저장소에 남은 FPP–TD3 기록을 기계별로 보면

| cohort | 기계 | device | HER AUC | SABER AUC | 차이 |
|---|---|---|---|---|---|
| 주 비교 seed 1–5 (6월) | 다른 기계 | cuda | 0.372 | 0.442 | +0.071 |
| 주 비교 seed 6–10 (10-seed 확장) | 다른 기계 | cuda | 0.394 | 0.416 | +0.023 |
| inclusive-future seed 1–5 (7월 27일) | 기록 없음 | cuda | 0.326 | 0.393 | +0.067 |
| inclusive-future seed 6–10 (7월 30일) | 이 PC | cuda | 0.327 | 0.339 | +0.011 |
| 주 비교 seed 1–5 재학습 (11월 10일) | 이 PC | cpu | 0.336 | 0.341 | +0.004 |

- 주 비교 규칙의 대조군(count-matched 0.391, context-count-only 0.396, query-replay 0.368, query-shuffled 0.347)은 모두 7월에 이 PC에서 cuda로 돌았다. 논문에서 이들과 비교한 SABER(0.442)는 다른 기계의 실행이다.
- 기계가 확인된 cohort만 보면, 이득이 뚜렷한 것은 다른 기계의 실행이고 이 PC의 두 cohort는 이득이 거의 없다. 다만 cohort마다 seed, HER 규칙, device가 함께 달라서 기계 효과라고 단정할 수 없다.

## 다음에 확인할 것

1. 다른 기계의 6월 환경(패키지 버전, GPU, 드라이버)을 찾아 이 PC와 비교한다.
2. 이 PC에서 cuda로 같은 실행을 일부 다시 돌려 device 효과를 분리한다. GPU를 쓰는 다른 작업과 조율해야 한다.
3. 가능하면 다른 기계에서 같은 코드를 다시 돌려 그쪽에서는 재현되는지 본다.

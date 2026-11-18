# FPP TD3 count-matched 대조군, seed 11–30 (2026-11-16 고정)

- 상태: **실행 전에 고정.** 분석 규칙은 `analyze.py`에 코드로 고정했다.
- 전체 계획에서의 위치: [논리 공백 로드맵](../../experment_design/2026-11-10_logic_gap_roadmap_ko.md)의 다음 단계 2번

## 1. 왜 하나

20쌍 확장에서 SABER > HER가 확인됐다(AUC +0.045 [+0.018, +0.073]). 그러나 이 결과만으로는 이득이 **어느 샘플을 올리느냐(support)** 에서 오는지, **같은 수의 샘플을 올리는 재가중 자체**에서 오는지 가를 수 없다. 리뷰 세 개가 공통으로 묻는 "왜 support인가"가 이 질문이다.

제출 논문의 count-matched 대조군(seed 1–5, AUC 0.391)은 SABER 원 실행(0.442)과 비교됐다. SABER가 20쌍 기준 0.416으로 내려오면서 격차가 0.02 수준으로 줄었다. 같은 seed 11–30에서 다시 비교한다.

## 2. arm

`experiment_FPP/td3_saber_context_eligible_count_matched_ls512`의 고정된 코드를 그대로 쓴다(freeze manifest 검사 통과).

- E: context-eligible mask (support 판정 전 단계의 조건을 통과한 샘플)
- M: SABER의 support-positive mask, k = |M|
- 대조군은 E 안에서 정확히 k개를 무작위로 골라 올린다. 같은 λ, 같은 평균 1 정규화, critic만.
- 즉 SABER와 **올리는 개수와 후보 집합이 같고, 고르는 기준만 다르다.**

## 3. 고정 조건

| 항목 | 값 |
|---|---|
| seed | 11–30. n20 HER·SABER와 같은 env·torch·numpy·noise seed. assignment seed는 30000 + seed |
| 길이, 저장본 | 1M step, 50k마다 저장 |
| device | cuda, 실행당 1 thread |
| 동시 실행 | 최대 10개. 정책 조회 측정(`fpp_n20`)을 함께 돌리고, LLM 서버가 공유 메모리를 쓰고 있어서다 |
| 출력 | `outputs/runs/` |
| 실패 처리 | 실패가 나면 새 실행을 시작하지 않는다. 재시도나 seed 교체는 하지 않는다 |
| 짝 | n20의 같은 seed HER·SABER 실행. 그 실행들은 이미 끝나 있고 AUC를 알고 있다 |

## 4. 분석 규칙 (`analyze.py`)

- 단위: seed. 통계는 n20 분석과 같은 코드(짝 bootstrap 95% 구간 100,000회, 난수 seed 20261113, 정확한 sign-flip p)를 쓴다.
- **주 대비: SABER − count-matched, AUC 10k–1M.**

| 결과 | 판정 |
|---|---|
| 95% 구간 하한 > 0 | SABER > count-matched. support 위치가 중요하다 |
| 95% 구간 상한 < 0 | count-matched > SABER |
| 구간이 0을 포함하고 상한 < +0.0225 (SABER 이득의 절반) | support 위치의 추가 이득은 SABER 효과의 절반도 안 된다. 이득의 대부분은 같은 수를 무작위로 올려도 나온다 |
| 구간이 0을 포함하고 상한 ≥ +0.0225 | 판단 보류 |

- 보조(판정 없음): count-matched − HER, 그리고 세 대비의 final-quarter AUC와 최종 성공률.
- 주의: 대조군이 SABER와 HER 사이에 있으면 두 대비 모두 0을 포함할 수 있다. 그 경우 "판단 보류"로 보고하고 효과 크기를 그대로 적는다.

## 5. 실행

저장소 root에서 실행한다.

```bash
.venv/bin/python ICML2027/experiments/fpp_td3_count_matched_n20_20261116/launch.py run --device cuda --threads 1 --max-parallel 10
```

```bash
.venv/bin/python ICML2027/experiments/fpp_td3_count_matched_n20_20261116/launch.py status
```

```bash
.venv/bin/python ICML2027/experiments/fpp_td3_count_matched_n20_20261116/analyze.py
```

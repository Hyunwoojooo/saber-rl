# FPP TD3 context-count-only 대조군, seed 11–30 (2026-11-17 고정)

- 상태: **실행 전에 고정.** 분석 규칙은 `analyze.py`에 코드로 고정했다.
- 전체 계획에서의 위치: [논리 공백 로드맵](../../experment_design/2026-11-10_logic_gap_roadmap_ko.md)
- 앞선 대조군: [count-matched n20](../fpp_td3_count_matched_n20_20261116/PROTOCOL_ko.md) — SABER > count-matched, AUC +0.051 [+0.015, +0.086]

## 1. 왜 하나

count-matched 결과로 "같은 수를 E 안에서 무작위로 올리면 이득이 없다"가 확인됐다. 이번 대조군은 SABER 판정에서 **a^π 조회(정책 행동의 bin이 국소 support에 있는가)를 빼고 context 개수 조건만 남긴다.** 이득이 정책 행동 조회에서 오는지, context 빈도 조건만으로도 나오는지 가른다. 제출 논문의 query ablation(seed 1–5)에 있던 대조군을 n20 seed로 다시 돈다.

## 2. arm

`experiment_FPP/td3_saber_query_ablation_ls512`의 `--support_query_mode context_count_only`. 그 코드의 current_policy 모드는 SABER 코드와 같은 가중치와 난수 상태를 낸다(`test_current_policy_legacy_parity.py`, Python·Numba 모두 2026-11-17 통과). 이 패키지는 freeze manifest가 없어서 실행 전에 이 parity 테스트와 `test_support_query_modes.py`, `test_run_config_parity.py`를 돌렸다.

- ψ = 1 ⇔ context 개수 조건 통과. a^π bin 조건은 없다.
- 같은 λ = 1, 같은 평균 1 정규화, critic만.
- **주의:** 7월 실행에서 양성 비율이 약 0.96, 정규화 가중치 ESS가 253/256이었다. 즉 이 대조군은 거의 균등 가중이고, 바뀌는 것은 context 조건을 통과하지 못한 약 4%가 상대적으로 절반 가중이 되는 것뿐이다. 그래서 HER와 비슷하게 나올 것으로 예상하며, "개수 조건만으로 이득이 나오는가"라는 질문에 대한 답이다.

## 3. 고정 조건

| 항목 | 값 |
|---|---|
| seed | 11–30. n20 HER·SABER와 같은 env·torch·numpy·noise seed. weight seed 20000 + seed |
| 인자 | 7월 seed 1 실행의 config와 seed·출력 경로 외 56개 인자 일치 확인 |
| 길이, 저장본 | 1M step, 50k마다 저장 |
| device | cuda, 실행당 1 thread |
| 동시 실행 | 최대 10개 |
| 출력 | `outputs/runs/` |
| 실패 처리 | 실패가 나면 새 실행을 시작하지 않는다. 재시도나 seed 교체는 하지 않는다 |
| 짝 | n20의 같은 seed HER·SABER 실행 |

## 4. 분석 규칙 (`analyze.py`)

- 단위: seed. 통계는 n20 분석과 같은 코드(짝 bootstrap 95% 구간 100,000회, 난수 seed 20261113, 정확한 sign-flip p).
- **주 대비: SABER − context-count-only, AUC 10k–1M.**

| 결과 | 판정 |
|---|---|
| 95% 구간 하한 > 0 | SABER > context-count-only. a^π 조회가 context 조건에 더해 기여한다 |
| 95% 구간 상한 < 0 | context-count-only > SABER |
| 구간이 0을 포함하고 상한 < +0.0225 (SABER 이득의 절반) | a^π 조회의 추가 이득은 SABER 효과의 절반도 안 된다 |
| 구간이 0을 포함하고 상한 ≥ +0.0225 | 판단 보류 |

- 보조(판정 없음): context-count-only − HER, 그리고 세 대비의 final-quarter AUC와 최종 성공률.

## 5. 실행

저장소 root에서 실행한다.

```bash
.venv/bin/python ICML2027/experiments/fpp_td3_context_count_only_n20_20261117/launch.py run --device cuda --threads 1 --max-parallel 10
.venv/bin/python ICML2027/experiments/fpp_td3_context_count_only_n20_20261117/launch.py status
.venv/bin/python ICML2027/experiments/fpp_td3_context_count_only_n20_20261117/analyze.py
```

# FPP TD3 주 비교 재학습 protocol (2026-11-10 고정)

- 목적: 제출 논문의 FetchPickAndPlace–TD3 주 비교(HER 대 SABER, seed 1–5)를 **저장본을 남기며** 다시 학습한다. 원 실행은 요약 파일만 남아 있어 [실험 4의 측정](../draweropen_ddpg_first_order_usefulness_20261109/PROTOCOL_fpp_ko.md)을 적용할 수 없다.
- 전체 계획에서의 위치: [논리 공백 로드맵](../../experment_design/2026-11-10_logic_gap_roadmap_ko.md) 2번의 선택지 C
- 이것은 새 방법이나 새 설정의 실험이 아니다. 같은 코드, 같은 명령, 같은 seed를 다시 돌린다.

## 고정 조건

| 항목 | 값 |
|---|---|
| 코드 | `experiment_FPP/td3_her_baseline_ls512/code/src`, `experiment_FPP/td3_saber_critic_only_v4_ls512/code/src`. 수정하지 않는다. 두 폴더는 `config.py`의 `as_dict` 세 줄만 다르다 |
| 명령 | 각 폴더의 `run.py`가 만드는 것과 같은 인자, 같은 순서(`launch.py`의 `build_command`). 설정값은 각 폴더의 `config.json`에서 읽는다 |
| arm | HER(`td3_her`), SABER(`td3_saber_runtime_opt`, critic-only, λ = 1) |
| seed | 1–5. env·torch·numpy·action noise seed = s, weight seed = 20000 + s, eval seed = 10000 + s |
| 길이 | 1,000,000 step. 평가는 10k마다 20 episodes |
| 저장본 | 50k마다(코드 기본값). 실행당 20개 |
| 출력 | 이 폴더의 `outputs/runs/<run_slug>/<run_name>/`. 원 실행 폴더는 건드리지 않는다. 이미 있는 실행 폴더는 덮어쓰지 않는다 |

## 원 실행과 달라지는 점

| 항목 | 원 실행 (2026-06) | 이번 |
|---|---|---|
| 기계 | 다른 기계(`/home/mimiclab/...`) | 이 PC |
| device | cuda | **cpu**, 실행당 1 thread |
| 실행 방식 | seed를 차례로 | 10개를 동시에 |

- cpu를 쓰는 이유: 이 PC의 GPU는 다른 작업이 쓰는 LLM 서버(vLLM, 76GB)가 점유하고 있다. 측정해 보니 속도 차이가 작다(60 대 67 env step/s, 9k step 벤치마크).
- 기계와 device가 달라 부동소수 연산 결과가 다르므로 **원 실행과 같은 궤적을 재현하지 않는다.** 같은 seed의 독립 재실행으로 본다.
- 실패한 실행이 있으면 새 실행을 시작하지 않고 멈춘다. 자동 재시도나 seed 교체는 하지 않는다.

## 끝난 뒤 보고할 것

1. 실행별 AUC와 원 실행 AUC(`launch.py status`가 나란히 출력한다). 원 실행의 평균은 HER 0.372, SABER 0.442, 차이 +0.070이다.
2. 재학습의 SABER − HER 차이(seed 짝). 이 값이 원 실행과 크게 다르면 그 사실을 측정 결과 해석 앞에 적는다.
3. 설정 일치: 새 `config.json`과 원 실행 `config.json`에서 값이 다른 키. 출력 경로, device, thread 관련 키만 달라야 한다.
4. 실험 4의 측정(`--set fpp_primary`). 판정 규칙은 [PROTOCOL_fpp_ko.md](../draweropen_ddpg_first_order_usefulness_20261109/PROTOCOL_fpp_ko.md)에 이미 고정했다.

## 실행

```bash
.venv/bin/python ICML2027/experiments/fpp_td3_primary_checkpoint_rerun_20261110/launch.py run --device cpu --threads 1 --max-parallel 10
```

```bash
.venv/bin/python ICML2027/experiments/fpp_td3_primary_checkpoint_rerun_20261110/launch.py status
```

---

# 추가: GPU 재실행 (2026-11-12 고정)

## 왜 하나

11월 10일의 cpu 재학습에서는 원 실행의 이득이 재현되지 않았다(SABER − HER +0.071 → +0.004). [비교 자료](../../paper_artifacts/fpp_td3_primary_rerun_20261110/README.md). 원 실행과 달랐던 것은 기계와 device 두 가지다. 이 가운데 **device만 원 실행과 같게(cuda) 맞춰** 다시 돌린다.

- 원 실행의 기계: 설정 파일에 출력 경로가 `/home/mimiclab/Hyunwoo/metacog-rl-v0-lite/...`로 적혀 있다. 이 PC에는 `/home/mimiclab` 계정이 없다(이 PC는 `/home/mimiclab-gigabyte-gen5`). 저장소의 실행 기록에는 이 밖에 `mimiclab-gigabyte-gen4`, `mimiclab_asus` 경로도 있다.
- cpu 재학습의 평가 곡선은 처음 3–17개 평가 시점(30k–180k)까지 원 실행과 값이 같고 그 뒤 갈라진다. 환경과 seed는 같게 재현되고, 부동소수 차이로 궤적이 갈라진다는 뜻이다.

## 고정 조건

11월 10일 재학습과 device만 다르다.

| 항목 | 값 |
|---|---|
| 코드, 명령, arm, seed, 길이, 저장본 | 위와 같음(HER·SABER × seed 1–5 × 1M, 50k마다 저장) |
| device | **cuda** |
| thread | 실행당 1 (11월 10일과 같음) |
| 출력 | `outputs/runs_cuda/`. 태그 `cuda` |

## 읽는 규칙 (결과를 보기 전에 정함)

짝(seed)별 SABER − HER AUC 차이 5개로 본다.

| 결과 | 읽기 | 다음 |
|---|---|---|
| 평균 ≥ +0.035(원 실행의 절반)이고 5개 중 4개 이상이 양수 | 이 PC에서도 cuda로는 이득이 재현된다. device가 결과를 가른다 | 이 저장본으로 실험 4를 측정한다. 이득이 있는 cohort의 측정이 처음으로 가능해진다 |
| 평균 ≤ +0.020 | cuda로도 재현되지 않는다. device는 원인이 아니다 | 남는 설명은 원 실행 기계의 환경 차이 또는 우연이다. seed를 늘린 재실행과 원 기계 환경 확인으로 간다 |
| 그 사이 | 판단 보류 | seed 6–10을 같은 조건으로 추가한다 |

함께 보고할 것: 실행별 AUC(원 실행, cpu 재학습과 나란히), 원 실행과 평가 곡선이 처음 갈라지는 시점, 마지막 1/4 구간 AUC, 마지막 평가 성공률.

## 실행

```bash
.venv/bin/python ICML2027/experiments/fpp_td3_primary_checkpoint_rerun_20261110/launch.py run --device cuda --threads 1 --max-parallel 10 --tag cuda
```

```bash
.venv/bin/python ICML2027/experiments/fpp_td3_primary_checkpoint_rerun_20261110/launch.py status --tag cuda
```

# 1단계: 과제 × 알고리즘별 HER·SABER 새 20쌍 (2026-11-18 고정)

- 상태: **실행 전에 고정.** 분석 규칙은 `analyze.py`에 코드로 고정했다.
- 앞 단계: [0단계 정리](../../paper_artifacts/stage0_cohort_inventory_20261118/README_ko.md). 20쌍 이상으로 이득이 확립된 조건은 FPP TD3 하나다.
- 전체 계획에서의 위치: [논리 공백 로드맵](../../experment_design/2026-11-10_logic_gap_roadmap_ko.md)

## 1. 왜 하나

기존 짝은 FPP TD3를 빼면 조건마다 5–10개다. 그중 FPP TD3는 5쌍 결과(+0.071)가 20쌍에서 +0.045로 바뀌었다. 다른 조건도 같은 방식, 즉 새 seed 11–30, 이 PC, 같은 규칙으로 다시 잰다. 이 결과로 논문이 주장할 수 있는 과제·알고리즘 범위를 정한다.

## 2. 이번 묶음 (첫 번째)

| 조건 | HER 패키지 | SABER 패키지 | 길이 |
|---|---|---|---|
| FPP DDPG | `experiment_FPP/b0_same_env_v4` | `experiment_FPP/b3_critic_only_runtime_opt_v4_ls512` | 1M |
| FPP SAC | `experiment_FPP/sac_her_baseline_ls512` | `experiment_FPP/sac_saber_critic_only_v4_ls512` | 1M |

Drawer 조건은 이 PC에서 실행 시간을 측정한 뒤 다음 묶음으로 같은 규칙을 적용해 추가한다.

**패키지에 관한 기록**
- 각 arm은 패키지 자신의 `run.py` `build_command`가 만든 명령을 그대로 쓴다. 메모리 안에서 바꾸는 것은 두 가지뿐이다: 허용 seed 목록(1–5 → 11–30 추가), 그리고 옛 저장소 구조를 찾는 `repo_root_from`. 후자는 python 경로와 기본 출력 경로에만 쓰이고, 출력 경로는 이 폴더로 바꾼다.
- **FPP SAC:** 두 패키지의 코드가 같다(sha256 일치). 명령 인자 53개가 6월 seed 1 실행의 config와 같다(seed·출력 경로·장치 제외).
- **FPP DDPG HER:** 논문의 DDPG HER(`experiment_FPP/runs/B0`, 5월)는 지금 저장소에 없는 옛 코드(`version_v0_lite`)로 돌았다. 대신 `b0_same_env_v4`를 쓴다. 이 패키지는 SABER DDPG 패키지와 같은 코드 트리이고, 다른 점은 weights.py의 로그 항목과 method(`b0_ddpg_her`)·λ = 0·weight_scope = none뿐이다. 그래서 이번 짝은 같은 코드 트리 안의 비교가 되지만, 논문의 seed 1–5 HER와는 코드가 다르다.
- **FPP DDPG SABER:** 5월 실행의 config(키 117개)에는 지금 명령이 넘기는 인자 3개(`numba_support_scan`, `support_predicate_early_exit`, `support_diagnostic_audit_interval`)가 없다. 그래서 5월 코드와 지금 패키지 코드가 완전히 같은지는 확인할 수 없다. 앞의 두 인자는 parity 테스트가 있는 정확한 가속 경로이고, 세 번째(감사)는 진단 값만 센다(코드 확인).

## 3. 고정 조건

| 항목 | 값 |
|---|---|
| seed | 11–30. env·torch·numpy·noise seed = seed, weight seed = 20000 + seed, eval seed = 10000 + seed (패키지 규칙) |
| device | cuda, 실행당 1 thread |
| 동시 실행 | 기계 전체 학습 프로세스 14개 이하, MemAvailable ≥ 8 GB일 때만 새 실행 시작. 긴 실행부터 |
| 출력 | `outputs/runs/<조건>/<arm>/` |
| 실패 처리 | 실패가 나면 그 실행기는 새 실행을 시작하지 않는다. 재시도나 seed 교체는 하지 않는다 |

## 4. 분석 규칙 (`analyze.py`)

- 단위: seed. AUC와 통계는 FPP TD3 n20 분석과 같은 코드를 쓴다. AUC는 `eval/success_mean`을 [10k, 끝] 구간에서 사다리꼴 적분하고, 짝 bootstrap 95% 구간은 100,000회(난수 seed 20261113), sign-flip p는 정확 계산이다.
- **주 대비 (조건마다): SABER − HER, AUC.**

| 결과 | 판정 |
|---|---|
| 95% 구간 하한 > 0 | SABER > HER |
| 95% 구간 상한 < 0 | HER > SABER |
| 구간이 0을 포함 | 판단 보류. 효과 크기와 구간을 그대로 보고 |

- 보조(판정 없음): final-quarter AUC, 최종 성공률.
- 조건 사이의 다중 비교 보정은 하지 않는다. 조건마다 따로 보고하고, 조건 수를 함께 적는다.
- 기존 seed 1–10과의 합산은 서술로만 보고한다. 기기·시기가 다르고, FPP DDPG는 HER 코드도 다르기 때문이다.

## 5. 실행

저장소 root에서 실행한다.

```bash
.venv/bin/python ICML2027/experiments/stage1_n20_extension_20261118/launch.py run --cells fpp_ddpg fpp_sac --max-total 14
.venv/bin/python ICML2027/experiments/stage1_n20_extension_20261118/launch.py status
.venv/bin/python ICML2027/experiments/stage1_n20_extension_20261118/analyze.py --cells fpp_ddpg fpp_sac
```

## 6. 실행 중 변경 기록

- **2026-11-19 14:12 KST, 동시 실행 상한 14 → 18.** 사용자가 이 PC에서 쓰이지 않던 vLLM 컨테이너(`qwen38-27b-vllm`, 10월 12일 이후 요청 0건)를 내리기로 했고, 그 결과 가용 메모리가 약 9 GB에서 약 85 GB로 늘었다. 상한은 CPU 20코어에 맞춰 18로 정했다.
  - 이때 완료 14개(FPP SAC s11–17 × 2 arm)와 실행 중 14개는 손대지 않았다. 기존 실행기만 종료했고, 새 실행기(`--adopt-existing`)가 그 28개를 넘겨받아 지켜보기만 하며 나머지 52개를 띄운다.
  - 바뀌는 것은 실행 시간뿐이다. 명령, seed, 분석 규칙은 그대로다.
  - 기존 실행기의 상태 파일은 `outputs/superseded_launcher_status_fpp_ddpg_fpp_sac.json`, 새 실행기의 상태 파일은 `launcher_status_fpp_ddpg_fpp_sac_adopt.json`이다. 실행이 모두 끝나면 `DONE_fpp_ddpg_fpp_sac_adopt` 또는 `FAILED_fpp_ddpg_fpp_sac_adopt` 파일이 생긴다.

## 7. 두 번째 묶음: Drawer 조건 3개 (2026-11-22, Drawer 1단계 실행 시작 전에 고정)

첫 묶음 결과(FPP SAC +0.023, FPP DDPG −0.015, 둘 다 판단 보류)를 본 뒤 고정했다. 0단계에서 흩어짐이 작아 20쌍으로 결론이 날 것으로 본 조건부터 돌린다. DrawerOpen DDPG·TD3, DrawerClose TD3는 흩어짐이 커서 나중에 따로 정한다.

| 조건 | 코드 (origin/main 5facb92에서 추출, `sources/`) | 기준으로 삼은 기존 실행 | 길이 |
|---|---|---|---|
| DrawerOpen SAC | `ICML_27/experiments/draweropen_drawerclose_sac_her_vs_saber/source` → `sources/drawer_sac` | seed 6 확장 실행의 launch_record (`templates/draweropen_sac_*_seed6_launch_record.json`) | 2M |
| DrawerClose SAC | 위와 같은 코드 | seed 1 실행의 config (`templates/drawerclose_sac_*_seed1_config.json`) | 2M |
| DrawerClose DDPG | `drawer_close_ddpg_her_saber_seed_extension_2m/source_snapshot` → `sources/drawerclose_ddpg/<arm>` | seed 6 확장 실행의 argv (`templates/drawerclose_ddpg_*_seed6_launch_record.json`) | 2M |

**실행 방식**
- **Drawer SAC:** `train_stage1_drawer_sac_lite_v0.py`가 그 코드의 `protocol.make_config`로 설정을 만든다. 허용 seed 목록만 메모리 안에서 넓힌다.
  - 학습 전에 설정 전체를 기준 실행의 기록과 대조한다. seed, run_name, output_root, experiment_id 말고 다른 값이 하나라도 다르면 중단한다.
  - 2026-11-22에 대조해 보니 차이는 위 네 가지와 JSON 형식(tuple/list), 학습 중에 기록되는 파생 값뿐이었다.
  - 학습 경로는 원래 `train.py`의 production 경로와 같다(thread 1개 고정, `run_training`, `validate_run`). 다만 seed 1–5 전용 production gate와 claim 폴더는 쓰지 않는다.
- **DrawerClose DDPG:** 기록된 argv를 그대로 쓰고 다음 값만 바꾼다: seed 6개(env·torch·numpy·noise = seed, weight = 20000 + seed, eval = 10000 + seed), `--run_name seed_XX`, `--output_root`.
  - `source_snapshot`은 이 저장소의 `metaworld_Drawer_Close_DDPG/code/src`와 같다(테스트 파일 하나만 빠짐).
  - HER와 SABER 코드 트리도 서로 같다.
- seed, 장치, 동시 실행, 실패 처리는 3절과 같다. 동시 실행 상한은 18이다(6절).

**분석 (`analyze.py`, 같은 규칙):** `metrics.csv`의 `eval/success_mean`을 쓴다. Drawer 코드에서 이 값은 평가 episode 중 한 번이라도 성공했는지(env success ever)이고, 0단계와 기존 Drawer 결과도 같은 정의다. AUC 구간은 [10k, 2M], final-quarter는 [1.5M, 2M], 최종은 2M 시점의 값이다. 판정 규칙은 4절과 같다.

**알려진 점**
- **Drawer SAC 관찰값:** 관찰 벡터의 마지막 세 성분에 원래 목표가 들어 있다(기존 README에 기록된 사항). 두 arm이 같은 구현을 쓰므로 그대로 둔다.
- **DrawerClose DDPG 과거 기록:** 과거 seed 1–5의 정확한 config는 남아 있지 않다. 그래서 seed 6–10 확장 실행을 기준으로 삼았다.
- **예상 실행 시간:** 기존 기록 기준으로 SAC는 HER 18시간·SABER 27시간, DrawerClose DDPG는 HER 11.4시간·SABER 19.5시간이다. 이 PC에서 18개를 동시에 돌리면 약 1.3배 느려진다고 보고, 전체 약 3,100 실행-시간, 약 8일로 예상한다.

## 8. 두 번째 묶음 변경: SAC 두 조건 보류 (2026-11-25, 중단 전에 기록)

**결정:** DrawerOpen SAC와 DrawerClose SAC의 20쌍 실행을 보류한다. DrawerClose DDPG는 그대로 20쌍을 돌린다. 대신 DrawerOpen TD3 20쌍(9절)과 FPP TD3의 Q 수준 검정(`../fpp_td3_target_pessimism_20261125/`)을 먼저 돌린다.

**결정 시점의 상태 (2026-11-25 07:11 UTC)**
- SAC SABER seed 11–19는 두 조건 모두 끝났다(18개). seed 20–28은 2M 중 7만–27만 step까지 진행됐다(18개). seed 29–30과 SAC HER 40개, DrawerClose DDPG 40개는 시작하지 않았다.
- 새 seed의 HER 실행은 하나도 없다. 따라서 이 세 조건에서 새 seed의 SABER − HER 차이는 관찰된 적이 없다. 끝난 SABER 18개의 성능도 집계하지 않았다.

**사유 (모두 새 seed의 결과와 무관하다)**
1. **비용.** SAC SABER 한 개가 예상 35시간이 아니라 59–64시간 걸렸다. 남은 SAC 실행만 장비 전체로 약 7일이다.
2. **우선순위.** 0단계 값으로 보면 DrawerOpen SAC는 10쌍에서 +0.008 [−0.059, +0.076]이고 DrawerClose SAC는 5쌍에서 −0.157이다. 20쌍 결과가 어느 쪽이어도 논문의 주장 범위를 넓히지 못한다. 0단계에서 추정 이득이 20쌍 예상 구간 반폭을 넘고 seed 6–10에서도 유지된 조건은 DrawerOpen TD3 하나다(10쌍 +0.321, seed 6–10만 +0.406, 예상 반폭 ±0.21).
3. **7절 선정 기준의 정정.** 7절은 "흩어짐이 작은 조건부터"로 골랐다. 판정이 날 가능성은 흩어짐이 아니라 흩어짐 대비 이득 크기로 정해진다. 이 기준으로 순서를 다시 정했다.

**처리**
- 끝난 SAC SABER 18개(`outputs/runs/{draweropen,drawerclose}_sac/saber/*_s11..s19_2m`)는 그대로 둔다. 나중에 SAC를 이어 돌리면 이 실행을 그대로 쓴다.
- 진행 중이던 seed 20–28의 18개는 중단하고, 폴더와 로그를 `outputs/aborted_20261125_sac_saber_partial/`로 옮긴다. 분석에 쓰지 않는다. 이어 돌릴 때는 처음부터 다시 돌린다.
- 기존 launcher 상태 파일은 `superseded_` 접두어를 붙여 남긴다.
- SAC를 이어 돌릴 때의 규칙(seed 11–30, `analyze.py`, 4절 판정)은 바뀌지 않는다. 그 전까지 이 두 조건의 서술은 0단계 값(10쌍, 5쌍)에 근거한다.

## 9. DrawerOpen TD3 추가 (2026-11-25, 이 조건의 1단계 실행 시작 전에 고정)

| 조건 | 코드 (origin/main 76753b2에서 추출, `sources/draweropen_td3/<arm>`) | 기준으로 삼은 기존 실행 | 길이 |
|---|---|---|---|
| DrawerOpen TD3 | `metaworld_Drawer_Open_TD3/paper_artifacts/drawer_open_td3_main_seed_extension_2m/source_snapshot` | seed 6 확장 실행의 argv (`templates/draweropen_td3_*_seed6_launch_record.json`) | 2M |

**실행 방식**
- DrawerClose DDPG와 같다. 기록된 argv를 그대로 쓰고 seed 6개, `--run_name seed_XX`, `--output_root`만 바꾼다. HER와 SABER 코드 트리는 서로 같다.
- 한 가지를 더 바꿨다. `--no_save_step_checkpoints`를 뺐다. 50k step마다 checkpoint 사본을 남겨 나중에 critic을 측정할 수 있게 하기 위해서다. 파일을 더 쓸 뿐 학습 계산은 같다.
- 2026-11-25에 seed 11, 12,000 step, cpu로 두 arm을 짧게 돌려 `config.json`을 main의 seed 6 기록과 대조했다. 132개 항목 중 다른 것은 seed, run_name, output_root, save_step_checkpoints와 시험용으로 바꾼 device·total_steps뿐이었다.
- seed 11–30, cuda, thread 1개, 동시 실행 상한 18. 실패 처리는 3절과 같다.

**분석:** `analyze.py --cells draweropen_td3`. 지표와 판정 규칙은 4절·7절과 같다(AUC 구간 [10k, 2M]).

**미리 적어 두는 점**
- 이 조건의 실행은 대부분 끝까지 실패하거나(AUC 0) 50만–90만 step에 풀린다(AUC 0.6–0.76). seed 1–10에서 풀린 실행은 HER 2/10, SABER 7/10이다. 그래서 AUC 차이는 사실상 풀린 비율의 차이다.
- 같은 seed 1–10에서 PER-full은 6/10, HGR은 4/10이 풀렸다. 이 실행은 HER와 SABER만 비교하므로, SABER > HER로 판정되더라도 SABER가 다른 가중 방법보다 낫다는 뜻은 아니다.
- 기존 기록의 실행 시간은 HER 11.2시간(5개 동시)이다. 이 PC에서 18개를 동시에 돌릴 때의 시간은 재지 않았다.

**실행 순서 (2026-11-25):** FPP TD3 target 검정(`../fpp_td3_target_pessimism_20261125/`) → DrawerOpen TD3 → DrawerClose DDPG. 앞 묶음에 새로 시작할 실행이 없어지면 다음 묶음이 빈자리를 채운다(`launch.py run --after`).

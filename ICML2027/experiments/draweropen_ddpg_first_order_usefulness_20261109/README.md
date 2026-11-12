# 실험 4 구현: SABER 가중의 1차 유용성 측정

설계는 [04_first_order_usefulness/README_ko.md](../../experment_design/04_first_order_usefulness/README_ko.md)에 있다. 이 폴더는 그 설계를 저장된 checkpoint에 적용하는 코드다. 새 학습은 하지 않는다. checkpoint를 읽어 critic만 복제해 분기한다.

| checkpoint set | 구간 | 대상 | 상태 |
|---|---|---|---|
| `a1` | 초반부 10k–500k | A1의 4 arm × seed 101–103 × 8개 시점 | SABER·HER 48개 측정 완료(2026-11-09). [결과 요약](../../paper_artifacts/first_order_usefulness_20261109/early_phase_a1/REPORT_ko.md) |
| `ext2m` | 2M | assignment extension의 SABER·Random·TD × seed 11–20 | 30개 측정 완료(2026-11-10). [결과 요약](../../paper_artifacts/first_order_usefulness_20261109/late_ext2m/REPORT_ko.md) |
| `mature` | 500k–2M | 성숙 정책 pilot의 SABER·HER seed 1 × 4개 시점 | 8개 측정 완료(2026-11-10). [결과 요약](../../paper_artifacts/first_order_usefulness_20261109/late_mature/REPORT_ko.md) |

| `fpp_incl` | 50k–1M | FetchPickAndPlace TD3, inclusive-future HER·SABER × seed 6–10 × 9개 시점 | 90개 측정 완료(2026-11-10). [protocol](PROTOCOL_fpp_ko.md), [결과 요약](../../paper_artifacts/first_order_usefulness_20261109/fpp_incl/REPORT_ko.md) |
| `fpp_primary` | 50k–1M | FetchPickAndPlace TD3, 주 비교 규칙 HER·SABER × seed 1–5 × 9개 시점 | 90개 측정 완료(2026-11-10). [재학습](../fpp_td3_primary_checkpoint_rerun_20261110/PROTOCOL_ko.md), [결과 요약](../../paper_artifacts/first_order_usefulness_20261109/fpp_primary/REPORT_ko.md) |

폴더 이름은 DrawerOpen이지만 코드는 두 계열을 다룬다. `drawer_ddpg`(critic 1개)와 `fpp_td3`(critic 2개, target smoothing)가 같은 측정 경로를 쓴다. 정책 조회 오차 R과 κ는 actor가 읽는 critic(head 0) 기준이다. FPP 쪽은 그 실행을 학습시킨 코드 폴더(`experiment_FPP/.../code/src`)를 별칭으로 import해서 replay·HER 규칙·support 판정을 그대로 쓴다.

후반부(`ext2m`, `mature`)의 대상, 판정 규칙, 실행 명령은 [PROTOCOL_late_ko.md](PROTOCOL_late_ko.md)에 있다. 전체 계획에서의 위치는 [논리 공백 로드맵](../../experment_design/2026-11-10_logic_gap_roadmap_ko.md)을 본다.

## 무엇을 재나

checkpoint 하나, 고정 minibatch 하나에서 다음을 비교한다.

| arm | 높은 가중치 위치 |
|---|---|
| uniform | 없음. HER와 같음 |
| saber | ψ = 1 (제출 코드의 support 판정) |
| inverse | ψ = 0 |
| random | eligible E 안에서 \|M\|개를 무작위로 |
| td_topk | E 안에서 \|δ\|가 큰 \|M\|개 |

비교 기준은 정책 조회 오차 R다.
- **R1**: 노이즈 없는 정책 실행, 원래 goal, 고정 target 1-step Bellman 오차
- **R2**: snapshot 복원, HER goal, a^π의 1-step Bellman 오차

`gain = R(uniform 갱신 후) − R(arm 갱신 후)`이고, 양수면 그 arm이 더 낫다.

| 열 이름 | 계산 |
|---|---|
| `G1` | SGD 1차 예측 ⟨∇R, g_arm − g_uniform⟩ |
| `G2` | clip 10을 적용한 gradient 방향 기준 1차 예측 |
| `G3` | 저장된 Adam moment + clip을 반영한 갱신(float64)의 1차 예측 |
| `G3mid` | G3과 같고, ∇R을 두 갱신의 중점에서 계산 |
| `A1` | 실제 torch Adam step 1회 후의 R 차이 |
| `K{1,10,100}` | 실제 step K회. actor와 target은 고정, 배치 순서는 arm 간 공유 |

분해 지표(`per_sample.npz`)는 샘플별로 u, cos(∇ℓ, ∇R), κ = cos(∇Q(a^π), ∇Q(a^r)), ‖a^r − a^π‖, |δ|, 자기 support 여부, phase를 담는다. 2026-11-10부터는 replay slot과 나이(`age`, 저장된 뒤 지난 env step 수)도 담는다.

## 파일

| 파일 | 역할 |
|---|---|
| `fou/core.py` | checkpoint 로딩, 제출 코드와 parity를 확인한 support mask, R1/R2 query bank, u·G1–G3·실제 step, K-step branch |
| `fou/protocol.py` | 사전 고정한 checkpoint set·시점·seed·배치 수·난수 seed·판정 규칙 |
| `PROTOCOL_late_ko.md` | 후반부 측정의 대상과 판정 규칙. 측정 전에 고정 |
| `fou/vendor_sim.py` | `~/ICML2027/draweropen_ddpg_policy_query/saber_qdiag/simulator.py`의 감사된 snapshot adapter 사본. 원본 sha256은 파일 머리에 있다 |
| `run.py` | `plan`, `measure` |
| `validate.py` | 측정 전에 통과해야 하는 검사. `--set a1`은 25개, `--set late`는 63개, `--set fpp`는 43개 |
| `aggregate.py` | checkpoint 요약, 가설 판정, `REPORT_ko.md` |

모델·replay·task 클래스는 A1의 `code/src/metacog_lite_v0`를 그대로 import한다. A1 폴더는 수정하지 않는다.

## 실행

저장소 root 기준이며 `.venv`를 쓴다.

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/validate.py --out outputs/validation_<날짜>
```

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/run.py measure --out outputs/main --arms saber her --seeds 101 102 103
```

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/aggregate.py --out outputs/main
```

- `--out`의 상대 경로는 이 폴더 기준이다.
- 이미 측정한 checkpoint는 건너뛴다. checkpoint 폴더는 측정이 모두 끝난 뒤에만 만들어지므로 중간에 멈춰도 다시 실행하면 된다.
- 분석 폴더는 덮어쓰지 않는다. 다시 집계하려면 `--name`을 바꾼다.

## 검증

기록은 [paper_artifacts/first_order_usefulness_20261109/validation](../../paper_artifacts/first_order_usefulness_20261109/validation)에 복사해 두었다.

- 2026-11-09 `validation_20261109_a`: 17개 통과(초반부 측정 전).
- 2026-11-10 `validation_20261110_a1`: 23개 통과. 로더를 확장한 뒤 초반부 checkpoint 하나를 다시 측정해 `batches.csv`와 `ksteps.csv`가 바이트 단위로 같음을 확인했다.
- 2026-11-10 `validation_20261110_late`: 58개 통과. 후반부 fixture 5개(성공한 SABER, 실패한 SABER, TD 학습, 성숙 SABER 1.5M, 실패한 HER 2M)다.

- 2026-11-10 `validation_20261110_fpp`: 43개 통과. FPP TD3 fixture 3개(SABER 500k, HER 1M, SABER 50k).
- 2026-11-10 `validation_20261110_a1_v2`(25개), `validation_20261110_late_v2`(63개): 코드를 critic 여러 개 구조로 일반화한 뒤 다시 통과. 초반부·후반부 checkpoint 각 1개를 재측정해 `batches.csv`, `ksteps.csv`, 샘플별 배열이 기존과 같음을 확인했다.

검사 항목:
- **우리의 critic 갱신이 학습 코드의 `agent.update`와 비트 단위로 같음**(DDPG, TD3 모두)
- FPP: 다시 만든 task spec이 저장된 index 키와 행동 bin을 재현. 재계산한 보상이 `env.compute_reward`와 같음. 시뮬레이터 상태 복원 뒤 한 step이 reset부터 재생한 결과와 비트 단위로 같음
- 제출 코드 `LiteWeightComputer`와 ψ·eligibility·Ω가 일치 (SABER와 HER checkpoint 각 8 batches)
- per-sample Jacobian이 autograd와 일치 (상대오차 ≤ 1.7e-7)
- 1차 항등식(uniform, Random 기댓값) 확인
- float64 유한차분과 일치. 후반부는 곡률 항이 커서 η = 1e-9, 1e-10의 Richardson 외삽을 float64 기준값과 비교한다
- 샘플 나이가 episode 순서와 일치
- checkpoint 내용(arm, seed, step)이 경로 라벨과 일치. `best_checkpoint.pt`는 로더가 거부
- float32로 재현한 Adam+clip 갱신이 torch 갱신과 같은 파라미터에 도달 (오차 ≤ 6.9e-5 × 갱신 norm)
- 원본 상태 hash 불변, snapshot 복원의 비트 단위 재현
- 11/5 분석 문서 §8.1 반례에서 SABER가 더 나쁘게 나옴(G1 = −1/6). 측정이 SABER에 불리한 결과도 낼 수 있음을 확인
- end-to-end 실행

## 비용

checkpoint당 80–100초(4 threads)가 든다. 64 batches, R1 4,000 queries, R2 1,024 queries, K = 100 기준이다. 초반부 48개는 2–4개 병렬로 약 25분이 걸렸다. 후반부 38개는 4개 병렬로 21분이 걸렸다. FPP TD3는 checkpoint당 약 115초다(critic 2개의 Jacobian 때문에 더 무겁다).

## 해석 주의

설계 문서 §11과 같다.
- 배치는 checkpoint 안의 반복이다.
- 고정 target 기준은 Q^π 정확도가 아니다.
- 값 오차 개선이 ∇_a Q 개선을 뜻하지는 않는다.
- A1은 3-seed pilot이다.
- R1 query는 정책 성능에 따라 한 phase에 몰릴 수 있다. 예: SABER seed 101의 100k는 4,000개 모두 approach. `meta.json`의 `phase_frac`을 함께 본다.

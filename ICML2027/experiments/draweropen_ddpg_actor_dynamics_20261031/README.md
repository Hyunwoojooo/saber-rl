# A1: DrawerOpen–DDPG 초기 학습과 actor 포화 추적

**목적은 실패가 진행되는 순서를 관찰하는 것이다.** HER, SABER, Random assignment, TD-error Top-k를 같은 초기 조건에서 학습시키면서 행동이 언제 ±1에 붙는지, critic이 제시하는 학습 방향과 실제 actor 변화량이 그 전후에 어떻게 달라지는지 기록한다.

이는 [추가 실험 검토의 A1](../../experment_design/2026-10-30_additional_experiments/README_ko.md) 구현이다. 기존 policy-query 가치 정확도 실험과는 다르다. Q와 gradient를 관찰하지만 참조 rollout return과 비교한 Q 정확도는 측정하지 않는다.

## 구성

| 항목 | 고정 계획 |
|---|---|
| 환경 / backbone | DrawerOpen-v3 / DDPG + HER |
| 방법 | HER 균일 가중치, SABER, eligible Random, eligible TD-error Top-k |
| 새 학습 seed | 101, 102, 103 |
| 실행당 길이 | 500,000 steps |
| 전체 pilot | 4방법 × 3seeds = 12회, 6M training steps |
| 평가 | 10k마다 20 episodes, 기존 평가 seed 규칙 |
| 실행 장치 | CPU, 실행당 1 thread, 기본 2 workers, 최대 3 |
| 목적 | 초기 현상과 시간 순서를 찾는 탐색 pilot; 우월성 확증 실험 아님 |

초기 10k는 환경에서 무작위 행동을 실행하지만, 학습은 5k부터 시작한다. 따라서 학습 직전 4,800-step checkpoint도 보존한다. 긴 실행의 best checkpoint만 분석하던 한계를 보완하려는 설계다.

## 기록하는 내용

- **행동 출력:** tanh 이전 값, ±1에 정확히 붙은 비율, tanh 미분이 0 또는 매우 작은 비율.
- **학습 신호:** critic의 행동에 대한 gradient, actor의 batch gradient와 개별 query gradient. Batch 내 상쇄와 개별 gradient 소멸을 구분한다.
- **실제 업데이트:** 기존 optimizer가 만든 actor/critic parameter 변화량, native gradient norm과 clipping 여부.
- **데이터 및 support:** 실제 실행 행동의 다양성, 학습 성공 경험, eligibility, support, 배정 포화, 실제 가중치 분산.
- **관찰 시점:** 고정된 original/HER query 집합, 현재 학습 minibatch, 실제 평가에서 방문한 상태를 구분한다.

첫 5k–10k 구간은 100-step마다, 이후에는 1k마다 상세 gradient를 측정한다. 모든 update의 요약은 별도 window로 남긴다. 체크포인트에는 모델·target·optimizer·replay/index·난수 상태를 함께 저장한다.

## 파일

| 파일 | 역할 |
|---|---|
| [PROTOCOL.md](PROTOCOL.md) | 고정 조건, 측정 정의, 해석 범위 |
| [protocol.py](protocol.py) | 방법·seed·저장 시점·실행 순서 |
| [train.py](train.py) | 한 실행을 처음부터 학습; 기본 profile은 짧은 smoke |
| [diagnostics.py](diagnostics.py) | 학습을 바꾸지 않는 관측과 checkpoint 저장 |
| [launch.py](launch.py) | 계획 출력, pilot 실행, 상태 조회 |
| [analyze.py](analyze.py) | 완료 검증과 seed별 결과·사건 시점 정리 |
| [validate.py](validate.py) | 단위 테스트와 5회의 짧은 실제 실행 검증 |
| [provenance.json](provenance.json) | 사용한 기존 실행 코드·설정의 출처와 hash |
| `reference/code/` | 기존 seed 11–20 실행 코드의 원본 복사 |
| `code/` | 원본에 HER 균일 arm 및 observer hook을 추가한 실행 코드 |
| `validation/` | 검증 보고서 및 본 pilot 실행 조건 통과 기록 |

기존 실험 코드·결과를 덮어쓰지 않는다. 동일 경로 재실행, 자동 재시도 및 checkpoint continuation을 제공하지 않는다.

## 실행 방법

아래 명령은 저장소 root 기준이다. `.venv`의 기존 설치 환경을 사용한다.

계획 확인만 수행:

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_actor_dynamics_20261031/launch.py plan
```

개발 중 코드 고정 및 검증. 검증 출력 경로는 매번 새 경로여야 한다:

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_actor_dynamics_20261031/freeze.py --create
.venv/bin/python ICML2027/experiments/draweropen_ddpg_actor_dynamics_20261031/validate.py --output ICML2027/experiments/draweropen_ddpg_actor_dynamics_20261031/validation/runs/check_01
```

검증은 4방법의 diagnostics-on 및 SABER diagnostics-off를 각각 10,200 steps 실행한다. 총 51,000 training steps와 별도의 평가/probe rollouts가 발생한다. **500k pilot은 시작하지 않는다.**

본 pilot 실행을 선택한 경우에만 사용할 명령:

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_actor_dynamics_20261031/launch.py run --workers 2
```

이 명령은 12회를 순서대로 실행하므로 오래 걸린다. 실행 전에 검증 당시와 source hash·패키지 환경·검증 보고서가 같은지 확인한다. 장시간 실행은 tmux에서 위 명령을 실행하면 된다. 실패한 job이 있으면 새 job 시작을 중단하고 이미 실행 중인 job의 종료를 기다린다. 자동으로 실패 seed를 교체하지 않는다.

상태와 결과:

```bash
.venv/bin/python ICML2027/experiments/draweropen_ddpg_actor_dynamics_20261031/launch.py status
.venv/bin/python ICML2027/experiments/draweropen_ddpg_actor_dynamics_20261031/analyze.py --attempt ICML2027/experiments/draweropen_ddpg_actor_dynamics_20261031/outputs/pilot_01/runs/her/seed_101
```

완료 시 `outputs/pilot_01/analysis/REPORT_ko.md`, `seed_metrics.csv`, `report.json`이 자동 생성된다. 분석 폴더가 이미 있으면 덮어쓰지 않는다. 실제 검증 결과는 `validation/`에 있으며 실행 계획과 구분한다.

## 해석 시 주의

500k에서 아직 성공하지 않은 실행을 2M에서도 실패할 실행으로 단정하지 않는다. Probe에서 포화를 처음 본 시점은 **그 probe에서 처음 관측한 시점**이다. 모든 상태의 최초 발생 시점이나 실패 원인 증명은 아니다. Seeds 1–20의 기존 2M 결과와 합쳐 평균·CI를 만들지 않는다.

체크포인트는 critic 분기 분석에 필요한 학습 상태를 보관하지만 **MuJoCo 내부 상태 복원이나 online 학습 재개를 지원하지 않는다.** 원본 goal 입력, 지원 조건, optimizer 등을 이번 실험에서 수정하지 않는다.

## 구현 및 자원 방법 기록

설계 근거는 연결된 A1 제안과 기존 seed 확장 감사다. `get-available-resources` skill의 관측 절차를 사용해 CPU affinity, 가용 메모리, 저장 공간을 확인하고 CPU 1 thread 및 작은 병렬 수를 선택했다. 전체 host 정보는 기록하지 않는다. 각 실행은 Python/핵심 package 버전과 설정을 남긴다.

관련 skill 출처: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065). 이 인용은 연구 결과의 근거가 아니라 사용한 절차 도구의 출처다.

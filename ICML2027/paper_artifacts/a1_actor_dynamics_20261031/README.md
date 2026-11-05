# A1: DrawerOpen–DDPG 초기 학습 추적 — 완료 자료

**HER / SABER / Random assignment / TD-error Top-k, seeds 101–103, 각 500k의 12회 학습과 자동 분석이 완료됐다.** 총 training steps는 6M이며, 마지막 분석 종료는 2026-11-01 04:35:16 UTC / 13:35:16 KST다.

이 실험은 행동 출력 포화와 actor 학습 신호, support 배정의 시간적 변화를 관찰하는 3-seed 탐색 pilot이다. 기존 2M 성능 비교나 policy-query Q 정확도 실험과 구분한다. 모든 실행과 실패 양상을 포함한다.

## 핵심 자료

- [완료 보고서](analysis/REPORT_ko.md): 방법별 지표와 seed별 포화 관측 시점.
- [Seed별 결과 CSV](analysis/seed_metrics.csv), [집계 JSON](analysis/report.json).
- [실제로 실행한 코드](../../experiments/draweropen_ddpg_actor_dynamics_20261031/README.md), [고정 protocol](../../experiments/draweropen_ddpg_actor_dynamics_20261031/PROTOCOL.md).
- [실행 일정과 완료 상태](cohort/status.json), [실행 계획과 전체 설정](cohort/plan.json).
- [실행 전 검증](validation/REPORT.md): 14개 단위 테스트, 5회 smoke, 계측 on/off 학습 상태 일치.

| 방법 | 10k–500k AUC 평균 | 500k 평가 성공률 평균 | 평가가 전부 0인 실행 |
|---|---:|---:|---:|
| HER | 0.131633 | 0.667 | 1/3 |
| SABER | 0.011735 | 0.117 | 1/3 |
| Random assignment | 0.072279 | 0.283 | 0/3 |
| TD-error Top-k | 0.132653 | 0.333 | 1/3 |

SABER seed 102에서는 고정 HER probe 전체의 tanh 미분이 0인 상태를 399k에, 고정 original-goal probe에서는 407k에 처음 관측했다. 다른 실행에서는 같은 전체 포화 조건을 500k까지 관측하지 않았다. 부분 포화와 gradient 크기의 자세한 변화는 진단 기록에 있다. 이 관측만으로 실패 원인이나 장기 성능 우위를 확정하지 않는다.

## 공개한 파일과 원본 보존

`records/<arm>/seed_<seed>/`에는 각 실행의 전체 설정, 환경 버전, 완료 감사, native metrics CSV, assignment window, evaluation/probe/update/transition 진단, checkpoint 목록 및 hash를 담았다. CSV/JSONL은 `.gz`로 무손실 압축했다. 고정 probe bank도 포함한다.

학습·분석 원본 파일을 수정하거나 training source manifest를 다시 만들지 않았다. `manifest.json`은 공개 파일 hash와 압축 전 원본 hash를 함께 기록한다. 실행 기록의 절대 경로와 PID는 원본의 과거 실행 metadata이며 다른 PC에서 그대로 사용할 실행 경로가 아니다.

큰 PyTorch checkpoint, 중복 native `metrics.jsonl`, console log 및 runtime cache는 이 Git 묶음에 포함하지 않는다. Checkpoint 156개의 목록과 hash는 보존하지만 아래 공개 검증이 checkpoint tensor 내용을 다시 검사하는 것은 아니다. Checkpoint 기반 재분석은 로컬 원본이 필요하다.

## 검증

저장소 clone 후 Python 표준 라이브러리만으로 실행할 수 있다. 학습이나 GPU 실행을 시작하지 않는다.

```bash
python3 ICML2027/paper_artifacts/a1_actor_dynamics_20261031/verify.py
```

파일 및 동결 코드 hash, gzip 원본 byte 동일성, 12개 실행의 설정과 완료 상태, 평가 600개, AUC/최종 성공률 재계산, 고정 probe의 포화 최초 관측 시점, update·transition window와 HER 균일 가중치 기록을 확인한다.

이 PC의 원본 파일까지 추가 대조하려면:

```bash
python3 ICML2027/paper_artifacts/a1_actor_dynamics_20261031/verify.py --original-experiment ICML2027/experiments/draweropen_ddpg_actor_dynamics_20261031
```

`validation/gate.json`은 실행 당시의 로컬 검증 기록이다. 다른 PC에서 새 학습을 시작하려면 source 안내대로 검증을 다시 수행한다. 이미 완료된 이 cohort의 결과를 확인하는 데 새 학습이나 smoke 실행은 필요 없다.

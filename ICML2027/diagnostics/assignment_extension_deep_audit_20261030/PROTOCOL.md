# DrawerOpen DDPG seed 11–20 사후 원인 감사

2026-10-30. 모든 결과를 이미 본 뒤 시작한 탐색 분석이다. 기존 primary 결과·CI·학습 source·checkpoint를 수정하지 않는다. 새 본학습, optimizer update, seed 교체는 없다.

## 질문과 독립 단위

- 질문: 추가 cohort에서 SABER의 평균 AUC가 낮아진 양상과 이를 설명하는 근거·미확인 가설은 무엇인가?
- 학습 run이 독립 분석 단위다. 시간 window, query, 평가 episode는 학습 seed 수로 세지 않는다.
- 기존 1–10의 공개 source/compact summaries와 새 11–20의 30개 원시 run을 구분한다. 과거 run별 환경·raw curve·checkpoint가 없어 실행 시기와 seed 효과를 분리할 수 없다.

## 분석 범위

1. 30개 run의 config/environment/budget/평가 grid/지표 재계산, source/release hash, 적용 mask 감사.
2. 전체 seed 유지. AUC 분포, Hit90 도달률, 첫 성공 경험, support와 포화의 시간 순서. 사후 성공/실패 그룹은 원인 진단용이며 primary 비교를 대체하지 않는다.
3. 평균 변화의 대칭적 산술 분해: Hit90 비율 변화와 각 그룹의 조건부 AUC 변화. 인과 매개효과로 해석하지 않는다.
4. 추가 cohort의 Hit90 paired discordance에 대한 exploratory exact McNemar/binomial 검토. 기존 bootstrap CI를 대체하거나 유리한 검정을 고르지 않는다.
5. 30개 최종 checkpoint + SABER 최종 실패 6개의 retained best checkpoint를 CPU에서 순차 복원한다. Best checkpoint는 저장 정책이 남긴 시점이며 동등한 학습 단계가 아니다. Seed20 best는 680k, 다른 실패 seed의 best는 10k다.
6. Checkpoint마다 고정 RNG로 HER minibatch 256개와 같은 replay slot의 original-goal query 256개를 검사한다. Native optimized support와 같은 Tail-512 후보에 대한 직접 NumPy full-context matching을 대조한다. Native 반복 호출로 cache 신호 일치도 확인한다. 각 goal 종류의 첫 8개 query에만 Full 검색·leave-one-out 진단을 추가한다. Full은 같은 이산 context 정의 안의 참조값이다.
7. Replay 전체 유효 slot의 coarse key/action-bin과 index membership 및 bin-count 일치 확인. 저장 network/RNG/파일을 변경하지 않는다.
8. 최종 정책 30개를 원래 최종 평가의 첫 4개 reset에서 재실행한다(총24k 환경 step). 아직 checkpoint 분석을 시작하기 전, 진단 목적을 새로운 분포 평가보다 원래 평가의 재현 확인으로 명확히 했다. Eval env의 생성 seed는 원래 cfg.eval_seed, reset seed는 cfg.eval_seed+10*2M+episode_index다. 같은 training seed의 세 arm은 같은 초기 상태인지 hash로 대조한다. 신경망 업데이트 없음. 원래 20-episode primary 평가를 대체하지 않는 재현 진단이다. Action saturation, 접근/접촉/열기 phase, 거리, 성공 여부를 기록한다.

## 산출물과 한계

Checkpoint 재평가에서 실패 정책의 행동 성분이 모두 ±1로 고정된 것을 관찰한 뒤, 후속 탐색 진단을 추가했다: 같은 36개 checkpoint의 동일한 고정 replay query에서 actor pre-tanh 출력, tanh 도함수, original/HER goal별 actor gradient norm을 측정한다. Optimizer step은 하지 않는다. 이 추가는 결과를 본 뒤 정한 가설 검사이며 초기 실패 원인을 확정하는 개입은 아니다.

첫 gradient 측정 이후 실패 SABER 6개의 독립적인 4개 추가 batch/goal 종류(각256 query)와 Adam moment를 더 확인한다. 이 역시 zero gradient 발견 이후의 후속 검사다. 마지막 정책의 z=+1, noise sigma=.2 조건에서 음수 z가 나올 확률은 모델 조건부 계산으로만 제시하며 학습 전 구간이 그 상태였다고 가정하지 않는다.

로그 의미 감사에서 CSV의 unprefixed 기하·phase 열이 HER batch 통계이고 JSONL eval row에는 실제 평가 기하 통계가 보존된 것을 확인했다. 실제 평가 geometry는 JSONL에서 추출한다. 이후 탐색 marker로 approach_fraction=1 및 평균 손–손잡이 거리>0.4m를 사용해 첫 관측·이후 지속 시점을 요약한다. 이는 결과를 본 뒤 정한 상태 기술자이며 actor 완전 포화의 정확한 최초 시점이나 사전 실패 예측 지표로 해석하지 않는다.

- 원시 seed/time-band/checkpoint/query/rollout 요약 CSV, source/input/output manifest, 재현 script, 한국어 보고서, standalone figures.
- 모든 시각화는 unsmoothed 원자료 또는 명시한 run 단위 요약을 사용한다. 0과 missing을 구분한다.
- Offline query는 각 run의 고유 replay 분포에서 뽑으므로 support 활성률의 arm 차이를 곧바로 인과효과로 해석하지 않는다.
- 탐색 분석은 원인 후보를 좁힐 수 있지만, 초기 support 배정이 실패를 일으켰는지와 seed/실행환경 효과의 분리는 별도의 개입·재현 실험 없이는 확정할 수 없다.

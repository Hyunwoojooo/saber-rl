# A1 고정 protocol — 2026-10-31

## 1. 질문과 비교 단위

DrawerOpen–DDPG에서 actor의 행동 포화, critic 행동 gradient 변화, actor 학습 신호 소멸, support 배정 포화와 성공 경험이 어떤 순서로 나타나는가? 이러한 현상이 네 방법에서 어떻게 다른가?

독립적인 단위는 training seed다. Method별 probe 32개, 평가 episode 20개, 여러 시간 지점을 독립 학습 실행처럼 집계하지 않는다. 이 pilot의 3 seeds로 확정적인 실패 확률·성능 우월성·인과관계를 주장하지 않는다.

## 2. 고정 학습 조건

실제로 완료된 assignment extension support seed 11의 `requested_config.json`을 기준으로 사용한다. 파일 출처 및 SHA256은 `provenance.json`, 모든 명시적 설정은 `config_template.json`과 실행별 `requested_config.json`에 남긴다.

주요 조건: CPU DDPG, batch 256, replay capacity 300k, UTD 1, gamma .98, tau .005, actor/critic LR .0003, hidden 256, exploration sigma .2, horizon 200, warmup 5k, random actions 10k, HER nominal .8 strict-future. Success 조기 종료 없음; timeout bootstrap. Tail support query: K=5, N_scan=512, N_local=4, M_Q_min=4, M_B=16, lambda=1. Critic/actor gradient clipping과 네트워크 초기화는 원본을 따른다.

각 방법의 변경 사항은 critic coefficient 배정뿐이다:

| Arm | 배정 |
|---|---|
| `her` | 모든 critic/actor coefficient가 정확히 1 |
| `saber` | 현재 policy-action support-positive M에 높은 coefficient |
| `random` | context eligible E 안에서 M의 크기 k와 같은 개수를 균일 비복원 추출 |
| `td_topk` | E 안에서 업데이트 전 TD error 절댓값 상위 k개; 경계 동률만 무작위 |

세 재가중 방법은 `(1+mask)/mean(1+mask)`를 사용한다. HER도 관측을 위해 같은 support 및 가상 배정 계산을 수행하지만 loss에는 전부 1만 전달한다. 따라서 이번 HER의 runtime은 계측 없는 HER의 알고리즘 비용을 대표하지 않는다. 모든 actor coefficient는 1이며 actor/target 업데이트는 원본과 같다.

각 실행은 자신의 policy/replay로 E, M, k를 계산한다. 방법 사이의 k나 전체 학습 trajectory가 항상 같다는 설계가 아니다. 동일 seed는 초기 조건과 난수 규칙을 맞춘다.

## 3. 실행 규모와 순서

- Seeds 101, 102, 103, 각 500k, 총 12회 / 6M training steps.
- 각 seed 안의 네 방법 launch 순서를 randomized cyclic Latin square로 고정한다. Order RNG seed 20261031. 한 seed block을 끝낸 다음 다음 seed로 간다. 기본 2 workers이므로 일부 실행 시간은 겹친다.
- 환경/torch/numpy/exploration seed=s; replay/HER RNG=s+17; weight RNG=20000+s; random assignment=30000+s; TD tie=40000+s. 평가 reset seed는 원본의 `10000+s+10*step+episode_index`.
- 추가 진단 rollout은 별도 환경과 private RNG를 사용한다. Python/NumPy/PyTorch global RNG는 복원한다.
- 고정 500k 종료, 결과를 본 뒤 길이·seed·방법·threshold를 바꾸지 않는다. 추가 단계는 새 protocol로 구분한다.

## 4. 측정 지점

### Checkpoint

0, 4,800, 5,000, 6,000, 8,000, 10,000, 20,000, 50,000, 100,000, 200,000, 300,000, 400,000, 500,000 steps. 모두 horizon 200의 episode boundary다. 4,800은 학습 전 상태다. 5,000에는 첫 critic/actor update가 이미 포함된다.

각 checkpoint는 actor/critic과 target, Adam state, replay 및 index, 학습 RNG, update 수, 설정을 포함한다. 추가 checkpoint에서 native learning state를 수정하지 않는다. 저장된 simulator 내부 상태가 없으므로 online continuation은 금지한다. best checkpoint도 native code가 만들 수 있지만 A1 분석의 입력은 위 고정 시점이다.

### 상세 gradient 관측

Checkpoint 시점, 5k–10k 사이 매 100 steps, 10k 이후 매 1k steps. 각 시점에서:

1. 별도 random rollout 4개에서 미리 뽑은 32개 고정 observation을 original goal 및 strict-future HER goal로 조회한다. 각 rollout의 0/20/40/60/80/100/140/180 시점을 쓴다. 이 800 steps는 training replay에 들어가지 않는다.
2. 학습이 있는 시점에는 실제 minibatch 첫 32개 observation을 HER goal 및 저장된 original goal로 조회한다. Critic update 전과 후에 같은 actor로 측정한다.
3. Actor update 전후 parameter 차이를 기록한다. 별도 optimizer step을 추가하지 않는다.
4. 실제 evaluation 종료 후 방문한 observation의 등간격 32개에서도 측정한다. 이 상태는 transition **후** 관측이며, goal은 실제 평가 goal이다.

`autograd.grad`를 사용하고 `.grad` buffer와 optimizer state를 변경하지 않는다. 원본 actor/critic에는 dropout/BatchNorm이 없다. 전체 환경 학습을 계측 on/off로 각각 실행해 최종 모델·optimizer·replay·난수와 assignment history parity를 검증한다.

### Gradient의 의미

- `critic_action_gradient_l2_mean`: 각 query의 ∂Q/∂a norm. Batch 평균으로 나눈 gradient가 아니다.
- `actor_batch_gradient_l2`: 고정 query 평균 `-Q`의 actor parameter gradient norm.
- `individual_actor_gradient_*`: 같은 집합 앞 8개 query별 parameter gradient. Batch gradient 0이 상쇄 때문인지 구분하기 위한 제한된 표본이다.
- `native_*_preclip_gradient_norm`: 원본 update가 측정한 gradient norm. 원본 float32 norm은 매우 작은 gradient에서 reduction underflow 가능성이 있다. 이것만으로 모든 gradient 성분이 정확히 0이라고 단정하지 않는다.
- Probe norm과 parameter delta norm은 float64로 제곱합을 누적한다. 추가 정밀도로 원래 float32 backward에서 이미 소실된 정보를 복구하는 것은 아니다.
- `tanh_derivative_exact_zero_fraction`: 현재 float32 tanh 출력에서 `1-a²==0`인 성분 비율. 근접 포화 `|a|>=.999`, derivative `<1e-6`도 따로 기록한다.
- Actor gradient 0이어도 Adam momentum 때문에 실제 parameter가 움직일 수 있으므로 실제 delta를 함께 기록한다.
- Critic update 전후 ∂Q/∂a의 cosine은 둘 중 norm이 0이면 `null`이다. Q/gradient가 크다는 사실은 정확도나 좋은 정책 개선 방향의 증명이 아니다.

### 연속 window

모든 native update에서 loss, gradient norm, clipping, eligibility, support, k=m>0 포화, actual weight variance를 관측한다. 첫 10k까지 100 updates, 이후 1,000 updates 단위 평균과 정확한 start/end/count를 기록한다. 마지막 불완전 window도 보존한다.

실제 학습 transition은 1,000-step window로 실행 행동 평균/표준편차, K=5 action-bin 다양성, 손–handle 거리와 original goal success 비율을 기록한다. 최초 학습 goal/env success 및 episode 내 성공도 별도 event로 기록한다. Actual action에는 random prefix와 exploration noise가 포함되어 policy probe action과 다르다.

평가 geometry는 native JSONL에서 수집한 실제 evaluation observation을 사용한다. 기존 CSV의 동일 이름 training geometry 열을 평가 geometry로 재사용하지 않는다. Observation 끝의 native goal과 외부 HER goal의 차이는 측정만 하며 입력 구성을 바꾸지 않는다.

## 5. 결과와 실패 처리

- 완료된 모든 seed를 포함한 AUC(10k–500k), final success, Hit90, all-zero-evaluation 여부를 기술적으로 보고한다. AUC 분모 490k; 50개 checkpoint. 우월성 검정/CI는 이 pilot에서 생성하지 않는다.
- 고정 original/HER probe 전체 tanh 미분이 0인 첫 관측 시점, 최초 학습 success 및 최초 support 포화가 포함된 window를 함께 보고한다. Window timestamp는 정확한 최초 발생 step이 아니다.
- 500k까지 관측되지 않은 사건은 미관측/우측 검열이다. 500k 실패 양상이 2M에도 유지된다는 결론을 내리지 않는다.
- 모든 값이 유한하지만 성공률 0인 run은 유효한 결과다. Non-finite는 실패로 기록하고 0으로 대체하지 않는다. Checkpoint/평가 grid가 누락되면 분석 실패로 남긴다.
- 새 출력 경로만 사용하며 overwrite/resume/자동 재시도를 금지한다. 실패 후 재실행을 원한다면 원인과 별도 attempt 계획을 먼저 기록한다.
- 같은 seed의 비교와 시간적 선후는 인과관계의 충분한 증거가 아니다. 관측에 따라 critic 또는 actor를 실제로 바꾸는 분기 실험은 후속 A2로 구분한다.

## 6. 구현 검증과 자원

단위 테스트는 세 기존 selector 및 실제 Adam update의 원본 대비 parity, HER uniform parity, probe RNG/gradient 불변, batch cancellation, saturation, 작은 gradient reduction을 확인한다.

실제 smoke는 seed101, 네 arm diagnostics-on과 SABER diagnostics-off의 총 5회×10,200 steps다. 과학적 설정과 warmup/random-prefix 규칙은 pilot과 같고 total_steps만 짧다. 5,201 updates 및 초기부터 policy 행동으로 전환된 뒤까지 점검한다. 각 실행의 checkpoint/config/evaluation/관측 window를 감사하고 SABER on/off 전체 학습 state를 비교한다. Four-arm 4,800-step 학습 전 상태와 fixed probe bank도 대조한다. 이 smoke는 성능 표에 넣지 않는다.

Source manifest가 바뀌거나 핵심 package 환경이 달라지면 기존 validation gate를 재사용할 수 없다. 최초 pilot 이후 코드는 새 버전 폴더에서 관리한다. Runtime 비교가 목적은 아니지만 계측량과 CPU thread 규칙을 네 방법에 동일하게 적용한다. 실제 checkpoint 크기에 따라 공간 사용량이 달라지므로 launcher는 최소 64 GiB의 여유 공간을 요구한다.

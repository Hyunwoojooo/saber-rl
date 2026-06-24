# SABER 이론 계보와 현재 논문용 정의

작성일: 2026-06-22
확인한 저장소:

- `https://github.com/Hyunwoojooo/metacog-rl`
- `https://github.com/Hyunwoojooo/saber-rl`

이 문서는 `metacog-rl`에서 시작된 아이디어가 `saber-rl`의 SABER 실험으로 어떻게 축소/구체화되었는지 정리한다. 결론부터 말하면, 현재 논문에서 가장 방어 가능한 SABER 정의는 **local replay support를 Bellman critic fitting의 신뢰도 신호로 사용해 critic TD loss만 재가중하는 방법**이다.

## 1. 한 줄 요약

`metacog-rl`의 원래 출발점은 “실패 원인을 ignorance, incompetence, impossibility로 나누고 원인별로 다른 학습 개입을 하자”는 메타인지적 RL 프레임워크다. 그중 SABER로 이어진 부분은 **incompetence/PRACTICE** 축이다. 즉, 모델이나 환경 전체를 다시 탐색하기보다, 현재 정책이 아직 잘 못하지만 replay buffer 안에 충분한 local support가 있는 상태-행동 근방을 더 신뢰하고 더 강하게 critic에 학습시키는 방향으로 단순화되었다.

현재 `saber-rl`에서 논문 메인으로 쓰기 좋은 정의는 다음이다.

> **SABER = Support-aware critic-side Bellman error reweighting.**
> Goal-conditioned sparse-reward off-policy RL에서, HER로 relabel된 각 transition에 대해 local replay support 여부를 판정하고, support-positive transition의 critic Bellman loss에 더 큰 가중치를 준다. Actor loss와 replay sampling distribution은 기본적으로 건드리지 않는다.

주의: 저장소 내부에서 `SABER`의 공식 acronym 확장은 명시적으로 확인되지 않았다. 위 문구는 코드 동작에 가장 잘 맞는 논문용 해석이다. 코드/실험 문서에서는 `SABER = B3-critic-only`로 반복 정의된다.

## 2. 계보

### 2.1 MetaCog-RL 원래 이론

`metacog-rl/MetaCog-RL_Paper_Draft.md`는 RL 실패를 세 종류로 분해한다.

| 실패 유형 | 의미 | 진단 신호 | 대응 전략 |
| --- | --- | --- | --- |
| Ignorance | dynamics/world model을 모름 | ensemble disagreement `I(s,a)` | EXPLORE |
| Incompetence | 모델은 아는데 정책이 못함 | competence gap `C(s)` | PRACTICE |
| Impossibility | 환경 자체가 stochastic/irreducible | aleatoric variance `A_var(s)` | ROBUST |

핵심 수식은 competence gap이다.

```text
C(s) = max_a Q_WM^pi(s, a) - V_WM^pi(s)
     = max_a A_WM^pi(s, a)
```

이는 “현재 정책 행동보다 더 좋은 행동이 현재 모델 관점에서 얼마나 남아 있는가”를 뜻한다. 중요한 점은 `C(s)`가 world model이 맞을 때만 의미가 있다는 것이다. 그래서 원래 이론은 ignorance를 먼저 확인하고, low-ignorance 영역에서만 incompetence/PRACTICE를 적용한다.

`metacog-rl/src/metacog_v5/core/diagnostics.py` 구현에서도 이 흐름이 남아 있다. 특히 `_competence()`는 supported candidate action들 중 최고 Q값과 policy action Q값의 차이를 `g_sup`로 계산한다.

```text
g_sup = max(Q(x, a_supported)) - Q(x, a_policy)
kappa = normalized(g_sup)
psi_q_rank = support 조건이 충분할 때 1
```

`metacog-rl/src/metacog_v5/core/intervention.py`에서는 PRACTICE utility가 다음 성분을 곱해서 만들어진다.

```text
u_practice_online = eta_p * psi_f * psi_q_rank * phi_hi * kappa
w_P = 1 + lambda_p * clipped(u_practice_online)
```

즉 원래 MetaCog 구현에는 이미 “support가 있고, feasible하고, competence gap이 큰 곳을 더 practice한다”는 구조가 있다.

### 2.2 MetaCog-Lite로의 축소

`saber-rl` 초기 코드들은 `metacog_lite_v0` 패키지명을 유지한다. 하지만 실제 실험은 MBRL/world-model ensemble이 아니라 DDPG/TD3/SAC + HER 기반 model-free sparse manipulation으로 옮겨졌다.

이 전환에서 빠진 것:

- world model ensemble
- explicit ignorance/aleatoric decomposition
- CVaR-style robust control
- full `C(s) = max_a Q - Q_pi` 기반 expensive candidate Q evaluation을 메인 경로에서 사용

남은 것:

- replay buffer 안의 local support 판정
- support-positive 샘플에 대한 학습 가중치
- HER goal-conditioned context
- PRACTICE에 해당하는 critic fitting 강화

따라서 SABER는 MetaCog-RL 전체가 아니라, **MetaCog의 PRACTICE/incompetence 아이디어를 sparse HER actor-critic에 맞게 최소화한 파생 방법**으로 보는 것이 정확하다.

### 2.3 B3에서 SABER로

`saber-rl/experiment_FetchPush/EXPERIMENT_PLAN.md`에는 논문용 정의가 가장 직접적으로 적혀 있다.

```text
SABER = B3-critic-only
SABER uses local replay support as a critic-side Bellman fitting reliability signal.
```

즉 현재 최종 실험에서 SABER는 B3 계열 중 `critic_only` scope를 의미한다. 특히 `experiment_FPP`, `FetchPush`, `FetchSlide`, `MetaWorld DrawerOpen`의 메인 결과는 다음 공통 구조를 따른다.

- B0: HER baseline, support weighting 없음
- SABER/B3 critic-only: support-positive 샘플의 critic loss만 재가중
- PER/TD-error baseline: support가 아니라 TD-error 또는 priority 기반 비교군
- random/shuffled controls: support 신호와 weight distribution 효과를 분리하기 위한 ablation

## 3. SABER의 정확한 알고리즘 정의

아래는 현재 `saber-rl` 코드 기준 논문에 쓸 수 있는 추상화다.

### 3.1 입력

off-policy actor-critic + HER 학습 배치:

```text
B = {(o_i, g_i, h_i, a_i, r_i, o'_i, g_i)}
```

여기서 `g_i`는 HER로 relabel된 training goal일 수 있고, `h_i`는 남은 horizon 또는 goal-conditioned context에 들어가는 시간 정보다.

### 3.2 Local replay support

각 샘플 `i`에 대해 task-specific context key를 만든다.

```text
c_i = context(o_i, g_i, h_i)
```

그리고 replay buffer에서 같은 또는 유사한 context를 가진 transition들을 찾는다.

```text
L_i = {j in replay buffer : context(o_j, g_i, h_j) ~= c_i}
```

실제 구현은 task별 coarse/full context hash와 bounded local scan을 쓴다.

- FetchPickAndPlace: hand-object 거리 bin, object-goal 거리 bin, object z bin, gripper gap bin, horizon bin 등
- FetchPush/FetchSlide: end-effector/object/goal 거리, contact/slide phase, velocity/align bin 등
- DrawerOpen: hand-handle 거리, handle-goal 거리, handle-y, gripper, phase, horizon bin 등

### 3.3 Support-positive predicate

현재 actor가 낸 action `pi(o_i, g_i)`가 replay의 local action support 안에 충분히 있다고 볼 수 있으면 `psi_i = 1`, 아니면 `0`이다.

코드상 B3 support predicate는 대략 다음 조건이다.

```text
local_count_i >= N_local_min
available_support_i = min(local_count_i, M_B)
available_support_i >= M_Q_min
policy action bin is observed in the local replay action bins
```

최종 주요 실험에서 자주 쓰인 threshold 예시는 다음이다.

```text
M_B = 16
M_Q_min = 4
N_local_min = 4
N_gap_min = 128
lambda_support = 1.0
weighting_mode = batch_norm
weight_scope = critic_only
```

환경/실험별 `local_search_max`는 다르다. FPP/FetchSlide에서는 runtime 때문에 `512`가 핵심 설정으로 수렴했고, FetchPush DDPG rerun은 `256`을 사용한 기록이 있다.

### 3.4 Weight construction

SABER/B3 support-only critic-only에서는 Q candidate evaluation 없이 support predicate만으로 raw weight를 만든다.

```text
w_raw_i = 1 + lambda_support * psi_i
```

`batch_norm` 모드에서는 batch 평균을 1로 맞춘다.

```text
omega_i = w_raw_i / mean_j(w_raw_j)
```

critic-only scope에서는:

```text
omega_critic_i = omega_i
omega_actor_i  = 1
```

actor-critic 또는 actor-only variant도 코드에는 있지만, 현재 논문 메인 claim에는 critic-only가 가장 깨끗하다.

### 3.5 Critic update

backbone별 Bellman target을 `y_i`라고 하면 SABER critic loss는 다음이다.

```text
L_Q(theta) = (1 / |B|) * sum_i omega_critic_i * (Q_theta(s_i, a_i) - y_i)^2
```

DDPG 코드에서는 이 형태가 직접 보인다.

```text
critic_loss = mean(omega_critic * td_error^2)
actor_loss  = -mean(omega_actor * Q(s, pi(s)))
```

critic-only SABER에서는 `omega_actor = 1`이므로 actor objective는 baseline과 동일한 형태다. TD3/SAC에서는 target 계산은 backbone에 맞게 바뀌지만, 핵심은 support-derived weight가 critic Bellman fitting 쪽에만 들어간다는 점이다.

## 4. 왜 이것이 MetaCog에서 파생되었는가

MetaCog-RL의 PRACTICE는 “모델이 어느 정도 알고 있고, 정책이 아직 못하는 영역을 더 연습시킨다”는 개입이다. SABER는 이 개입을 model-free HER setting으로 옮기면서 다음과 같이 바꿨다.

| MetaCog-RL | SABER |
| --- | --- |
| world model ensemble로 ignorance 확인 | replay local support로 data-manifold 여부 확인 |
| `C(s)=max_a Q_WM - V_WM` competence gap | current policy action이 local replay support 안에 있는지 확인 |
| PRACTICE prioritized replay/focused imagination | critic Bellman loss reweighting |
| 세 실패 유형 전체 라우팅 | support-aware critic fitting 하나에 집중 |

즉 SABER는 MetaCog-RL의 전체 3-way diagnosis를 주장하지 않는다. 대신 “정책 개선에 필요한 critic target을 모든 샘플에 똑같이 맞추는 것이 아니라, replay support가 충분한 샘플을 더 신뢰해서 Bellman fitting한다”는 더 좁고 검증 가능한 주장으로 바뀐다.

## 5. SABER가 PER/TD-error weighting과 다른 점

SABER는 “큰 error를 더 학습”하는 방법이 아니다.

| 방법 | 기준 신호 | sampling 변경 | loss weighting | 핵심 의미 |
| --- | --- | --- | --- | --- |
| PER | TD-error priority | 있음 | importance correction 가능 | surprising sample 재사용 |
| TD-error loss weighting | TD-error magnitude | 보통 없음 | critic loss | Bellman error가 큰 샘플 강조 |
| SABER | local replay support | 없음 | critic loss | Bellman target fitting이 더 신뢰되는 샘플 강조 |

`experiment_FetchPush/EXPERIMENT_PLAN.md`도 이 차이를 명시한다.

```text
SABER keeps sampling fixed and changes critic loss weights.
Support signal is not simply TD-error under another name.
Support-rich samples should behave like more reliable critic fitting targets.
```

따라서 논문에서는 SABER를 PER의 변형으로 쓰면 약해진다. 더 정확한 framing은 **support-aware Bellman reliability weighting**이다.

## 6. 현재 실험 결과가 지지하는 claim 범위

### 6.1 강하게 말할 수 있는 것

현재 `aaai27-core-artifacts`의 주요 결과 기준으로는 다음 정도가 방어 가능하다.

1. SABER는 DDPG/TD3/SAC + HER에 붙는 backbone-agnostic critic-side weighting이다.
2. FetchPickAndPlace에서는 DDPG/TD3/SAC 계열에서 HER baseline 대비 성능 개선이 명확하다.
3. FetchSlide에서는 SAC+SABER, TD3+SABER가 각각 SAC+HER, TD3+HER보다 final/AUC 계열에서 개선을 보인다. DDPG+SABER는 이 환경에서는 baseline보다 약하다.
4. DrawerOpen TD3 2M에서는 B3-Critic-Only가 TD3+HER보다 더 많은 seed를 성공시키고 final/best/AUC를 개선한다.
5. Critic-only scope가 actor-only보다 더 깨끗한 메인 설계다. Actor+critic도 경쟁력은 있지만, mechanism isolation은 critic-only가 낫다.

### 6.2 조심해야 할 것

1. SABER가 항상 모든 backbone/task에서 이긴다고 쓰면 안 된다. FetchSlide DDPG에서는 SABER가 DDPG+HER보다 낮다.
2. DrawerOpen에서는 Random-Support가 의외로 강하다. 따라서 “정확한 structured support rule만이 전부”라고 주장하면 위험하다.
3. Shuffled-Weight control은 DrawerOpen에서 강한 negative control이다. 같은 weight distribution을 유지해도 sample-weight alignment가 깨지면 실패했다. 이 결과는 SABER claim에 유리하지만, Random-Support 결과와 함께 해석해야 한다.
4. 현재 final SABER는 MetaCog-RL의 full 3-way metacognitive diagnosis가 아니다. 논문에서 full MetaCog-RL theory를 그대로 주장하면 구현과 맞지 않는다.
5. runtime overhead가 크다. 논문에서는 sample efficiency와 wall-clock tradeoff를 분리해서 보고해야 한다.

## 7. 논문에서 추천하는 이론 서술

### 7.1 문제 정의

sparse-reward goal-conditioned manipulation에서 HER는 성공/실패 transition을 다양하게 relabel하지만, critic은 여전히 off-policy Bellman target을 모든 샘플에 동일 신뢰도로 fitting한다. 이때 어떤 transition은 local replay context에서 충분히 support되어 Bellman target이 안정적인 반면, 어떤 transition은 policy action이 replay support 밖에 있어 extrapolation 위험이 크다.

### 7.2 핵심 가설

```text
Replay support가 충분한 transition은 critic Bellman fitting에 더 신뢰할 수 있다.
따라서 support-positive transition의 critic TD loss를 더 크게 반영하면
sparse manipulation에서 value learning이 안정화되고 sample efficiency가 좋아진다.
```

### 7.3 방법 한 문단

각 HER training sample에 대해 task-specific local context를 구성하고, replay buffer에서 같은 context의 action support를 검색한다. 현재 policy action이 해당 local support 안에서 충분히 관측되고, 최소 support count 조건을 만족하면 sample을 support-positive로 표시한다. SABER는 support-positive sample에 raw weight `1 + lambda`를 부여하고 batch-normalized weight를 critic TD loss에 곱한다. Actor objective와 replay sampling은 baseline과 동일하게 유지하여, support signal의 효과를 critic Bellman fitting으로 분리한다.

### 7.4 이론적 위치

SABER는 MetaCog-RL의 full diagnosis가 아니라, 그중 PRACTICE/incompetence 아이디어의 conservative model-free instantiation이다. 원래 competence gap `C(s)`는 world-model 기반 `max_a Q - V`지만, sparse HER setting에서는 expensive하고 불안정하다. SABER는 이를 직접 추정하지 않고, “policy action이 local replay manifold 위에 있는가”라는 support predicate를 사용해 critic update의 신뢰도를 제어한다.

## 8. 논문 claim 문장 초안

강한 버전:

> SABER introduces support-aware Bellman error reweighting for goal-conditioned off-policy RL. Instead of prioritizing transitions by TD error, SABER identifies samples whose current policy actions are locally supported by replay actions under the HER-relabeled goal context and upweights only their critic losses.

한글 해석:

> SABER는 goal-conditioned off-policy RL을 위한 support-aware Bellman error reweighting 방법이다. TD-error가 큰 transition을 더 뽑는 대신, HER로 relabel된 goal context에서 현재 policy action이 replay action support 안에 있는 sample을 찾고, 해당 sample의 critic loss만 더 크게 반영한다.

더 조심스러운 버전:

> SABER is a lightweight descendant of the MetaCog-RL PRACTICE mechanism: it replaces explicit competence-gap estimation with a replay-support predicate and applies the resulting signal as critic-side Bellman fitting weights.

## 9. 코드 근거 지도

### metacog-rl

| 파일 | 의미 |
| --- | --- |
| `MetaCog-RL_Paper_Draft.md` | ignorance/incompetence/impossibility 3-way diagnosis, competence gap, PRACTICE 전략 정의 |
| `src/metacog_v5/core/diagnostics.py` | `g_sup`, `kappa`, `psi_q_rank`, support filtering 구현 |
| `src/metacog_v5/core/intervention.py` | PRACTICE utility와 `w_P` learning weight 계산 |
| `src/metacog_v5/core/replay.py` | `w_P` 기반 prioritized sample path |
| `ANALYSIS_REPORT.md` | 원래 MetaCog 이론의 약점/수정 필요점 자체 검토 |

### saber-rl

| 파일 | 의미 |
| --- | --- |
| `experiment_FetchPush/EXPERIMENT_PLAN.md` | `SABER = B3-critic-only`, support-aware critic Bellman reliability signal 정의 |
| `experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/metacog_lite_v0/weights.py` | support predicate, raw weight, batch normalization, critic-only scope 구현 |
| `experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/metacog_lite_v0/agent.py` | `omega_critic * td_error^2` critic loss 적용부 |
| `experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/metacog_lite_v0/replay.py` | replay action index와 context 기반 후보 검색 |
| `experiment_FPP/reports/fpp_main_experiment_results_20260604.md` | FPP DDPG/TD3/SAC + SABER 주요 결과 |
| `experiment_FetchSlide/reports/FetchSlide/agent_family_results.md` | FetchSlide DDPG/TD3/SAC family 비교 |
| `metaworld_Drawer_Open_TD3/README.md` | DrawerOpen TD3 critic-only, actor-scope, random/fixed/shuffled controls |

## 10. 남은 정리 과제

논문 작성 전에 아래를 확정하는 것이 좋다.

1. `SABER` acronym을 공식적으로 정할지 여부
   추천: **Support-Aware Bellman Error Reweighting** 또는 **Support-Aware Bellman Error Reliability**. 현재 코드상 공식 확장은 없으므로 논문에서 새로 정의해야 한다.

2. 메인 claim 범위
   추천: “full MetaCog-RL”이 아니라 “MetaCog-derived support-aware critic-side reliability weighting”으로 제한한다.

3. DrawerOpen Random-Support 해석
   Random-Support가 강한 결과는 숨기지 말고, support-conditioned critic reweighting이 exploration/stabilization 효과도 가진다는 discussion으로 처리한다.

4. runtime overhead
   SABER는 wall-clock이 느릴 수 있으므로 sample efficiency와 runtime efficiency를 별도 축으로 제시한다.

5. support vs TD-error 분리 분석
   가능하면 support-positive 샘플과 high TD-error 샘플의 overlap/correlation을 추가로 계산하면 “PER 변형이 아니다”라는 주장이 강해진다.

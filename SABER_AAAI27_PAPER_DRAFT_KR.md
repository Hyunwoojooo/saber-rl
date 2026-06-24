# SABER AAAI-27 논문 한국어 초안

작성일: 2026-06-23  
상태: 한국어 1차 초안. 영어 LaTeX 변환 전 원고.  
권장 제목: **SABER: Support-Aware Bellman Error Reweighting for Sparse Goal-Conditioned Reinforcement Learning**

## 작성 메모

이 문서는 논문 본문으로 옮기기 위한 한국어 초안이다. 기술 용어는 영어 표현을 유지했다. AAAI-27 제출용 최종본에서는 anonymous submission 규칙에 맞춰 GitHub 경로, 내부 실험명, 작성자 식별 가능 표현을 제거해야 한다.

현재 논문 방향은 다음 thesis로 고정한다.

> HER creates targets; SABER decides which Bellman errors the critic should trust more.

SABER의 논문용 정의는 다음이다.

> SABER는 sparse-reward goal-conditioned HER setting에서 HER가 만들어낸 Bellman target들을 모두 동일하게 신뢰하지 않고, local replay-action support가 있는 target의 critic Bellman error를 더 강하게 학습시키는 방법이다.

논문에서 피해야 할 주장:

- SABER is a general-purpose RL algorithm.
- SABER solves extrapolation error.
- SABER is a new HER method.
- SABER always improves over HER.
- SABER proves support-positive samples always have lower Bellman target noise.

논문에서 유지해야 할 안전한 주장:

- SABER is a support-aware critic-side Bellman error reweighting method.
- SABER keeps replay sampling and actor objective unchanged in the main variant.
- SABER is compatible with multiple off-policy actor-critic backbones.
- The strongest evidence is FetchPickAndPlace across DDPG, TD3, and SAC.
- FetchSlide and DrawerOpen show both benefits and limitations.

---

# Abstract

Sparse-reward goal-conditioned reinforcement learning은 성공 reward가 드물기 때문에 off-policy actor-critic 방법만으로는 효율적인 학습이 어렵다. Hindsight Experience Replay (HER)는 실패 trajectory를 다른 goal의 성공 경험으로 relabel하여 학습 가능한 Bellman target을 늘리는 강력한 방법이다. 그러나 HER는 어떤 goal로 relabel할지를 정할 뿐, relabeled Bellman update가 현재 policy action 근방에서 얼마나 reliable한지는 보장하지 않는다. 본 논문은 이 문제를 **hindsight Bellman target reliability gap**으로 정의한다.

우리는 이 gap을 줄이기 위해 **SABER (Support-Aware Bellman Error Reweighting)**를 제안한다. SABER는 각 HER training sample에 대해 local state-goal-horizon context를 구성하고, replay buffer 안에서 현재 policy action이 local replay-action support를 갖는지를 판정한다. Support-positive sample에는 더 큰 critic weight를 부여하고, batch-normalized weight를 critic Bellman error에만 곱한다. Actor objective와 replay sampling distribution은 baseline과 동일하게 유지하여 support signal의 효과를 critic fitting으로 분리한다.

FetchPickAndPlace에서 SABER는 DDPG+HER, TD3+HER, SAC+HER 모두에서 final success와 success AUC를 개선했다. FetchSlide에서는 TD3와 SAC에서 개선을 보였지만 DDPG에서는 개선하지 못해 task/backbone-dependent limitation을 드러냈다. MetaWorld DrawerOpen에서는 critic-only SABER가 TD3+HER보다 더 많은 seed를 성공시켰지만, random-support control도 강하게 나타나 support-aware weighting과 reweighting-induced regularization의 경계를 추가로 분석해야 함을 보여준다. 이 결과들은 SABER가 HER의 대체물이 아니라, HER가 만든 Bellman targets 중 critic이 더 신뢰할 supervision을 구분하는 간단한 critic-side objective임을 보여준다.

# 1. Introduction

Goal-conditioned reinforcement learning은 agent가 하나의 고정된 reward가 아니라 다양한 goal을 달성하도록 학습하는 문제를 다룬다. 특히 robotic manipulation에서는 pushing, sliding, pick-and-place처럼 성공 조건이 명확하지만 reward가 sparse한 task가 많다. 이러한 setting에서는 대부분의 trajectory가 실패로 끝나기 때문에, naive off-policy actor-critic은 critic이 충분한 positive signal을 얻기 어렵다.

Hindsight Experience Replay (HER)는 이 문제를 해결하는 대표적인 방법이다. HER는 agent가 원래 목표를 달성하지 못했더라도, 실제로 도달한 상태를 새로운 goal로 relabel하여 실패 경험을 학습 가능한 transition으로 바꾼다. 이 방식은 sparse reward에서도 goal-conditioned Bellman target을 대량으로 만들 수 있게 해준다.

그러나 HER가 해결하지 않는 문제가 있다. HER는 relabeled goal을 만들어 Bellman target의 quantity를 늘리지만, 그 target이 현재 policy action 근방에서 얼마나 reliable한지는 판단하지 않는다. Off-policy actor-critic에서 critic은 replay action을 중심으로 Bellman regression을 수행하고, actor는 critic이 높게 평가하는 current policy action을 따라간다. 이때 현재 policy action이 local replay action support 바깥에 있다면, critic은 그 action region에서 extrapolation에 의존할 수 있다. HER로 relabel된 transition이라고 해서 이 문제가 자동으로 사라지지는 않는다.

본 논문의 핵심 관찰은 다음이다.

> HER creates useful training goals, but it does not guarantee that every relabeled Bellman update is equally reliable for the current policy.

우리는 이 문제를 **hindsight Bellman target reliability gap**으로 부른다. 같은 HER batch 안에서도 어떤 sample은 local state-goal-horizon context에서 현재 policy action이 replay buffer에 충분히 support되어 있어 critic fitting target으로 더 신뢰할 수 있고, 어떤 sample은 그렇지 않다. 따라서 모든 relabeled Bellman error를 동일한 weight로 fitting하는 것은 critic에게 불균일한 reliability의 supervision을 같은 강도로 주는 셈이다.

이를 해결하기 위해 우리는 **SABER: Support-Aware Bellman Error Reweighting**을 제안한다. SABER는 각 training sample에 대해 local replay-action support를 계산하고, support-positive sample의 critic Bellman error를 더 크게 반영한다. 중요한 점은 SABER가 HER를 대체하지 않는다는 것이다. HER는 training goal을 만들고, SABER는 그 goal-conditioned Bellman update 중 critic이 더 신뢰할 update를 구분한다.

SABER의 main variant는 critic-only이다. 즉, support-derived weight는 critic TD loss에만 적용되며 actor objective와 replay sampling distribution은 baseline과 동일하게 유지된다. 이 설계는 SABER의 효과를 actor distribution bias나 prioritized replay effect가 아니라 critic Bellman fitting의 reliability weighting으로 해석할 수 있게 한다.

본 논문의 contribution은 다음과 같다.

1. Sparse-reward goal-conditioned HER setting에서 relabeled Bellman target의 quantity와 reliability가 다르다는 문제를 제기하고, 이를 hindsight Bellman target reliability gap으로 정식화한다.
2. Local replay-action support를 critic-side Bellman fitting reliability signal로 사용하는 SABER objective를 제안한다.
3. SABER를 DDPG, TD3, SAC와 결합 가능한 backbone-compatible critic reweighting method로 정의하고, actor update와 replay sampling을 유지하는 critic-only 설계의 장점을 보인다.
4. FetchPickAndPlace, FetchSlide, MetaWorld DrawerOpen 실험을 통해 SABER의 장점과 한계를 함께 제시한다. 특히 FetchPickAndPlace에서는 DDPG/TD3/SAC 모두에서 HER baseline보다 높은 success AUC와 final success를 보인다.
5. PER-style weighting, full PER, actor-scope variants, random/shuffled controls와 비교하여 SABER가 단순 TD-error prioritization과 다른 신호를 사용함을 분석한다.

# 2. Related Work

## 2.1 Goal-Conditioned RL and HER

Goal-conditioned RL은 policy와 value function이 상태뿐 아니라 goal을 조건으로 받도록 학습한다. Sparse reward manipulation에서는 성공 signal이 드물기 때문에 goal relabeling이 특히 중요하다. HER는 실패 trajectory에서 실제로 달성된 state를 hindsight goal로 사용하여 sparse reward 문제를 완화한다. HER의 핵심 역할은 training target을 늘리는 것이다.

SABER는 HER와 경쟁하는 방법이 아니다. HER가 "어떤 goal로 Bellman target을 만들 것인가"를 다룬다면, SABER는 "그렇게 만들어진 Bellman error를 critic이 얼마나 강하게 fitting할 것인가"를 다룬다. 따라서 SABER는 HER의 relabeling distribution을 바꾸는 방법이 아니라, HER 이후 critic update의 reliability weighting을 조정하는 방법이다.

## 2.2 Off-Policy Actor-Critic and Critic Error

DDPG, TD3, SAC와 같은 off-policy actor-critic 방법은 continuous control에서 널리 사용된다. 이 방법들은 replay buffer를 사용해 sample efficiency를 높이지만, critic approximation error와 actor exploitation 문제가 함께 발생할 수 있다. TD3는 twin critics, target policy smoothing, delayed policy update를 통해 overestimation과 function approximation error를 줄인다. SAC는 entropy-regularized objective를 통해 exploration과 robustness를 높인다.

SABER는 backbone 자체를 바꾸지 않는다. DDPG, TD3, SAC의 Bellman target 계산은 유지하되, critic regression loss에 sample-wise support-derived weight를 곱한다. 따라서 SABER는 새로운 actor-critic algorithm이라기보다, off-policy actor-critic에 부착되는 critic-side Bellman error reweighting layer로 보는 것이 정확하다.

## 2.3 Prioritized Replay and TD-Error Weighting

Prioritized Experience Replay (PER)는 TD-error가 큰 transition을 더 자주 sampling하여 learning progress가 클 것으로 예상되는 sample을 재사용한다. PER-style loss weighting도 TD-error magnitude를 기준으로 critic loss를 조정한다.

SABER는 TD-error가 큰 sample을 우선시하지 않는다. SABER의 기준 신호는 local replay-action support이다. 즉, TD-error magnitude가 아니라 현재 policy action이 해당 state-goal-horizon context의 replay action support 안에 있는지를 본다. 또한 main SABER는 replay sampling distribution을 바꾸지 않고, critic loss weight만 바꾼다.

이 차이를 논문에서 명확히 해야 한다.

> SABER does not prioritize surprise; it prioritizes supported Bellman supervision.

## 2.4 Dataset Support and Extrapolation

Offline RL에서는 learned policy가 dataset support 밖의 action을 선택할 때 critic extrapolation error가 커질 수 있다는 문제가 잘 알려져 있다. BCQ, CQL 등은 이러한 distribution shift와 OOD action overestimation을 줄이기 위해 policy constraint나 conservative value learning을 사용한다.

SABER는 offline RL 방법이 아니다. SABER는 online off-policy HER setting에서 action을 제한하지 않으며, policy를 dataset support 안으로 강제로 묶지도 않는다. 대신 replay support가 충분한 Bellman update를 critic이 더 강하게 fitting하도록 weight를 조정한다. 따라서 SABER는 offline RL의 support mismatch 문제와 개념적으로 연결되지만, 알고리즘적 개입은 훨씬 좁다.

## 2.5 MetaCog-RL Lineage

SABER는 MetaCog-RL의 full diagnosis framework를 구현한 것이 아니다. 원래 MetaCog-RL은 RL 실패 원인을 ignorance, incompetence, impossibility로 나누고, world model ensemble과 competence gap을 통해 EXPLORE/PRACTICE/ROBUST 개입을 선택하는 프레임워크다.

SABER로 이어진 부분은 그중 PRACTICE/incompetence 축이다. 원래 PRACTICE는 모델이 어느 정도 알고 있고 정책이 아직 못하는 영역을 더 연습시키는 개입이다. SABER는 이를 sparse HER actor-critic setting으로 축소하여, world model 기반 competence gap 대신 replay local support predicate를 사용하고, focused practice 대신 critic Bellman loss reweighting을 적용한다.

따라서 본 논문에서 SABER는 MetaCog-RL 전체의 구현이 아니라, MetaCog의 PRACTICE intuition에서 파생된 conservative model-free instantiation으로 위치시킨다.

# 3. Preliminaries

## 3.1 Goal-Conditioned MDP

Goal-conditioned MDP를 다음과 같이 둔다.

$$
\mathcal M = (\mathcal S, \mathcal A, \mathcal G, P, r_g, \gamma).
$$

여기서 \(\mathcal S\)는 state space, \(\mathcal A\)는 continuous action space, \(\mathcal G\)는 goal space, \(P\)는 transition dynamics, \(r_g\)는 goal-conditioned reward, \(\gamma\)는 discount factor다.

Replay buffer에는 transition이 다음 형태로 저장된다.

$$
\tau_i = (o_i, a_i, o'_i, g_i^{env}, d_i, h_i),
$$

여기서 \(o_i\)는 observation, \(a_i\)는 replay action, \(o'_i\)는 next observation, \(g_i^{env}\)는 original environment goal, \(d_i\)는 termination indicator, \(h_i\)는 horizon 또는 time-to-goal context다.

HER를 사용할 때 training goal은 original goal이 아니라 relabeling distribution에서 샘플링된다.

$$
g_i^{tr} \sim \mathcal H(\tau_i).
$$

HER를 사용하지 않는 경우에는 \(g_i^{tr}=g_i^{env}\)이다. 따라서 SABER는 HER 자체를 필수로 가정하기보다, training goal \(g_i^{tr}\)이 주어진 상태에서 작동하는 critic weighting method로 정의할 수 있다. 다만 본 논문의 empirical focus는 sparse-reward HER benchmark이다.

Goal-conditioned input을 다음처럼 둔다.

$$
x_i = (o_i, g_i^{tr}).
$$

Actor와 critic은 각각 다음과 같다.

$$
\pi_\theta(x_i)=\pi_\theta(o_i,g_i^{tr}),
$$

$$
Q_\phi(x_i,a_i)=Q_\phi(o_i,g_i^{tr},a_i).
$$

## 3.2 Bellman Regression in Off-Policy Actor-Critic

DDPG-style deterministic actor-critic의 Bellman target은 다음과 같이 쓸 수 있다.

$$
y_i =
r(o_i,a_i,o'_i,g_i^{tr})
+ \gamma(1-d_i)
Q_{\bar\phi}
(o'_i,g_i^{tr},\pi_{\bar\theta}(o'_i,g_i^{tr})).
$$

Baseline critic loss는 모든 sample을 동일 weight로 fitting한다.

$$
\mathcal L_Q^{base}(\phi)
=
\frac{1}{B}\sum_{i=1}^{B}
\left(Q_\phi(x_i,a_i)-y_i\right)^2.
$$

TD3와 SAC에서는 target 계산 방식이 backbone에 맞게 달라지지만, critic이 Bellman target을 regression한다는 구조는 같다. SABER는 이 regression loss의 sample weight만 바꾼다.

# 4. Motivation: Target Quantity Is Not Target Reliability

HER는 sparse reward에서 target quantity를 늘린다. 하지만 target quantity와 target reliability는 같은 개념이 아니다.

같은 mini-batch 안의 두 HER-relabeled transition을 생각하자. 두 transition 모두 reward를 재계산할 수 있고 Bellman target \(y_i\)를 만들 수 있다. 그러나 첫 번째 transition에서는 현재 actor action \(\pi_\theta(o_i,g_i^{tr})\)이 같은 local state-goal-horizon context 안에서 replay buffer에 여러 번 관측된 action family에 속할 수 있다. 두 번째 transition에서는 현재 actor action이 그 local context의 replay action support 바깥에 있을 수 있다.

두 경우 모두 Bellman target은 존재한다. 그러나 critic이 해당 target을 현재 policy improvement에 유용한 supervision으로 fitting할 때의 reliability는 다를 수 있다. 첫 번째 경우는 local interpolation에 가깝고, 두 번째 경우는 extrapolation risk가 크다.

SABER의 기본 가설은 다음이다.

> Replay support가 충분한 transition은 critic Bellman fitting에 더 reliable한 supervision을 제공한다. 따라서 support-positive transition의 critic TD loss를 더 크게 반영하면 sparse manipulation에서 value learning이 안정화되고 sample efficiency가 좋아질 수 있다.

이 가설은 theorem으로 주장하지 않는다. 본 논문에서는 이를 reliability-weighted Bellman regression interpretation으로 제시하고, 실험적으로 검증한다.

# 5. SABER

## 5.1 Overview

SABER는 각 training sample마다 local replay-action support를 계산하고, support-positive sample의 critic Bellman error에 더 큰 weight를 준다. Main variant는 critic-only이다.

절차는 다음과 같다.

1. HER 또는 baseline replay sampling으로 mini-batch를 가져온다.
2. 각 sample \(i\)에 대해 local context \(c_i\)를 만든다.
3. Replay buffer에서 같은 또는 유사한 context의 transition set \(\mathcal L_i\)를 찾는다.
4. Current policy action \(a_i^\pi=\pi_\theta(o_i,g_i^{tr})\)이 local replay action support 안에 있는지 확인한다.
5. Support-positive이면 \(w_i=1+\lambda_S\), 아니면 \(w_i=1\)을 부여한다.
6. Batch normalization으로 평균 weight를 1로 맞춘다.
7. Normalized weight를 critic Bellman error에만 곱한다.

## 5.2 Local Context

각 train item \(i\)에 대해 task-specific local context를 정의한다.

$$
c_i = C(o_i, g_i^{tr}, h_i).
$$

이 context는 observation, training goal, horizon information을 coarse bin 또는 hash로 변환한 것이다. 예를 들어 FetchPickAndPlace에서는 hand-object distance, object-goal distance, object height, gripper gap, horizon bin 등이 context에 들어갈 수 있다. FetchSlide/FetchPush에서는 object-goal alignment, contact/slide phase, velocity-related bin 등이 사용될 수 있다. DrawerOpen에서는 hand-handle distance, handle-goal distance, handle position, gripper state, phase, horizon bin 등이 사용된다.

중요한 점은 local context가 exact state matching이 아니라 support lookup을 위한 coarse abstraction이라는 점이다. SABER는 동일한 continuous state를 다시 찾으려는 것이 아니라, critic fitting reliability를 추정할 수 있을 정도의 local neighborhood를 구성한다.

## 5.3 Local Replay-Action Support

Sample \(i\)에 대한 local replay pool을 다음과 같이 둔다.

$$
\mathcal L_i =
\{j \in \mathcal D:
C(o_j,g_i^{tr},h_j) \approx C(o_i,g_i^{tr},h_i)\}.
$$

여기서 \(\mathcal D\)는 replay buffer다. 같은 local context에서 관측된 replay action들을 action bin으로 변환한다.

$$
\mathcal B_i = \{b(a_j): j\in \mathcal L_i\}.
$$

Current policy action과 그 bin은 다음과 같다.

$$
a_i^\pi = \pi_\theta(o_i,g_i^{tr}),
$$

$$
b_i^\pi = b(a_i^\pi).
$$

SABER는 \(b_i^\pi\)가 \(\mathcal B_i\)에 포함되는지를 통해 current policy action이 local replay support 안에 있는지 판정한다.

## 5.4 Support Indicator

Support-positive indicator는 다음과 같이 정의한다.

$$
\psi_i =
\mathbf 1
\left[
|\mathcal L_i|\ge N_{\min}
\land
|\mathcal A_i^{sup}|\ge M_{\min}
\land
b_i^\pi \in \mathcal B_i
\right].
$$

구현상 주요 실험에서 사용한 조건은 다음과 같다.

$$
\text{local\_count}_i \ge N_{\text{local\_min}},
$$

$$
\text{available\_support}_i = \min(\text{local\_count}_i, M_B),
$$

$$
\text{available\_support}_i \ge M_{Q\_min},
$$

$$
b(a_i^\pi) \in \mathcal B_i.
$$

대표 threshold는 다음과 같다.

- \(M_B=16\)
- \(M_{Q\_min}=4\)
- \(N_{\text{local\_min}}=4\)
- \(N_{\text{gap\_min}}=128\)
- \(\lambda_S=1.0\)
- `weighting_mode=batch_norm`
- `weight_scope=critic_only`

## 5.5 Weight Construction

Raw weight는 다음과 같다.

$$
w_i^{raw} = 1 + \lambda_S \psi_i.
$$

Batch-normalized weight는 다음과 같다.

$$
\Omega_i =
\frac{w_i^{raw}}
{\frac{1}{B}\sum_{j=1}^{B} w_j^{raw} + \epsilon}.
$$

이 normalization은 batch 평균 learning rate scale을 유지하면서 support-positive sample과 support-negative sample 사이의 상대 weight만 조정한다.

Critic-only SABER에서는 다음과 같이 둔다.

$$
\Omega_i^{critic} = \Omega_i,
$$

$$
\Omega_i^{actor} = 1.
$$

따라서 actor objective는 baseline과 동일하다.

## 5.6 Critic Objective

SABER critic loss는 다음이다.

$$
\mathcal L_Q^{SABER}(\phi)
=
\frac{1}{B}\sum_{i=1}^{B}
\Omega_i
\left(Q_\phi(x_i,a_i)-y_i\right)^2.
$$

Actor objective는 backbone의 baseline actor objective를 유지한다. DDPG/TD3에서는 deterministic policy gradient 형태를 유지하고, SAC에서는 entropy-regularized actor objective를 유지한다.

이 설계는 SABER의 개입을 다음 한 문장으로 요약한다.

> SABER keeps the targets and sampling fixed, but changes how strongly the critic fits each Bellman error.

## 5.7 Algorithm Sketch

```text
Algorithm: SABER critic update

Input: replay buffer D, actor pi_theta, critic Q_phi, target networks,
       HER relabeling distribution H, support weight lambda_S

1. Sample a mini-batch from D using the baseline replay sampler.
2. For each transition i, sample or assign training goal g_i^tr.
3. Compute the backbone Bellman target y_i.
4. For each sample i:
   a. Construct local context c_i = C(o_i, g_i^tr, h_i).
   b. Retrieve local replay pool L_i from D.
   c. Compute current policy action a_i^pi = pi_theta(o_i, g_i^tr).
   d. Set psi_i = 1 if a_i^pi is supported by local replay actions;
      otherwise set psi_i = 0.
   e. Set raw weight w_i = 1 + lambda_S psi_i.
5. Normalize weights within the batch.
6. Update the critic using weighted Bellman regression.
7. Update the actor using the unchanged baseline actor objective.
```

# 6. Theoretical Interpretation

SABER can be interpreted as reliability-weighted Bellman regression. Suppose the observed Bellman target can be decomposed as

$$
y_i = y_i^\star + \epsilon_i,
$$

where \(y_i^\star\) is the ideal target and \(\epsilon_i\) is target or approximation noise induced by off-policy bootstrapping, function approximation, and support mismatch. SABER assumes that support-positive samples tend to have lower effective fitting risk than support-negative samples:

$$
\operatorname{Var}(\epsilon_i \mid \psi_i=1)
<
\operatorname{Var}(\epsilon_i \mid \psi_i=0).
$$

This is not a theorem in the current paper. It is an interpretation and empirical hypothesis. Under this view, SABER does not claim that unsupported samples are useless. It only assigns stronger critic fitting weight to samples whose current policy action is locally supported by replay actions.

This interpretation also explains why critic-only is the cleanest main variant. If support weights are applied to the actor loss, the method may change which state-goal regions the actor is encouraged to exploit, making it harder to separate critic reliability from actor distribution bias. Critic-only SABER isolates the mechanism: support affects Bellman regression, not direct policy optimization.

# 7. Experiments

## 7.1 Research Questions

본 실험은 다음 질문에 답하기 위해 설계한다.

**RQ1.** SABER는 sparse-reward HER manipulation에서 baseline HER보다 sample efficiency와 final success를 개선하는가?

**RQ2.** SABER는 DDPG, TD3, SAC와 같은 여러 off-policy actor-critic backbone에 호환되는가?

**RQ3.** SABER의 이득은 TD-error/PER-style weighting으로 설명될 수 있는가?

**RQ4.** Critic-only weighting이 actor-only 또는 actor+critic weighting보다 더 깨끗한 mechanism을 제공하는가?

**RQ5.** SABER의 한계는 무엇이며, runtime overhead와 task/backbone-dependent failure는 어떻게 나타나는가?

## 7.2 Tasks and Backbones

주요 benchmark는 sparse-reward goal-conditioned manipulation tasks다.

- FetchPickAndPlace: primary evidence. DDPG+HER, TD3+HER, SAC+HER 모두에서 SABER를 비교한다.
- FetchSlide: secondary evidence. SAC/TD3 positive case와 DDPG negative case를 함께 보고한다.
- FetchPush: incomplete rows가 있으므로 appendix 또는 보조 evidence로 제한한다.
- MetaWorld DrawerOpen: external stress-test와 actor/random/shuffled ablation 분석에 사용한다.

Backbone은 DDPG, TD3, SAC를 사용한다. 모든 main SABER variant는 critic-only support weighting을 사용한다.

## 7.3 Metrics

주요 metric은 다음이다.

- Final success: 마지막 evaluation의 success rate.
- Best success: training 중 최대 evaluation success.
- Success AUC: evaluation success curve의 area under curve.
- First hit threshold: 특정 success threshold를 처음 넘은 step.
- Runtime steps/sec: wall-clock efficiency.
- Drop best-final: best success와 final success의 차이.

Sample efficiency와 wall-clock efficiency는 분리해서 해석한다. SABER는 support lookup 때문에 runtime overhead가 있으므로, success AUC가 개선되더라도 wall-clock 기준으로 항상 유리하다고 주장하지 않는다.

## 7.4 Main Result: FetchPickAndPlace

FetchPickAndPlace는 현재 가장 강한 main evidence다. 모든 결과는 1M environment steps, seeds 1..5 기준이다.

| Backbone | Method | Final success | Best success | Success AUC | First hit >=0.90 | Steps/sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DDPG | HER | 0.940 +/- 0.042 | 1.000 +/- 0.000 | 0.539 +/- 0.125 | 600k | 65.03 +/- 0.10 |
| DDPG | HER+SABER | 0.960 +/- 0.042 | 1.000 +/- 0.000 | 0.609 +/- 0.038 | 540k | 41.72 +/- 0.20 |
| TD3 | HER | 0.840 +/- 0.188 | 0.880 +/- 0.144 | 0.372 +/- 0.032 | 980k | 65.33 +/- 0.13 |
| TD3 | HER+SABER | 0.970 +/- 0.045 | 0.990 +/- 0.022 | 0.442 +/- 0.035 | 840k | 41.25 +/- 0.59 |
| SAC | HER | 0.790 +/- 0.288 | 0.910 +/- 0.124 | 0.463 +/- 0.077 | 800k | 38.93 +/- 0.07 |
| SAC | HER+SABER | 0.990 +/- 0.022 | 1.000 +/- 0.000 | 0.565 +/- 0.030 | 750k | 29.10 +/- 0.12 |

FPP에서는 SABER가 세 backbone 모두에서 final success와 success AUC를 개선한다. DDPG에서는 final success가 0.940에서 0.960으로 소폭 상승하고 AUC가 0.539에서 0.609로 오른다. TD3에서는 final success가 0.840에서 0.970으로, AUC가 0.372에서 0.442로 개선된다. SAC에서는 final success가 0.790에서 0.990으로, AUC가 0.463에서 0.565로 개선된다.

이 결과는 SABER가 하나의 특정 actor-critic implementation에만 붙는 trick이 아니라, 여러 off-policy actor-critic backbone에 호환되는 critic-side objective임을 보여준다. 다만 runtime은 감소한다. 예를 들어 DDPG에서는 65.03 steps/sec에서 41.72 steps/sec로, SAC에서는 38.93 steps/sec에서 29.10 steps/sec로 낮아진다. 따라서 FPP 결과는 sample efficiency 관점에서는 강하지만, wall-clock cost를 함께 보고해야 한다.

## 7.5 Comparison with PER-Style Weighting

FPP DDPG setting에서 PER-style 및 full PER baseline과 비교한다.

| Method | Final success | Best success | Success AUC | Steps/sec |
| --- | ---: | ---: | ---: | ---: |
| DDPG+HER | 0.940 +/- 0.042 | 1.000 +/- 0.000 | 0.539 +/- 0.125 | 65.03 +/- 0.10 |
| DDPG+HER+SABER | 0.960 +/- 0.042 | 1.000 +/- 0.000 | 0.609 +/- 0.038 | 41.72 +/- 0.20 |
| DDPG+HER+PER-style | 0.930 +/- 0.104 | 0.990 +/- 0.022 | 0.568 +/- 0.069 | 61.08 +/- 0.09 |
| DDPG+HER+PER-full | 0.780 +/- 0.438 | 0.840 +/- 0.358 | 0.460 +/- 0.238 | 58.02 +/- 0.14 |

PER-style weighting은 AUC에서 HER보다 높지만 SABER보다 낮고, final success는 HER보다도 낮다. Full PER는 seed 5 collapse 때문에 평균 final success가 크게 낮아진다. 이 결과는 SABER의 이득이 단순히 TD-error 기반 weighting이나 prioritized replay로 설명되기 어렵다는 근거가 된다.

다만 이 주장을 더 강하게 만들려면 support-positive sample과 high TD-error sample의 overlap/correlation 분석이 필요하다.

TODO:

- \(\operatorname{corr}(\psi_i, |\delta_i|)\) 계산.
- support-positive sample과 top-k TD-error sample overlap 계산.
- support bin별 update 전후 TD-error 감소량 분석.

## 7.6 Secondary Result: FetchSlide

FetchSlide는 SABER의 장점과 한계를 동시에 보여주는 중요한 secondary result다. 모든 결과는 1M steps, seeds 1..5 기준이다.

| Backbone | Method | Final success | Best success | Success AUC | Last-quarter AUC | Steps/sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DDPG | HER | 0.600 +/- 0.138 | 0.740 +/- 0.037 | 0.313 +/- 0.012 | 0.519 +/- 0.023 | 119.08 +/- 0.13 |
| DDPG | HER+SABER | 0.520 +/- 0.093 | 0.740 +/- 0.097 | 0.313 +/- 0.030 | 0.476 +/- 0.060 | 26.85 +/- 0.29 |
| TD3 | HER | 0.630 +/- 0.163 | 0.770 +/- 0.112 | 0.281 +/- 0.059 | 0.594 +/- 0.108 | 111.43 +/- 0.29 |
| TD3 | SABER | 0.690 +/- 0.073 | 0.830 +/- 0.068 | 0.323 +/- 0.039 | 0.627 +/- 0.064 | 36.43 +/- 0.26 |
| SAC | HER | 0.820 +/- 0.103 | 0.950 +/- 0.045 | 0.500 +/- 0.027 | 0.775 +/- 0.044 | 66.78 +/- 0.06 |
| SAC | SABER | 0.840 +/- 0.073 | 0.950 +/- 0.032 | 0.535 +/- 0.020 | 0.806 +/- 0.023 | 30.57 +/- 0.17 |

FetchSlide에서 TD3+SABER는 TD3+HER보다 final success, best success, success AUC, last-quarter AUC를 모두 개선한다. SAC+SABER도 SAC+HER보다 final success와 AUC가 소폭 개선된다. 그러나 DDPG+SABER는 DDPG+HER보다 final success와 last-quarter AUC가 낮고, success AUC는 거의 같다.

이 결과는 SABER가 모든 task/backbone 조합에서 항상 개선된다는 주장을 피해야 함을 보여준다. 더 정확한 해석은 다음이다.

> SABER is compatible with multiple off-policy actor-critic backbones and shows consistent gains on FetchPickAndPlace, while FetchSlide reveals task/backbone-dependent limitations.

또한 FetchSlide에서는 runtime overhead가 훨씬 크다. DDPG+HER는 119.08 steps/sec인 반면 DDPG+SABER는 26.85 steps/sec다. TD3와 SAC에서도 SABER는 약 2배에서 3배 이상 느리다. 따라서 SABER는 sample-efficient할 수 있지만 항상 wall-clock-efficient하지는 않다.

## 7.7 External Stress-Test: MetaWorld DrawerOpen

MetaWorld DrawerOpen은 Fetch tasks와 다른 dynamics와 success structure를 가진 stress-test로 사용한다. 여기서는 TD3+HER와 TD3+HER+B3-Critic-Only를 2M steps, seeds 1..5로 비교한다.

| Method | Final success | Best success | Success AUC | Hit90 | Steps/sec |
| --- | ---: | ---: | ---: | --- | ---: |
| TD3+HER 2M | 0.190 +/- 0.425 | 0.210 +/- 0.442 | 0.127 +/- 0.283 | 740k, 1/5 seeds | 41.55 |
| TD3+HER+SABER 2M | 0.590 +/- 0.539 | 0.620 +/- 0.522 | 0.362 +/- 0.338 | 803k +/- 256k, 3/5 seeds | 21.57 |

DrawerOpen에서 SABER는 TD3+HER보다 final success, best success, AUC, Hit90 seed count를 개선한다. 그러나 variance가 매우 크다. SABER도 seed 2에서 실패했고, seed 3은 weak nonzero signal만 보인다. 따라서 DrawerOpen은 main proof라기보다 외부 stress-test로 해석해야 한다.

## 7.8 Actor-Scope Ablation

DrawerOpen 2M actor-scope ablation은 critic-only 설계가 mechanism isolation 측면에서 가장 적절함을 보여준다.

| Method | Final success | Best success | Success AUC | Hit90 |
| --- | ---: | ---: | ---: | --- |
| B3-Critic-Only | 0.590 +/- 0.539 | 0.620 +/- 0.522 | 0.361 +/- 0.338 | 3/5 seeds |
| B3-Actor-Only | 0.400 +/- 0.548 | 0.410 +/- 0.539 | 0.249 +/- 0.341 | 2/5 seeds |
| B3-Actor+Critic | 0.600 +/- 0.548 | 0.610 +/- 0.534 | 0.432 +/- 0.395 | 3/5 seeds |

Actor+Critic은 aggregate AUC에서 critic-only보다 높지만, high variance 아래에서 clean dominance를 보이지 않는다. Actor-only는 더 약하다. Critic-only는 actor update를 바꾸지 않기 때문에, 논문의 main mechanism인 Bellman fitting reliability를 가장 깨끗하게 보여준다.

## 7.9 Random and Shuffled Controls

DrawerOpen에서는 Random-Support가 의외로 강하게 나타난다.

| Method | Final success | Best success | Success AUC | Hit90 |
| --- | ---: | ---: | ---: | --- |
| B3-Critic-Only | 0.590 +/- 0.539 | 0.620 +/- 0.522 | 0.361 +/- 0.338 | 3/5 seeds |
| B3-Random-Support | 0.600 +/- 0.548 | 0.600 +/- 0.548 | 0.426 +/- 0.392 | 3/5 seeds |
| B3-Random-Fixed-Count | 0.200 +/- 0.447 | 0.200 +/- 0.447 | 0.144 +/- 0.322 | 1/5 seeds |
| B3-Shuffled-Weight | 0.000 +/- 0.000 | 0.010 +/- 0.022 | 0.00005 +/- 0.00011 | 0/5 seeds |

이 결과는 두 가지를 동시에 말한다.

첫째, Shuffled-Weight가 모든 seed에서 실패했다는 점은 sample-weight alignment가 중요할 수 있음을 보여준다. 같은 support lookup path와 weight distribution을 유지하더라도, weight를 batch 안에서 섞어 sample과 weight의 대응을 깨면 성능이 무너진다.

둘째, Random-Support가 Critic-Only와 비슷하거나 더 높은 AUC를 보였다는 점은 exact structured support rule만이 유일한 원인이라는 강한 claim을 약화시킨다. DrawerOpen에서는 support-conditioned critic reweighting과 reweighting-induced regularization이 함께 작동했을 가능성이 있다.

따라서 이 결과는 숨기지 말고 limitation/discussion에 넣어야 한다.

권장 해석:

> In the main sparse HER tasks, support-aware critic weighting improves Bellman fitting, and shuffled controls show that sample-weight alignment can matter. DrawerOpen suggests that in some tasks, support-conditioned or even randomized reweighting may also act as a stabilizing regularizer.

# 8. Discussion

## 8.1 What SABER Is

SABER는 HER의 대체물이 아니다. HER는 sparse reward에서 학습 가능한 goal-conditioned targets를 만든다. SABER는 HER가 만든 target 중 어떤 Bellman error를 critic이 더 강하게 fitting할지 정한다.

SABER는 PER의 변형도 아니다. PER는 TD-error priority를 기반으로 sampling distribution을 바꾼다. SABER는 sampling distribution을 유지하고, TD-error magnitude가 아니라 local replay-action support를 기준으로 critic loss weight를 조정한다.

SABER는 full MetaCog-RL 구현도 아니다. SABER는 MetaCog-RL의 PRACTICE/incompetence idea를 sparse HER actor-critic에 맞게 축소한 critic-side instantiation이다.

## 8.2 Why Critic-Only

Critic-only design은 논문적으로 중요하다. Actor loss까지 support weight를 적용하면, support signal이 critic fitting reliability를 반영하는지 actor objective shaping을 반영하는지 분리하기 어렵다. Critic-only SABER는 actor objective와 replay sampling을 baseline과 동일하게 유지하므로, support signal의 역할을 Bellman regression reliability로 해석하기 쉽다.

## 8.3 Runtime Cost

SABER는 local support lookup 때문에 runtime overhead가 있다. FPP에서는 performance gain과 함께 moderate slowdown이 나타났고, FetchSlide에서는 slowdown이 훨씬 크다. 따라서 SABER를 "항상 더 efficient하다"고 주장하면 안 된다.

정확한 claim은 다음이다.

> SABER can improve sample efficiency, but its current local support implementation is not always wall-clock efficient.

이 limitation은 future work로 자연스럽게 이어진다. 더 빠른 support index, approximate nearest neighbor lookup, learned support model, vectorized context matching 등이 후속 개선 방향이 될 수 있다.

## 8.4 Limitations

본 논문의 주요 limitation은 다음이다.

첫째, SABER는 모든 task/backbone에서 개선되지 않는다. FetchSlide DDPG에서는 HER baseline보다 낮은 final success와 late success를 보였다. 이는 support-aware critic weighting이 underlying actor-critic learner의 stability와 상호작용함을 시사한다.

둘째, support predicate는 task-specific context design에 의존한다. 현재 구현은 각 task의 geometry와 phase를 반영한 handcrafted context bin을 사용한다. 더 일반적인 learned context 또는 representation-based support estimator가 필요하다.

셋째, runtime overhead가 크다. 특히 FetchSlide에서 SABER는 baseline보다 훨씬 느렸다. 현재 SABER는 sample efficiency 관점에서 평가되어야 하며, wall-clock efficiency 개선은 별도 과제다.

넷째, DrawerOpen Random-Support 결과는 structured support claim을 약화시킨다. Shuffled-Weight failure는 sample-weight alignment의 중요성을 지지하지만, Random-Support가 강한 결과는 reweighting-induced regularization도 성능에 기여할 수 있음을 보여준다.

다섯째, support와 TD-error의 독립성 분석이 아직 필요하다. SABER가 PER와 다르다는 주장을 강하게 하려면 support-positive indicator와 TD-error magnitude의 correlation/overlap 분석을 추가해야 한다.

# 9. Conclusion

본 논문은 sparse-reward goal-conditioned HER setting에서 relabeled Bellman target의 quantity와 reliability가 다르다는 문제를 제기했다. HER는 실패 경험을 relabel하여 학습 가능한 targets를 늘리지만, 모든 relabeled Bellman update가 현재 policy action 근방에서 equally reliable한 것은 아니다.

우리는 이 문제를 해결하기 위해 SABER, Support-Aware Bellman Error Reweighting을 제안했다. SABER는 local state-goal-horizon context에서 current policy action이 replay-action support를 갖는지를 확인하고, support-positive sample의 critic Bellman error를 더 강하게 fitting한다. Main variant는 critic-only이며, actor objective와 replay sampling distribution은 baseline과 동일하게 유지한다.

FetchPickAndPlace에서 SABER는 DDPG, TD3, SAC 모두에서 HER baseline보다 높은 final success와 success AUC를 달성했다. FetchSlide에서는 TD3와 SAC에서 개선을 보였지만 DDPG에서는 실패하여 task/backbone-dependent limitation을 드러냈다. DrawerOpen은 SABER가 harder external setting에서도 유용할 수 있음을 보였지만, Random-Support와 Shuffled-Weight controls는 support-aware weighting의 mechanism을 더 정교하게 분석해야 함을 보여준다.

결론적으로 SABER는 새로운 general-purpose RL algorithm이 아니라, HER 기반 off-policy actor-critic에서 critic이 어떤 Bellman supervision을 더 신뢰해야 하는지를 다루는 간단한 support-aware objective다. 이 좁은 claim이 SABER의 강점이다.

# References To Add

최종 영어 LaTeX 변환 시 `.bib`로 정리할 reference 후보:

- Hindsight Experience Replay: Andrychowicz et al.
- DDPG: Lillicrap et al.
- TD3: Fujimoto et al.
- SAC: Haarnoja et al.
- Prioritized Experience Replay: Schaul et al.
- BCQ: Fujimoto et al.
- CQL: Kumar et al.
- Meta-World benchmark: Yu et al.
- Fetch robotics environments / multi-goal RL benchmark: Plappert et al.

# Remaining TODO Before English Draft

1. Support-positive indicator와 TD-error magnitude의 correlation/overlap 분석을 계산한다.
2. FPP random/shuffled/actor-scope ablation 중 main paper에 넣을 수 있는 row가 있는지 재확인한다.
3. FetchPush는 결과가 완전한 row만 appendix에 넣을지 결정한다.
4. Runtime plot: AUC improvement vs steps/sec 또는 AUC/hour를 만든다.
5. Conceptual diagram을 만든다: HER relabeling -> local support lookup -> critic loss weighting.
6. AAAI-27 LaTeX template으로 옮길 때 7-page main text 안에 들어갈 표/그림 수를 조정한다.
7. BibTeX key와 citation 문장을 정리한다.

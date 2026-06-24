# SABER 논문 이론 정리 및 AAAI-27 구조 초안

좋습니다. 앞으로 논문 프레임은 이렇게 잡는 게 가장 좋습니다.

> **SABER는 sparse-reward goal-conditioned HER setting에서, HER가 만들어낸 Bellman target들을 모두 동일하게 신뢰하지 않고, local replay-action support가 있는 target의 critic Bellman error를 더 강하게 학습시키는 방법이다.**
> 단, 알고리즘 정의 자체는 reward-agnostic하게 두어 dense reward / HER-off에도 확장 가능하다고 설명한다.

즉, **main story는 좁고 강하게**, **method definition은 넓게** 가져갑니다.

---

# 1. 논문 핵심 thesis

논문에서 가장 중심이 되는 주장은 다음입니다.

> **Hindsight relabeling creates useful training goals, but it does not guarantee that every relabeled Bellman update is equally reliable for the current policy. SABER uses local replay-action support as a reliability signal and reweights only the critic Bellman error accordingly.**

한국어로 풀면:

> HER는 sparse reward에서 학습 가능한 goal-conditioned Bellman target을 많이 만들어주지만, 그렇게 만들어진 모든 Bellman update가 현재 policy에 대해 똑같이 신뢰 가능한 것은 아니다. SABER는 동일한 local state-goal-horizon context 안에서 현재 policy action이 replay action support를 갖는지를 확인하고, support가 있는 sample의 critic Bellman error만 더 강하게 학습한다.

이 thesis가 중요한 이유는 HER의 장점과 약점을 동시에 건드립니다. HER는 sparse/binary reward에서 reward engineering 없이 학습 가능하도록 하는 대표적 방법이고, off-policy RL과 결합 가능하며 pushing, sliding, pick-and-place 같은 manipulation task에서 검증되었습니다. ([arXiv][1]) 그러나 HER는 “어떤 goal로 relabel할지”를 정할 뿐, 그 relabeled Bellman update가 현재 actor의 action 근방에서 충분한 replay support를 갖는지는 보장하지 않습니다.

---

# 2. Scope 정리

논문에서는 scope를 이렇게 선언하는 것이 가장 깔끔합니다.

## Main scope

$$
\text{sparse-reward goal-conditioned manipulation} + \text{HER} + \text{off-policy actor-critic}
$$

저장소 실험 계획도 FetchPush, FetchSlide, FetchPickAndPlace를 대상으로 하고, 공통 설정을 `reward_type = sparse`, `base learner = DDPG + HER`, `HER ratio = 0.8`로 두고 있습니다.

## Method scope

$$
\text{goal-conditioned off-policy actor-critic}
$$

SABER weight는 reward 값 자체를 보지 않기 때문에 dense reward에도 원리적으로 적용 가능합니다. 코드도 FetchPush, FetchPickAndPlace, FetchSlide에서 sparse와 dense reward env id를 모두 resolve할 수 있고, `reward_type`도 `sparse|dense`를 허용합니다.

논문 문장은 이렇게 쓰면 됩니다.

> While SABER is reward-agnostic, we focus on sparse-reward HER benchmarks because hindsight relabeling is most critical there and Bellman target reliability becomes a central bottleneck.

이 문장으로 “dense도 되는데 왜 sparse만 했냐”는 reviewer 질문을 방어할 수 있습니다.

---

# 3. Problem formulation

Goal-conditioned MDP를 다음처럼 둡니다.

$$
\mathcal M = (\mathcal S, \mathcal A, \mathcal G, P, r_g, \gamma)
$$

각 transition은 replay buffer에 다음 형태로 저장됩니다.

$$
\tau_i = (o_i, a_i, o'_i, g_i^{env}, d_i, h_i)
$$

HER를 사용하면 training 시점에 original goal 대신 relabeled training goal을 붙입니다.

$$
g_i^{tr} \sim \mathcal H(\tau_i)
$$

여기서 $\mathcal H$는 HER relabeling distribution입니다. HER-off라면 단순히

$$
g_i^{tr}=g_i^{env}
$$

입니다. 따라서 SABER는 HER를 필수 구성요소로 가정하지 않고, **training goal $g_i^{tr}$이 주어진 상태에서 작동하는 critic weighting method**로 정의할 수 있습니다.

Goal-conditioned state를

$$
x_i = (o_i, g_i^{tr})
$$

라고 두면, actor와 critic은 다음입니다.

$$
\pi_\theta(x_i)=\pi_\theta(o_i,g_i^{tr})
$$

$$
Q_\phi(x_i,a_i)=Q_\phi(o_i,g_i^{tr},a_i)
$$

Bellman target은 일반적으로

$$
y_i =
r(o_i,a_i,o'_i,g_i^{tr})
+
\gamma(1-d_i)
Q_{\bar\phi}
(o'_i,g_i^{tr},\pi_{\bar\theta}(o'_i,g_i^{tr}))
$$

입니다.

Sparse reward인지 dense reward인지는 $r$의 형태만 바꿉니다. SABER의 support signal은 $r$을 직접 보지 않습니다.

---

# 4. Motivation: HER의 “target quantity”와 “target reliability”는 다르다

HER의 역할은 실패 경험에서도 학습 가능한 goal을 만들어 Bellman target quantity를 늘리는 것입니다.

하지만 SABER가 보는 문제는 그 다음입니다.

> HER가 만든 Bellman target이 현재 policy의 action region에서 얼마나 reliable한가?

Off-policy actor-critic에서는 critic이 replay action에 대해 학습되고, actor는 critic이 높게 평가하는 current policy action을 따라갑니다. TD3 논문도 actor-critic에서 function approximation error와 overestimation이 문제가 되며, twin critics와 delayed policy update로 이를 줄이려 합니다. ([arXiv][2])

SABER의 관점은 이와 연결됩니다.

* critic은 replay data가 충분한 local action region에서는 interpolation에 가깝게 학습됩니다.
* replay data가 부족한 local action region에서는 extrapolation을 해야 합니다.
* actor는 바로 그 extrapolated critic value를 따라갈 수 있습니다.
* 따라서 현재 policy action이 local replay action support 안에 있는지 확인하는 것이 중요합니다.

offline RL의 extrapolation error 논의와도 개념적으로 연결됩니다. BCQ는 fixed batch data에서 current policy distribution과 dataset distribution이 어긋나면 standard off-policy RL이 extrapolation error로 실패할 수 있다고 보고, action space를 batch-constrained하게 제한합니다. ([arXiv][3]) CQL도 offline RL에서 dataset-policy distribution shift와 OOD action overestimation을 문제로 보고 conservative Q-function을 학습합니다. ([arXiv][4])

다만 SABER는 offline RL 알고리즘이 아닙니다. SABER는 online off-policy HER setting에서 action을 제한하지 않고, **critic Bellman error의 학습 강도만 support-aware하게 조절**합니다.

---

# 5. SABER의 핵심 정의

현재 저장소 문서에서도 target methods에서 `SABER = B3-critic-only`라고 두고, core claim을 “local replay support as a critic-side Bellman fitting reliability signal”로 정의합니다.  논문에서는 내부 이름 B3를 쓰지 않고, 이 방법을 전부 **SABER**로 통일합니다.

## 5.1 Local context

각 train item $i$에 대해 local context를 정의합니다.

$$
c_i = C(o_i, g_i^{tr}, h_i)
$$

저장소 문서에서는 이를 `query = (observation, g_train, horizon)`으로 표현하고, replay buffer에서 같은 local context에 속하는 transition들을 찾는다고 설명합니다.

이때 중요한 점은 다음입니다.

> 같은 local context에서 발견된 replay action이 반드시 $g_i^{tr}$을 목표로 실행되었다고 주장하는 것은 아니다. HER relabeling 이후 같은 local training context로 해석될 수 있는 state에서 실제 실행된 action이라는 의미다.

문서도 이 점을 명확히 합니다.

## 5.2 Local replay-action support

local replay pool을 다음처럼 둡니다.

$$
\mathcal L_i = \{j \in \mathcal D:
C(o_j,g_i^{tr},h_j)=C(o_i,g_i^{tr},h_i)\}
$$

해당 local pool의 action-bin set은

$$
\mathcal B_i = \{b(a_j): j\in \mathcal L_i\}
$$

입니다.

현재 policy action은

$$
a_i^\pi=\pi_\theta(o_i,g_i^{tr})
$$

이고, 그 action bin은

$$
b_i^\pi=b(a_i^\pi)
$$

입니다.

## 5.3 SABER support indicator

SABER support-positive indicator는 다음입니다.

$$
\psi_i =
\mathbf 1
\left[
|\mathcal L_i|\ge N_{\min}
\land
|\mathcal A_i^{sup}|\ge M_{\min}
\land
b_i^\pi \in \mathcal B_i
\right]
$$

문서상 support-positive 조건도 local pool size, sampled/available support size, policy action bin existence로 정리되어 있습니다.

직관은 간단합니다.

> 현재 policy가 선택할 action이 그 local state-goal context 안에서 replay buffer에 실제 실행된 action family에 속하면, 그 Bellman update는 critic에게 더 reliable한 fitting target이다.

## 5.4 SABER weight

Raw weight는 다음입니다.

$$
w_i = 1+\lambda_{\text{S}}\psi_i
$$

문서도 동일하게 `w_i = 1 + lambda_support * psi_support_i`로 정의합니다.

Batch-normalized weight는

$$
\Omega_i =
\frac{w_i}
{\frac{1}{B}\sum_{j=1}^{B}w_j+\epsilon}
$$

입니다. 문서에도 `Omega_i = w_i / mean_batch(w)`로 되어 있고, support-positive sample의 상대 학습 비중을 높이되 support-negative sample을 버리지는 않는다고 설명합니다.

## 5.5 Critic-only Bellman objective

SABER critic loss는 다음입니다.

$$
\mathcal L_Q^{SABER}(\phi) =
\frac{1}{B}
\sum_{i=1}^{B}
\Omega_i
\left(
Q_\phi(o_i,g_i^{tr},a_i) -
y_i
\right)^2
$$

Actor loss는 uniform하게 유지합니다.

$$
\mathcal L_\pi^{SABER}(\theta) =
-\frac{1}{B}
\sum_{i=1}^{B}
Q_\phi(o_i,g_i^{tr},\pi_\theta(o_i,g_i^{tr}))
$$

문서에서도 B3-original은 actor와 critic 모두 같은 weight를 적용하지만, SABER에 해당하는 critic-only variant는 critic loss에만 weight를 적용하고 actor loss는 uniform하게 둔다고 정의합니다.  코드에서도 `critic_only`일 때 `omega_critic=omega`, `omega_actor=ones`로 설정됩니다.

---

# 6. 이론적 해석 1: Reliability-weighted Bellman regression

SABER는 다음의 modeling assumption에 기반합니다.

## Assumption: Local support indicates lower Bellman fitting uncertainty

$$
\operatorname{Var}(\varepsilon_i \mid \psi_i=1)
<
\operatorname{Var}(\varepsilon_i \mid \psi_i=0)
$$

여기서

$$
y_i =
TQ(o_i,g_i^{tr},a_i)+\varepsilon_i
$$

라고 보면, $\varepsilon_i$는 Bellman target noise, function approximation error, local extrapolation error를 포함하는 residual noise입니다.

Weighted least squares 관점에서 noise variance가 낮은 sample은 더 큰 weight를 받는 것이 합리적입니다.

$$
\min_\phi
\sum_i
\frac{1}{\sigma_i^2}
\left(
Q_\phi(x_i,a_i)-y_i
\right)^2
$$

SABER는 $\sigma_i^2$를 직접 추정하지 않고, binary support indicator $\psi_i$로 근사합니다.

$$
\frac{1}{\sigma_i^2}
\approx
1+\lambda_S \psi_i
$$

즉, SABER는 정확한 uncertainty estimator라기보다 **support-based reliability proxy**입니다.

논문에서는 이를 theorem처럼 과하게 주장하기보다 다음 정도의 proposition으로 두는 게 좋습니다.

> **Proposition-style interpretation.** If local replay-action support is positively associated with lower Bellman target uncertainty near the current policy action, then SABER approximates reliability-weighted Bellman regression using a binary, reward-agnostic support proxy.

---

# 7. 이론적 해석 2: Actor update를 직접 bias하지 않는다

SABER가 critic-only인 것은 논문에서 매우 중요한 차별점입니다.

Actor까지 weight를 걸면 actor objective가 다음처럼 바뀝니다.

$$
\mathcal L_\pi^{AC} =
-\frac{1}{B}
\sum_i
\Omega_i
Q_\phi(x_i,\pi_\theta(x_i))
$$

이 경우 support signal이 critic reliability뿐 아니라 actor state-goal distribution까지 직접 bias합니다. 그러면 “support가 critic target reliability를 높였다”는 해석과 “actor가 support-rich contexts에 더 끌렸다”는 해석이 섞입니다.

SABER는 actor update를

$$
\mathcal L_\pi^{SABER} =
-\frac{1}{B}
\sum_i
Q_\phi(x_i,\pi_\theta(x_i))
$$

로 유지합니다. 따라서 support signal의 역할은 critic Bellman fitting에 국한됩니다.

논문에서 이 부분은 이렇게 표현하면 좋습니다.

> SABER decouples target reliability from policy improvement. The support signal changes how strongly the critic fits each Bellman equation, but does not directly change the actor’s empirical state-goal weighting.

---

# 8. PER / TD-error weighting과의 차이

PER는 replay transition을 uniform하게 재사용하는 대신 중요한 transition을 더 자주 replay해 learning efficiency를 높이려는 framework입니다. 원 PER 논문은 prioritized replay가 important transitions를 더 자주 replay하도록 한다고 설명합니다. ([arXiv][5])

SABER는 이와 다릅니다.

PER/TD-error weighting은 보통 다음을 봅니다.

$$
|\delta_i| =
|Q_\phi(x_i,a_i)-y_i|
$$

즉, “critic이 지금 얼마나 틀렸는가”입니다.

SABER는 다음을 봅니다.

$$
b(\pi_\theta(x_i)) \in \mathcal B_i
$$

즉, “현재 policy action이 local replay action support 안에 있는가”입니다.

둘은 다른 정보입니다. TD-error가 큰 sample은 중요할 수도 있지만, unsupported region의 noisy target일 수도 있습니다. SABER는 large-error sample이 아니라 **locally supported, policy-relevant Bellman equation**을 강조합니다.

논문에서 강한 문장:

> SABER does not prioritize surprise; it prioritizes supported Bellman supervision.

---

# 9. Mechanism hypotheses

논문은 다음 네 가지 가설을 실험으로 검증하는 구조가 좋습니다.

## H1. SABER improves sparse HER learning

$$
SABER > HER\ baseline
$$

주요 metric:

* Eval AUC
* final success
* best success
* steps-to-threshold
* best-to-final drop

## H2. Critic-only support weighting is cleaner than actor-critic weighting

$$
SABER > SABER\text{-}AC
$$

여기서 SABER-AC는 내부적으로 B3-original에 해당하는 actor-critic support weighting ablation입니다. 문서도 B3-original을 모든 B3-family 결과의 required reference로 두라고 명시합니다.

## H3. Support alignment matters, not just weight distribution

$$
SABER > SABER\text{-}Random
$$

$$
SABER > SABER\text{-}Shuffled
$$

문서도 random/shuffled control의 목적을 “support-positive sample을 정확히 강조해서인지, 단순 loss-scale/weight-distribution 효과인지 분리하기 위해서”라고 설명합니다.

## H4. SABER is not TD-error weighting

$$
SABER > TD\text{-}error\ critic\ weighting
$$

$$
SABER > PER\text{-}style\ loss\ weighting
$$

저장소 문서도 TD-error baseline과 PER-style baseline을 SABER와 비교하도록 설계하고, PER-style baseline은 full replay sampling replacement가 아니라 우선 loss weighting으로 구현하라고 정리합니다.

---

# 10. 논문 contribution 정리

논문 contribution은 네 개가 적당합니다.

## Contribution 1: Reliability gap in HER

HER는 relabeled training goals를 만들지만, 각 Bellman update가 current policy action에 대해 얼마나 supported/reliable한지는 보장하지 않는다는 문제를 제기합니다.

## Contribution 2: SABER objective

Local state-goal-horizon context 안에서 current policy action의 replay-action support를 확인하고, support-positive Bellman error만 critic-side로 더 강하게 학습하는 objective를 제안합니다.

## Contribution 3: Mechanism-preserving ablations

SABER-Random, SABER-Shuffled, SABER-AC, TD-error, PER-style 비교를 통해 gain이 단순 loss scaling이나 TD-error prioritization이 아니라 support/sample alignment에서 나온다는 것을 보입니다.

## Contribution 4: Efficient implementation 또는 runtime analysis

Runtime-optimized SABER는 semantic-preserving support lookup optimization으로 포지셔닝합니다. 문서도 runtime optimization이 support definition, action-bin rule, local_search_max 의미를 바꾸면 안 된다고 명시합니다.

이 contribution은 실험 결과가 충분하면 main contribution으로 넣고, 아니면 appendix/engineering contribution으로 낮추면 됩니다.

---

# 11. 대략적인 논문 구조

## Title 후보

1. **SABER: Support-Aware Bellman Error Reweighting for Hindsight Goal-Conditioned Reinforcement Learning**
2. **SABER: Reliable Bellman Fitting via Local Replay-Action Support in Sparse-Reward Manipulation**
3. **Support-Aware Critic Reweighting for Hindsight Goal-Conditioned Robotic Manipulation**

가장 무난한 건 1번입니다. 제목에서 HER를 너무 좁히고 싶으면 “Hindsight Goal-Conditioned”를 쓰고, 방법의 범용성을 남기고 싶으면 “Goal-Conditioned”만 씁니다.

추천:

> **SABER: Support-Aware Bellman Error Reweighting for Sparse Goal-Conditioned Reinforcement Learning**

---

## Abstract 구조

초록은 5문장으로 갑니다.

1. Sparse-reward goal-conditioned manipulation에서 HER는 relabeled goals로 학습 신호를 늘리지만, 모든 relabeled Bellman target이 equally reliable한 것은 아니다.
2. 현재 policy action이 local replay-action support 밖에 있으면 critic은 extrapolation에 의존하고 actor update가 불안정해질 수 있다.
3. SABER는 local state-goal-horizon context에서 current policy action의 replay-action support를 확인하고, supported samples의 critic Bellman error를 더 크게 가중한다.
4. Actor update는 unweighted로 유지해 support signal을 critic target reliability signal로 분리한다.
5. FetchPush/Slide/PickAndPlace sparse HER benchmarks에서 SABER가 HER baseline, actor-weighted support, random/shuffled controls, TD-error/PER-style weighting보다 learning efficiency와 stability를 개선한다.

마지막 문장은 실제 결과가 나오면 숫자로 바꿉니다.

---

## 1. Introduction

구성:

1. Sparse reward goal-conditioned robotics의 어려움.
2. HER의 역할: relabeling으로 sparse/binary reward에서도 학습 가능.
3. 남는 문제: relabeled Bellman targets are not equally reliable.
4. Off-policy actor-critic에서 critic extrapolation / current policy action support mismatch.
5. SABER 핵심 아이디어.
6. Contributions.

핵심 문장:

> HER answers which goal a transition can train on; SABER answers how much the resulting Bellman error should be trusted by the critic.

---

## 2. Related Work

네 묶음으로 정리합니다.

### 2.1 Hindsight Experience Replay and sparse goal-conditioned RL

HER, HGR, HGG, MHER, Soft HER 등. 여기서는 “goal relabeling / goal selection / hindsight sampling” 계열과 SABER가 다르다는 점을 강조합니다.

SABER는 goal을 새로 고르는 방법이 아니라, 주어진 training goal에 대해 **critic Bellman error reliability**를 조절합니다.

### 2.2 Off-policy actor-critic and critic error

DDPG/TD3/SAC. TD3는 actor-critic에서 function approximation error와 overestimation을 줄이려는 대표적 방법입니다. ([arXiv][2])

SABER는 TD3와 경쟁하는 base algorithm이라기보다 DDPG/TD3/SAC 위에 얹을 수 있는 critic loss weighting입니다.

### 2.3 Prioritized replay and TD-error weighting

PER와 TD-error baseline. PER는 important transition을 더 자주 replay하는 방법입니다. ([arXiv][5]) SABER는 TD-error magnitude가 아니라 local policy-action support를 사용합니다.

### 2.4 Support constraints and extrapolation in offline RL

BCQ/CQL류. 여기서는 “support mismatch”라는 개념적 연결만 하고, SABER가 offline RL method가 아님을 분명히 해야 합니다.

---

## 3. Preliminaries

여기서는 수식만 깔끔하게 둡니다.

* Goal-conditioned MDP
* HER relabeling
* Off-policy actor-critic objective
* Bellman target
* Replay buffer notation

이 섹션 끝에 problem statement를 둡니다.

> Given a HER-relabeled transition, how should we weight its Bellman error when the current policy action may or may not be supported by local replay actions?

---

## 4. Method: SABER

이 섹션이 논문의 중심입니다.

### 4.1 Local state-goal-horizon context

$$
c_i=C(o_i,g_i^{tr},h_i)
$$

Fetch task별로 context가 어떻게 만들어지는지는 main text에서는 간단히 설명하고, 세부 binning은 appendix로 보냅니다.

### 4.2 Replay-action support

$$
\mathcal L_i = \{j:C(o_j,g_i^{tr},h_j)=c_i\}
$$

$$
\mathcal B_i = \{b(a_j):j\in \mathcal L_i\}
$$

### 4.3 SABER support indicator

$$
\psi_i =
\mathbf 1[
|\mathcal L_i|\ge N_{\min}
\land
|\mathcal A_i^{sup}|\ge M_{\min}
\land
b(\pi_\theta(x_i))\in \mathcal B_i
]
$$

### 4.4 Critic-only support-aware Bellman error

$$
w_i=1+\lambda_S\psi_i
$$

$$
\Omega_i=\frac{w_i}{\operatorname{mean}_B(w)+\epsilon}
$$

$$
\mathcal L_Q^{SABER} =
\frac1B\sum_i
\Omega_i
(Q_\phi(x_i,a_i)-y_i)^2
$$

$$
\mathcal L_\pi^{SABER} =
-\frac1B\sum_iQ_\phi(x_i,\pi_\theta(x_i))
$$

### 4.5 Algorithm box

Pseudo-code는 이렇게 갑니다.

```text
Algorithm 1: SABER update

Input: replay buffer D, actor pi_theta, critic Q_phi
Sample batch from HER replay
For each sample i:
    construct training goal g_i^tr
    compute local context c_i = C(o_i, g_i^tr, h_i)
    compute current policy action a_i^pi = pi_theta(o_i, g_i^tr)
    find replay actions in local context c_i
    set psi_i = 1 if local support and policy-action-bin support exist
    set w_i = 1 + lambda_S psi_i
Normalize Omega_i = w_i / mean_batch(w)
Update critic with weighted Bellman error
Update actor with unweighted deterministic policy objective
```

---

## 5. Why critic-only? Theoretical interpretation

이 섹션은 짧지만 강하게 갑니다.

### 5.1 Reliability-weighted Bellman regression

Assumption과 proposition-style interpretation을 둡니다.

$$
\operatorname{Var}(\varepsilon_i|\psi_i=1)
<
\operatorname{Var}(\varepsilon_i|\psi_i=0)
$$

SABER는 inverse-variance weighting의 binary proxy.

### 5.2 Avoiding actor distribution bias

Actor까지 weight를 걸면 mechanism이 섞임. SABER는 critic reliability만 바꿈.

### 5.3 Relation to TD-error weighting

TD-error는 error magnitude, SABER는 empirical support. 두 signal은 다른 hypothesis를 테스트함.

---

## 6. Experiments

## 6.1 Main benchmarks

Main은 세 task로 갑니다.

| Task              | Reward | Relabeling | Main purpose                           |
| ----------------- | -----: | ---------: | -------------------------------------- |
| FetchPush         | sparse |        HER | easier manipulation, sample efficiency |
| FetchSlide        | sparse |        HER | stability / final drop                 |
| FetchPickAndPlace | sparse |        HER | hard compositional manipulation        |

## 6.2 Methods

Main table:

| 논문 이름               | 설명                                                         |
| ------------------- | ---------------------------------------------------------- |
| HER baseline        | DDPG/TD3/SAC + HER, uniform critic loss                    |
| SABER               | critic-only support-aware Bellman error reweighting        |
| SABER-AC            | actor and critic both weighted                             |
| SABER-Random        | same high-weight count, random assignment                  |
| SABER-Shuffled      | same weight distribution, broken support alignment         |
| TD-error weighting  | critic-only TD-error magnitude weighting                   |
| PER-style weighting | PER-like priority-derived loss weighting, uniform sampling |
| Efficient SABER     | semantic-preserving support lookup optimization            |

저장소의 execution matrix도 B0, B3-original, B3-critic-only, random, shuffled, runtime optimization, TD-error, PER-style을 비교군으로 둡니다.

## 6.3 Metrics

Main metrics:

* Eval AUC
* final success
* best success
* drop from best to final
* steps-to-threshold
* runtime / steps per second
* AUC per hour

문서도 Eval AUC를 normalized area under evaluation success curve로 정의하고, final/best success보다 learning efficiency를 더 잘 반영한다고 설명합니다.

## 6.4 Mechanism diagnostics

필수 diagnostic:

* support-positive rate
* mean / max / entropy / ESS of weights
* support-weight alignment rate
* support vs TD-error correlation
* support-bin TD-error reduction
* phase-wise support rate

문서도 weight distribution validation에서 SABER, shuffled, random의 mean/max/std/entropy/ESS를 비교하라고 정리합니다.

---

# 12. Figure 구성

논문에 들어갈 figure는 5개가 좋습니다.

## Figure 1: Conceptual diagram

HER transition이 relabeled goal을 받고, SABER가 local context를 찾아 current policy action이 replay action support 안에 있는지 확인한 뒤 critic loss weight를 정하는 그림.

## Figure 2: Learning curves

FetchPush / FetchSlide / FetchPickAndPlace success curves.

Main comparison:

* HER baseline
* SABER
* SABER-AC
* SABER-Shuffled
* SABER-Random

## Figure 3: AUC / final success bar plot

5 seeds mean ± CI.

## Figure 4: Mechanism plot

SABER vs shuffled/random의 weight distribution은 비슷하지만 performance가 다르다는 그림.

## Figure 5: Runtime-performance tradeoff

x-axis: runtime overhead
y-axis: Eval AUC improvement
point size: final success

---

# 13. Main paper vs appendix 배치

## Main paper에 반드시 넣을 것

1. Sparse HER main result
2. SABER vs HER baseline
3. SABER vs SABER-AC
4. SABER vs random/shuffled
5. TD-error or PER-style 최소 하나
6. Method equation
7. Reliability interpretation

## Appendix로 보낼 것

1. Dense reward compatibility
2. HER-off result
3. local context binning details
4. runtime optimization correctness test
5. per-phase diagnostics
6. extra seeds / full curves
7. hyperparameter sensitivity: $\lambda_S$, $N_{\min}$, $M_{\min}$, local_search_max

Dense/HER-off는 main claim이 아니라 다음 문장으로 appendix에 두면 됩니다.

> Dense-reward and HER-off experiments are included to test compatibility beyond the main sparse HER setting, not as the primary benchmark for SABER.

---

# 14. 논문에서 피해야 할 claim

다음은 피하는 게 좋습니다.

> SABER is a general-purpose RL algorithm for all continuous control.

너무 넓습니다. reviewer가 MuJoCo dense control, SAC, TD3, PER, offline RL, model-based RL까지 다 끌고 들어올 수 있습니다.

> SABER solves extrapolation error.

너무 강합니다. SABER는 extrapolation error를 직접 제거하는 게 아니라, local support가 있는 Bellman update를 더 신뢰하는 critic weighting입니다.

> SABER is a HER variant.

애매합니다. SABER는 HER와 결합하지만 goal relabeling method가 아닙니다.

더 좋은 claim은 다음입니다.

> SABER addresses a reliability gap in hindsight goal-conditioned Bellman learning: relabeling creates training targets, but local replay-action support determines whether those targets should be trusted equally by the critic.

---

# 15. 지금 단계의 논문 골격

최종적으로 AAAI paper outline은 이렇게 잡겠습니다.

```text
Title:
SABER: Support-Aware Bellman Error Reweighting for Sparse Goal-Conditioned Reinforcement Learning

Abstract

1. Introduction
   1.1 Sparse goal-conditioned manipulation and HER
   1.2 Reliability gap in relabeled Bellman targets
   1.3 SABER overview
   1.4 Contributions

2. Related Work
   2.1 Hindsight Experience Replay and goal relabeling
   2.2 Off-policy actor-critic and critic error
   2.3 Prioritized replay and TD-error weighting
   2.4 Support mismatch and offline RL

3. Preliminaries
   3.1 Goal-conditioned MDP
   3.2 HER relabeled Bellman updates
   3.3 Off-policy actor-critic objective

4. SABER
   4.1 Local state-goal-horizon context
   4.2 Local replay-action support
   4.3 Support indicator
   4.4 Support-aware critic Bellman error
   4.5 Critic-only actor-critic update
   4.6 Efficient implementation

5. Theoretical Interpretation
   5.1 SABER as reliability-weighted Bellman regression
   5.2 Why critic-only?
   5.3 Difference from TD-error/PER weighting

6. Experiments
   6.1 Sparse HER benchmark setup
   6.2 Baselines and ablations
   6.3 Main learning performance
   6.4 Mechanism analysis
   6.5 Runtime analysis
   6.6 Dense/HER-off compatibility check

7. Discussion and Limitations
   7.1 Dependence on local context design
   7.2 Runtime overhead
   7.3 Dense reward generality
   7.4 Failure modes

8. Conclusion
```

---

# 16. 가장 중요한 한 줄

논문 전체를 관통하는 한 줄은 이걸로 잡으면 됩니다.

> **HER tells an agent what else a transition could have achieved; SABER tells the critic when the resulting Bellman error is locally supported enough to trust.**

[1]: https://arxiv.org/abs/1707.01495?utm_source=chatgpt.com "Hindsight Experience Replay"
[2]: https://arxiv.org/abs/1802.09477 "[1802.09477] Addressing Function Approximation Error in Actor-Critic Methods"
[3]: https://arxiv.org/abs/1812.02900 "[1812.02900] Off-Policy Deep Reinforcement Learning without Exploration"
[4]: https://arxiv.org/abs/2006.04779?utm_source=chatgpt.com "Conservative Q-Learning for Offline Reinforcement Learning"
[5]: https://arxiv.org/abs/1511.05952?utm_source=chatgpt.com "Prioritized Experience Replay"
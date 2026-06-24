# SABER AAAI-27 Method Definition Fixes

작성일: 2026-06-24
목적: 리뷰에서 지적된 method definition gap을 닫기 위해, SABER의 정확한 수식/구현 의미/논문 수정 방향을 한국어로 고정한다.

## 1. 핵심 수정 판단

SABER의 가장 공격받기 쉬운 지점은 다음이다.

> SABER는 current policy action의 local support를 보지만, 실제 critic TD loss는 replay action에서 계산된다.

따라서 논문에서 다음처럼 주장하면 위험하다.

> policy action support가 replay-action Bellman target noise를 직접 측정한다.

더 방어 가능한 정의는 다음이다.

> SABER는 current policy action support를 policy-improvement-relevant critic fitting reliability proxy로 사용한다.

즉, off-policy actor-critic에서 critic은 replay action \(a_i^r\)에서 Bellman regression을 수행하지만, actor는 같은 \(x_i=(o_i,g_i^{tr})\)에서 current policy action \(a_i^\pi\)의 value를 exploit한다. SABER는 "이 local state-goal-horizon context가 actor가 실제로 사용할 action region을 replay buffer 안에서 support하는가"를 보고, 그 context의 replay-action Bellman residual을 더 강하게 fitting한다.

이 해석은 strong theorem이 아니라 empirical working hypothesis다.

## 2. 고정할 표기

Replay transition:

\[
\tau_i=(o_i,a_i^r,o_i',g_i^{env},\tilde d_i,\chi_i,h_i)
\]

여기서 \(\tilde d_i\)는 environment raw done flag이고, \(\chi_i\)는 time-limit timeout flag다. Bellman target에서 쓰는 \(d_i\)는 bootstrapping terminal indicator다. 기본 비교 설정은 `success_terminates=false`, `timeout_bootstrap=true`이므로

\[
d_i=\tilde d_i(1-\chi_i)
\]

이고, timeout transition과 HER success state는 target에서 terminal로 강제하지 않는다.

Training goal:

\[
g_i^{tr}\sim \mathcal H(\tau_i)
\]

Goal-conditioned input:

\[
x_i=(o_i,g_i^{tr}),\qquad x_i'=(o_i',g_i^{tr})
\]

Reward shorthand:

\[
r_i=r(o_i,a_i^r,o_i',g_i^{tr})
\]

Replay action:

\[
a_i^r
\]

Current policy action:

\[
a_i^\pi=\pi_\theta(x_i)
\]

Critic Bellman residual은 \(a_i^r\)에서 계산된다. Support predicate는 \(a_i^\pi\)의 action bin membership을 본다.

## 3. Local Context 정의

각 sample의 local context는 query sample의 training goal 아래에서 계산한다.

\[
c_i=C(o_i,g_i^{tr},h_i)
\]

중요한 점:

- replay reference transition \(j\)의 context도 자기 original goal이 아니라 query sample의 \(g_i^{tr}\)로 다시 계산한다.
- 따라서 local pool은 "같은 relabeled goal 아래에서 비슷한 state-horizon region"을 의미한다.

## 4. FetchPickAndPlace Exact Context

FPP main experiments 기준 observation layout:

- `obs[0:3]`: end-effector position
- `obs[3:6]`: object position
- `obs[9:11]`: two-finger gripper state

Phase:

- approach: \(d_{\text{hand,obj}}>d_{\text{reach}}\)
- grasp: \(z_{\text{obj}}\le z_{\text{lift}}\) and gripper gap \(>\) threshold
- lift: \(z_{\text{obj}}\le z_{\text{lift}}\)
- place: otherwise

Full context key:

\[
\begin{aligned}
C_{\mathrm{full}}(o,g,h)=&
10^6\,\mathrm{phase}
+10^5\,\mathrm{bin}(d_{\text{hand,obj}})\\
&+10^4\,\mathrm{bin}(d_{\text{obj,goal}})
+10^3\,\mathrm{bin}(z_{\text{obj}})\\
&+10^2\,\mathrm{bin}(\mathrm{gripper\_gap})
+\mathrm{bin}(h)
\end{aligned}
\]

FPP bins:

- hand-object distance: `(0.03, 0.05, 0.08, 0.12, 0.20, 0.40)`
- object-goal distance: `(0.05, 0.10, 0.20, 0.40, 0.70)`
- object height: `(0.43, 0.48, 0.55, 0.65, 0.75)`
- gripper gap: `(0.02, 0.05, 0.08)`
- horizon: `(10, 25, 50, 100)`

Runtime optimization은 먼저 coarse key로 후보를 제한한다.

\[
C_{\mathrm{coarse}}(o,h)=
10^4\,\mathrm{bin}(d_{\text{hand,obj}})
+10^3\,\mathrm{bin}(z_{\text{obj}})
+10^2\,\mathrm{bin}(\mathrm{gripper\_gap})
+\mathrm{bin}(h)
\]

그 다음 \(C_{\mathrm{full}}\)로 exact filtering을 한다.

## 5. Local Replay-Action Support

Local replay pool:

\[
\mathcal L_i=
\{j\in\mathcal D:
C_{\mathrm{full}}(o_j,g_i^{tr},h_j)
=C_{\mathrm{full}}(o_i,g_i^{tr},h_i)\}
\]

Action-bin set:

\[
\mathcal B_i=\{b(a_j^r):j\in\mathcal L_i\}
\]

Action binning:

- continuous action을 action range 기준으로 \([-1,1]\)에 normalize한다.
- 각 dimension을 \(K\)개 bin으로 나눈다.
- dimension별 bin id를 mixed-radix integer로 합친다.
- FPP main experiments는 `action_bins_per_dim=5`를 사용한다.

Policy action bin:

\[
b_i^\pi=b(a_i^\pi)
\]

## 6. Support Indicator

아래 정의로 통일한다.

\[
\begin{aligned}
n_i^{avail} &= \min(|\mathcal L_i|,M_B),\\
\psi_i
&=
\mathbf 1[
|\mathcal L_i|\ge N_{\mathrm{local}}
\land
n_i^{avail}\ge M_{\mathrm{avail}}
\land
b_i^\pi\in\mathcal B_i
].
\end{aligned}
\]

각 symbol:

- \(M_B\): support candidate cap
- \(n_i^{avail}\): cap을 적용한 available local transition count
- \(M_{\mathrm{avail}}\): minimum available support count
- \(N_{\mathrm{local}}\): minimum exact local pool size
- \(N_{\mathrm{scan}}\): coarse bucket에서 scan하는 최대 후보 수
- \(\lambda_S\): support-positive sample에 주는 raw weight 증가량

구현 config의 legacy 이름은 `M_Q_min`이지만, 이 값은 Q-value threshold나 unique action-bin diversity threshold가 아니다. 현재 구현에서는 `min(|L_i|, M_B) >= M_Q_min`으로 쓰이는 capped local transition count 문턱이다. \(\mathcal B_i\)는 current policy action bin membership을 확인하기 위한 unique action-bin set이며, main SABER predicate는 \(|\mathcal B_i|\)에 대한 별도 diversity 조건을 두지 않는다.

FPP main experiments:

- \(M_B=16\)
- \(M_{\mathrm{avail}}=4\) (`M_Q_min=4`)
- \(N_{\mathrm{local}}=4\)
- \(N_{\mathrm{scan}}=512\)
- \(\lambda_S=1.0\)
- `weighting_mode=batch_norm`
- `weight_scope=critic_only`

이 설정에서는 count 조건이 사실상 \(|\mathcal L_i|\ge 4\)로 축약되며, support-positive 여부를 가르는 핵심 조건은 \(b_i^\pi\in\mathcal B_i\)이다. 두 count symbol은 code/config 호환성과 다른 variant 가능성을 보존하기 위해 문서에 남기되, 논문 본문에서는 중복처럼 보이지 않도록 역할을 분명히 설명한다.

주의:

- config에 있는 \(N_{\mathrm{gap}}=128\)은 older gap-weight variants의 gap-stat readiness threshold다.
- main support-only/runtime-opt SABER predicate에는 \(N_{\mathrm{gap}}\)이 직접 들어가지 않는다.
- 원고의 main method 수식에는 \(N_{\mathrm{gap}}\)을 넣지 않는 것이 맞다.

## 7. Weight Construction

Raw support weight:

\[
w_i^{raw}=1+\lambda_S\psi_i
\]

Batch-normalized weight:

\[
\Omega_i=
\frac{w_i^{raw}}
{\frac{1}{B}\sum_{j=1}^{B}w_j^{raw}+\epsilon}
\]

Critic-only scope:

\[
\Omega_i^{critic}=\Omega_i,\qquad
\Omega_i^{actor}=1
\]

이 설계 때문에 actor objective와 replay sampling distribution은 baseline과 동일하게 유지된다.

## 8. Backbone별 Critic Loss 적용 위치

DDPG:

\[
y_i=r_i+\gamma(1-d_i)Q_{\bar\phi}(x_i',\pi_{\bar\theta}(x_i'))
\]

\[
\mathcal L_Q^{SABER}=
\frac{1}{B}\sum_i
\Omega_i
(Q_\phi(x_i,a_i^r)-y_i)^2
\]

TD3:

\[
\tilde a_i'=\operatorname{clip}(\pi_{\bar\theta}(x_i')+\epsilon_i,a_{\min},a_{\max})
\]

\[
y_i=r_i+\gamma(1-d_i)
\min_{k\in\{1,2\}}Q_{\bar\phi_k}(x_i',\tilde a_i')
\]

\[
\mathcal L_Q^{SABER}=
\frac{1}{B}\sum_i\Omega_i
\left[
(Q_{\phi_1}(x_i,a_i^r)-y_i)^2
+(Q_{\phi_2}(x_i,a_i^r)-y_i)^2
\right]
\]

TD3 actor update는 delayed update를 유지하고, critic-only setting에서는 \(\Omega_i^{actor}=1\)이다.

SAC:

\[
\tilde a_i'\sim\pi_\theta(\cdot\mid x_i')
\]

\[
y_i=r_i+\gamma(1-d_i)
\left[
\min_kQ_{\bar\phi_k}(x_i',\tilde a_i')
-\alpha\log\pi_\theta(\tilde a_i'\mid x_i')
\right]
\]

\[
\mathcal L_Q^{SABER}=
\frac{1}{B}\sum_i\Omega_i
\left[
(Q_{\phi_1}(x_i,a_i^r)-y_i)^2
+(Q_{\phi_2}(x_i,a_i^r)-y_i)^2
\right]
\]

SAC actor objective는 entropy-regularized baseline objective 그대로이며, 현재 구현은 `omega_actor`를 사용하지 않는다.

## 9. PER-style Baseline 정의

SABER와 비교할 PER-style baseline은 replay sampling을 바꾸지 않는다. Uniform replay sampling은 그대로 두고, TD-error-derived loss weight만 critic에 적용한다.

TD-error magnitude:

\[
\delta_i=
\operatorname{stopgrad}
\left(|Q_\phi(x_i,a_i^r)-y_i|\right)
\]

Twin-critic backbone에서는 두 critic residual의 minimum absolute error를 logging/priority용 \(\delta_i\)로 사용한다.
모든 TD-error-derived weight는 no-gradient로 계산하고, weight signal 자체에는 critic loss gradient를 흘리지 않는다.

PER-style priority:

\[
p_i=(\delta_i+\epsilon_{PER})^{\alpha_{PER}}
\]

Raw PER-style weight:

\[
w_i^{PER}
=
\operatorname{clip}
\left(
\frac{p_i}{\frac{1}{B}\sum_jp_j+\epsilon},
0,
U_{PER}
\right)
\]

그 뒤 SABER와 동일하게 batch-normalize하여 critic loss에만 적용한다.

FPP PER-style config:

- \(\alpha_{PER}=0.6\)
- \(\epsilon_{PER}=10^{-6}\)
- \(U_{PER}=2.0\)
- `support_lookup=false`
- `weight_scope=critic_only`
- replay sampling remains uniform

## 10. 논문에 반영한 수정

`paper/saber_aaai27_draft_kr.tex`에 다음을 반영했다.

- replay action \(a_i^r\)와 policy action \(a_i^\pi\)를 분리했다.
- DDPG/TD3/SAC Bellman target을 각각 명시했다.
- FPP exact context/hash/bin definition을 Method에 넣었다.
- local pool이 query training goal \(g_i^{tr}\) 기준으로 계산된다는 점을 명시했다.
- `A_i^sup` 미정의 표현을 제거하고 \(\mathcal B_i\), \(n_i^{avail}\), \(M_B\), \(M_{\mathrm{avail}}\), \(N_{\mathrm{local}}\)로 support predicate를 다시 썼다.
- `M_Q_min`이 unique action-bin diversity가 아니라 capped local transition count 문턱이라는 점을 명시했다.
- \(N_{\mathrm{gap}}\)이 main support-only predicate에 직접 들어가지 않는다고 명시했다.
- critic-only objective와 twin-critic loss 적용 위치를 명시했다.
- PER-style loss weighting baseline 수식을 추가했다.
- 이론 해석을 direct target-noise claim에서 policy-improvement-relevant fitting-risk proxy로 좁혔다.

## 11. 아직 남은 Method 관련 후속 작업

- FetchSlide/FetchPush/DrawerOpen의 exact local context도 appendix에서 task별로 정리해야 한다.
- HER relabeling strategy, reward convention, success threshold는 Reproducibility Appendix에서 별도로 고정해야 한다.
- support predicate가 reliability proxy라는 주장을 닫으려면 support-vs-TD-error diagnostics가 필요하다.
- FPP main task에서도 Random-Support/Shuffled/Fixed/PER controls를 같은 protocol로 비교해야 한다.

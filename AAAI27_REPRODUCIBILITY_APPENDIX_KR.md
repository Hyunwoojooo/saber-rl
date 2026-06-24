# SABER AAAI-27 Reproducibility Appendix 초안

작성일: 2026-06-24
목적: SABER AAAI-27 원고의 C3/C8 재현성 리스크를 줄이기 위해, 현재 repository에 남아 있는 코드와 결과 artifact 기준으로 실험 프로토콜을 고정한다. 이 문서는 `paper/saber_aaai27_draft_kr.tex`의 `Appendix A: Reproducibility Details`에 통합할 source note 역할을 한다.

## 1. Source of Truth

이 문서는 논문 원고보다 낮은 수준의 구현 세부를 보존하기 위한 재현성 부록 초안이다. 숫자와 절차는 아래 파일을 기준으로 확인했다.

| 범위 | 기준 파일 |
| --- | --- |
| FPP 공통 trainer/config | `experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/metacog_lite_v0/config.py`, `trainer.py`, `replay.py`, `envs.py` |
| FPP support context | `experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/metacog_lite_v0/task_fetch_pick_and_place.py` |
| FPP DDPG/SABER config | `experiment_FPP/B0`, `experiment_FPP/b3_critic_only_runtime_opt_v4_ls512`, `experiment_FPP/runs/*/config.json`, `env_snapshot.json` |
| FPP TD3/SAC config | `experiment_FPP/td3_her_baseline_ls512/config.json`, `experiment_FPP/sac_her_baseline_ls512/config.json`, corresponding run configs |
| FetchSlide runner | `experiment_FetchSlide/src/saber_fetch_slide/experiment.py` |
| FetchSlide support context | `experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/metacog_lite_v0/task_fetch_slide.py` |
| FetchSlide result snapshots | `experiment_FetchSlide/experiment/FetchSlide/**/config.json`, `env_snapshot.json`, `summary.json` |
| DrawerOpen scripts | `metaworld_Drawer_Open_TD3/TD3-HER-B0-2M.sh`, `TD3-HER-B3-Critic-Only-2M.sh`, and other control scripts |
| DrawerOpen support context | `metaworld_Drawer_Open_TD3/B3_v4_512_s1_5/code/src/metacog_lite_v0/task_drawer_open.py` |
| DrawerOpen env wrapper | `metaworld_Drawer_Open_TD3/B3_v4_512_s1_5/code/src/metacog_lite_v0/envs.py` |

주의: 일부 오래된 summary/config 파일은 `machine`, `thread_hint`, 일부 normalized config 값을 저장하지 않는다. 따라서 hardware/package version은 현재 artifact만으로 완전히 닫히지 않았고, 최종 appendix에는 새 logging 또는 environment freeze가 필요하다. 현재 LaTeX Appendix A에는 이 한계를 명시하고, steps/sec를 hardware-normalized claim으로 쓰지 않는다고 적었다.

## 2. 공통 Training Protocol

### 2.1 Goal-conditioned state

모든 Fetch 계열 실험은 state를 다음처럼 구성한다.

```text
x_t = concat(observation_t, desired_goal_t)
```

FetchPickAndPlace와 FetchSlide의 `obs_dim=25`, `goal_dim=3`, `state_dim=28`, `action_dim=4`가 `env_snapshot.json`에 저장되어 있다. Action range는 각 dimension에서 `[-1, 1]`이다.

Meta-World DrawerOpen은 Meta-World 환경을 Fetch-style GoalEnv처럼 감싼다. Wrapper는 drawer handle position을 `achieved_goal`, target position을 `desired_goal`로 둔다. State도 동일하게 `concat(observation, desired_goal)`이다.

### 2.2 Replay buffer and HER

Replay buffer는 uniform sampling 기반 `HERReplayBuffer`다. 각 transition은 다음 값을 저장한다.

```text
observation, achieved_goal, desired_goal,
next_observation, next_achieved_goal,
action_exec, reward_ext, done, timeout,
h, episode_id, episode_step, coarse_key, action_bin
```

Mini-batch sampling은 uniform random index sampling이다. HER relabeling은 `her_ratio=0.8`로 적용한다. Relabeled sample은 같은 episode 안에서 현재 step보다 뒤에 있는 future transition을 하나 뽑고, 그 transition의 `next_achieved_goal`을 training desired goal로 사용한다.

```text
for sample i:
  with probability her_ratio:
    choose future slot j from same episode with episode_step_j > episode_step_i
    g_i^tr = next_achieved_goal_j
  otherwise:
    g_i^tr = original desired_goal_i
```

HER 이후 reward는 항상 `compute_reward(next_achieved_goal, g_i^tr)`로 다시 계산한다.

### 2.3 Terminal convention

기본 convention은 다음과 같다.

```text
success_terminates = False
timeout_bootstrap = True
```

즉, HER로 성공 goal이 만들어져도 critic target에서 success를 terminal로 강제하지 않는다. Environment terminal은 true terminal만 사용하고, timeout은 bootstrapping을 허용한다.

```text
env_terminal = done * (1 - timeout)
terminal = env_terminal
nonterminal = 1 - terminal
```

이 설정은 Fetch 계열과 DrawerOpen stress-test에서 모두 논문 비교의 기본값으로 써야 한다.

### 2.4 Optimizer and networks

DDPG/TD3 actor와 critic은 2-hidden-layer MLP다.

```text
MLP: Linear -> ReLU -> Linear -> ReLU -> Linear
hidden_dim = 256
actor output = tanh(...)
critic input = concat(state, action)
```

Optimizer는 Adam이다.

| 항목 | 값 |
| --- | --- |
| actor_lr | `3e-4` |
| critic_lr | `3e-4` |
| gamma | `0.98` |
| tau | `0.005` |
| gradient clipping | max norm `10.0` |
| l2_action_penalty | `0.0` |

DDPG target:

```text
y = r + gamma * nonterminal * Q_target(x_next, pi_target(x_next))
```

TD3 target:

```text
next_action = clip(pi_target(x_next) + clipped Gaussian noise, action_low, action_high)
y = r + gamma * nonterminal * min(Q1_target(x_next,next_action), Q2_target(x_next,next_action))
```

TD3 추가 hyperparameter:

| 항목 | 값 |
| --- | --- |
| policy_delay | `2` |
| target_policy_noise | `0.2` |
| target_noise_clip | `0.5` |

SAC는 FetchSlide runner의 `SACAgent`를 기준으로 한다.

| 항목 | 값 |
| --- | --- |
| actor | tanh Gaussian actor |
| critics | twin Q critics |
| alpha initial | `0.2` |
| auto_alpha | `True` |
| alpha_lr | `3e-4` |
| target_entropy | `-action_dim` when unspecified |

SAC target:

```text
y = r + gamma * nonterminal *
    (min(Q1_target(x_next,a_next), Q2_target(x_next,a_next)) - alpha * log pi(a_next|x_next))
```

### 2.5 Common training values

| 항목 | FetchPickAndPlace | FetchSlide | DrawerOpen |
| --- | ---: | ---: | ---: |
| total_steps | `1,000,000` | `1,000,000` | `2,000,000` for main 2M stress-test |
| max_episode_steps | `50` | `50` | `200` |
| start_steps | `10,000` | `10,000` | `10,000` |
| warmup_steps | `5,000` | `5,000` | `5,000` |
| updates_per_step | `1` | `1` | `1` |
| batch_size | `256` | `256` | `256` |
| replay buffer | `300,000` | `300,000` | `300,000` |
| her_ratio | `0.8` | `0.8` | `0.8` |
| eval_episodes | `20` | `20` | `20` |
| eval_freq | `10,000` for FPP DDPG/TD3, `25,000` for SAC config | `25,000` | `10,000` |
| seeds | `1..5` for reported tables unless noted | `1..5` | `1..5` |

FetchSlide runner sets seeds as follows:

```text
env_seed = seed
torch_seed = seed
numpy_seed = seed
action_noise_seed = seed
weight_seed = 20000 + seed
eval_seed = 10000 + seed
```

DrawerOpen scripts use the same seed convention.

## 3. Environment and Reward Convention

### 3.1 FetchPickAndPlace

| 항목 | 값 |
| --- | --- |
| env_id | `FetchPickAndPlace-v4` |
| entry point | `gymnasium_robotics.envs.fetch.pick_and_place:MujocoFetchPickAndPlaceEnv` |
| reward_type | sparse |
| distance_threshold | `0.05` |
| success | `||achieved_goal - desired_goal|| <= 0.05` |
| reward | environment `compute_reward`; sparse convention is success `0`, failure `-1` |
| max_episode_steps | `50` |

### 3.2 FetchSlide

| 항목 | 값 |
| --- | --- |
| env_id | `FetchSlide-v4` |
| entry point | `gymnasium_robotics.envs.fetch.slide:MujocoFetchSlideEnv` |
| reward_type | sparse |
| distance_threshold | `0.05` |
| success | `||achieved_goal - desired_goal|| <= 0.05` |
| reward | environment `compute_reward`; sparse convention is success `0`, failure `-1` |
| max_episode_steps | `50` |

### 3.3 Meta-World DrawerOpen

| 항목 | 값 |
| --- | --- |
| env_name | `drawer-open-v3` |
| metaworld_reward_function_version | `v2` |
| reward_type | sparse in wrapper |
| max_episode_steps | `200` |
| achieved_goal | drawer handle position |
| desired_goal | `_target_pos`, `target_pos`, `_goal`, or `goal` from unwrapped env |
| success threshold | `drawer_success_threshold=0.05` |
| sparse reward | success `0`, failure `-1` under wrapper distance threshold |
| env_success logging | Meta-World `info["success"]` or `info["is_success"]` if available |

DrawerOpen wrapper additionally records `goal_success`, `env_success`, and `desired_goal_source` in `info`.

## 4. SABER Support Lookup Protocol

### 4.1 Shared hyperparameters

Main SABER rows use the following support parameters unless otherwise stated.

| 항목 | 값 |
| --- | --- |
| `M_B` | `16` |
| `M_Q_min` | `4` |
| `N_local_min` | `4` |
| `local_search_max` | task/method-specific; see Table below |
| `action_bins_per_dim` | `5` |
| `policy_support_mode` | `bin` |
| `lambda_support` | `1.0` |
| `weighting_mode` | `batch_norm` |
| `weight_scope` | `critic_only` for main SABER |
| `support_scan_mode` | `deterministic_tail_v1` |
| support cache | enabled |
| support cache max entries | `200,000` |
| local scan cache max entries | `50,000` |
| Numba support scan | enabled when available |
| predicate early exit | enabled in runtime-opt rows |
| support diagnostic audit interval | `100` |

`N_gap_min=128` is present in several configs because older gap-weight variants share the same config class. For the support-only runtime-opt SABER rows, the actual support-positive predicate is determined by `N_local_min`, `M_avail` (legacy config `M_Q_min`), action-bin support, and the local scan. `N_gap_min` is not part of the main support predicate.

Task/method-specific support settings used by the rows currently referenced in the paper:

| Task / row group | `N_local_min` | `M_avail` / legacy `M_Q_min` | `M_B` | `local_search_max` | `lambda_support` | action binning |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FetchPickAndPlace DDPG/TD3/SAC SABER main rows | `4` | `4` | `16` | `512` | `1.0` | `5` bins per action dimension, mixed-radix id |
| FetchSlide DDPG `b3_critic_only_runtime_opt_v5` main row | `4` | `4` | `16` | `1024` | `1.0` | `5` bins per action dimension, mixed-radix id |
| FetchSlide TD3/SAC SABER rows used in the current draft | `4` | `4` | `16` | `512` | `1.0` | `5` bins per action dimension, mixed-radix id |
| Meta-World DrawerOpen TD3 SABER/control rows | `4` | `4` | `16` | `512` | `1.0` | `5` bins per action dimension, mixed-radix id |

### 4.2 Action binning

Continuous action is normalized by environment action bounds, then discretized independently per dimension.

```text
norm_action = (action - center) / scale
scaled = (norm_action + 1) * 0.5 * bins_per_dim
bin_d = floor(clip(scaled_d, 0, bins_per_dim - 1))
action_bin = sum_d bin_d * bins_per_dim^d
```

Main runs use `bins_per_dim=5`.

### 4.3 Support predicate pseudocode

```text
Input:
  batch sample i = (o_i, g_i^tr, h_i, a_i^r)
  replay buffer D
  current policy pi

1. coarse_key_i = C_coarse(o_i, h_i)
2. candidate_indices = most recent transitions in D
   with the same coarse_key_i, capped by local_search_max
3. full_key_i = C_full(o_i, g_i^tr, h_i)
4. local_pool L_i = {
     j in candidate_indices:
       C_full(o_j, g_i^tr, h_j) == full_key_i
   }
5. a_i^pi = pi(o_i, g_i^tr)
6. b_i^pi = action_bin(a_i^pi)
7. B_i = unique action bins of replay actions in L_i
   n_i^avail = min(|L_i|, M_B)
8. support_positive psi_i = 1 iff:
     |L_i| >= N_local_min
     and n_i^avail >= M_avail
     and b_i^pi in B_i
9. raw weight = 1 + lambda_support * psi_i
10. batch-normalize raw weights to mean approximately 1
11. multiply only critic Bellman residual by the normalized weight
```

중요: `psi_i`는 current policy action의 support를 본다. 그러나 critic residual은 sampled replay action `a_i^r`에서 계산된다. 논문에서는 이를 direct target-noise estimator가 아니라 `policy-improvement-relevant reliability proxy`로 해석해야 한다.

구현 config의 `M_Q_min`은 위 pseudocode의 `M_avail`에 해당한다. 이는 Q-value threshold나 unique action-bin diversity threshold가 아니라, `M_B` cap 이후 사용할 수 있는 local transition count의 최소 문턱이다. 현재 main FPP 설정에서는 `N_local_min=4`, `M_Q_min=4`, `M_B=16`이므로 count 조건은 effectively `|L_i| >= 4`이고, support-positive를 가르는 나머지 핵심 조건은 `b_i^pi in B_i`이다.

## 5. Task-specific Support Context

### 5.1 FetchPickAndPlace context

Observation layout:

| feature | slice |
| --- | --- |
| end-effector position | `obs[0:3]` |
| object position | `obs[3:6]` |
| gripper gap | `obs[9] + obs[10]` |

Phase:

```text
approach if d_hand_obj > d_reach
grasp    if object_z <= z_lift_min and gripper_gap > grip_close_thresh
lift     if object_z <= z_lift_min
place    otherwise
```

Parameters:

| 항목 | 값 |
| --- | --- |
| `d_reach` | `0.08` |
| `d_goal_near` | `0.05` |
| `z_lift_min` | `0.48` |
| `grip_close_thresh` | `0.04` |
| hand-object bins | `(0.03, 0.05, 0.08, 0.12, 0.20, 0.40)` |
| object-goal bins | `(0.05, 0.10, 0.20, 0.40, 0.70)` |
| object-z bins | `(0.43, 0.48, 0.55, 0.65, 0.75)` |
| gripper-gap bins | `(0.02, 0.05, 0.08)` |
| horizon bins | `(10, 25, 50, 100)` |

Coarse index:

```text
hand-object distance bin, object-z bin, gripper-gap bin, horizon bin
```

Full context:

```text
phase, hand-object distance bin, object-goal distance bin,
object-z bin, gripper-gap bin, horizon bin
```

### 5.2 FetchSlide context

Observation layout:

| feature | slice |
| --- | --- |
| end-effector position | `obs[0:3]` |
| object position | `obs[3:6]` |
| object velocity | `obs[14:17]` |
| end-effector velocity | `obs[20:23]` |

Phase:

```text
near_goal if d_obj_goal <= d_goal_near
slide     if object_speed_xy >= v_slide_min
strike    if d_ee_obj <= d_contact
approach  otherwise
```

Parameters:

| 항목 | 값 |
| --- | --- |
| `d_contact` | `0.08` |
| `d_goal_near` | `0.05` |
| `v_slide_min` | `0.05` |
| `v_fast` | `0.15` |
| ee-object bins | `(0.05, 0.10, 0.20, 0.40)` |
| object-goal bins | `(0.05, 0.10, 0.20, 0.40, 0.70, 1.00)` |
| object-speed bins | `(0.02, 0.05, 0.10, 0.20, 0.40)` |
| velocity-alignment bins | `(-0.50, 0.0, 0.50, 0.80)` |
| horizon bins | `(10, 25, 50, 100)` |

Velocity alignment is the cosine alignment between object xy velocity and the object-to-goal xy direction, clipped to `[-1, 1]`.

Coarse index:

```text
ee-object distance bin, object speed bin, horizon bin
```

Full context:

```text
phase, ee-object distance bin, object-goal distance bin,
object speed bin, velocity alignment bin, horizon bin
```

### 5.3 Meta-World DrawerOpen context

Observation layout:

| feature | slice |
| --- | --- |
| end-effector position | `obs[0:3]` |
| gripper | `obs[3:4]` |
| drawer handle position | `obs[4:7]` |

Phase:

```text
approach if d_hand_handle > d_reach
contact  if hand is near handle
opening  if hand is near and handle-goal distance <= d_opening_mid
open     if handle-goal distance <= d_goal_near
```

Parameters:

| 항목 | 값 |
| --- | --- |
| `d_reach` | `0.08` |
| `d_goal_near` | `0.05` |
| `d_opening_mid` | `0.12` |
| hand-handle bins | `(0.03, 0.05, 0.08, 0.12, 0.20, 0.40)` |
| handle-goal bins | `(0.03, 0.05, 0.08, 0.12, 0.20, 0.40, 0.70)` |
| handle-y bins | `(-0.05, 0.00, 0.05, 0.10, 0.15, 0.25)` |
| gripper bins | `(-0.50, 0.0, 0.50)` |
| horizon bins | `(10, 25, 50, 100, 200, 500)` |

Coarse index:

```text
hand-handle distance bin, handle-y bin, gripper bin, horizon bin
```

Full context:

```text
phase, hand-handle distance bin, handle-goal distance bin,
handle-y bin, gripper bin, horizon bin
```

## 6. Method Rows and Scope

### 6.1 Main paper evidence

Main paper에서 가장 방어 가능한 primary evidence는 FetchPickAndPlace다.

| Task | Backbone | Baseline | SABER row |
| --- | --- | --- | --- |
| FetchPickAndPlace | DDPG | HER baseline | critic-only SABER runtime-opt |
| FetchPickAndPlace | TD3 | HER baseline | critic-only SABER |
| FetchPickAndPlace | SAC | HER baseline | critic-only SABER |

SABER main row는 replay sampling과 actor objective를 바꾸지 않는다. Weight is applied only to critic Bellman residuals.

### 6.2 Secondary evidence

FetchSlide는 backbone-dependent limitation을 보여주는 secondary evidence로 둔다. TD3/SAC는 개선이 있으나, DDPG는 실패 또는 악화가 있으므로 universal improvement claim에 사용하면 안 된다.

### 6.3 Stress-test evidence

DrawerOpen은 Meta-World external stress-test와 random/shuffled/actor-scope control 분석에 사용한다. Random-Support가 strong control로 나타나므로, structured support mechanism의 primary evidence로 쓰지 않는다.

## 7. Metrics and Runtime Measurement

Evaluation checkpoint의 success는 `eval_episodes=20` episode에서 environment success의 mean이다. Seed `s`의 evaluation history는 `(step, success)` pairs로 저장되며, 이를 \(\{(t_k,q_k^{(s)})\}_{k=1}^{K_s}\)로 둔다.

| Metric | 정의 |
| --- | --- |
| final success | \(\mathrm{Final}^{(s)}=q_{K_s}^{(s)}\) |
| best success | \(\mathrm{Best}^{(s)}=\max_k q_k^{(s)}\) |
| success AUC | selected evaluation checkpoints에 대한 normalized trapezoidal area |
| last-quarter AUC | \(t_k \ge 0.75T\) checkpoints에 대한 normalized trapezoidal AUC; denominator는 \(T/4\) |
| Hit90 first-hit | \(H_{0.90}^{(s)}=\min\{t_k:q_k^{(s)}\ge0.90\}\); unreached seed는 undefined |
| Hit90 Seeds | \(H_{0.90}^{(s)}\)가 defined인 seed 수 |
| steps/sec | total env steps divided by recorded training wall-clock runtime |

Discrete AUC는 다음 식을 사용한다.

\[
\widehat{\mathrm{AUC}}^{(s)}(I;Z)
=
\frac{1}{Z}
\sum_{\ell=1}^{m-1}
\frac{q_{j_\ell}^{(s)}+q_{j_{\ell+1}}^{(s)}}{2}
(t_{j_{\ell+1}}-t_{j_\ell}),
\quad I=\{j_1,\ldots,j_m\}.
\]

Implementation notes:

- FPP `_auc_success` uses the span between the first and last selected evaluation steps as `Z`.
- DrawerOpen summaries use the same selected-step span convention.
- FetchSlide stores both `auc_eval_mean` and `auc_trapz_0_to_total`; the current paper table uses the task analyzer's selected normalized AUC.
- Last-quarter AUC uses selected checkpoints with `step >= int(0.75 * total_steps)` and `Z=total_steps/4`.
- Cross-task AUC values should not be compared as absolute quantities unless all tasks are recomputed with one normalization convention.

All reported `mean +/- std` values use seeds 1--5 and sample standard deviation, not standard error:

\[
\bar m=\frac{1}{5}\sum_{s=1}^{5}m^{(s)}, \qquad
\mathrm{std}(m)=
\sqrt{\frac{1}{4}\sum_{s=1}^{5}(m^{(s)}-\bar m)^2}.
\]

Runtime claim은 sample efficiency와 분리한다. SABER support lookup은 training-time cost를 늘리지만, inference-time policy architecture는 baseline과 동일하다. `steps/sec`는 runner가 기록한 wall-clock runtime 기준이므로 support lookup overhead를 포함하며, runner에 따라 evaluation/logging overhead도 포함될 수 있다. 따라서 hardware-normalized wall-clock superiority claim에는 사용하지 않는다.

## 8. Current Artifact Gaps

현재 repository artifact 기준으로 아직 완전히 닫히지 않은 재현성 항목은 다음이다.

1. Package versions가 모든 run에 저장되어 있지 않다.
   - FetchSlide runner는 `collect_machine_metadata()`를 갖고 있지만, 일부 `env_snapshot.json`/`summary.json`에는 `machine=null`이다.
   - FPP/DrawerOpen old results도 Python, PyTorch, Gymnasium Robotics, MuJoCo, Meta-World exact version이 일관되게 저장되어 있지 않다.

2. Hardware 정보가 일부 run에 저장되어 있지 않다.
   - steps/sec는 결과 표에 남아 있지만 GPU/CPU 모델, thread count, CUDA version이 모든 run에서 닫히지 않는다.

3. FPP SAC와 DDPG/TD3의 eval frequency가 다를 수 있다.
   - DDPG/TD3 FPP configs는 `eval_freq=10000`.
   - FPP SAC config는 `eval_freq=25000`.
   - 논문 table caption 또는 appendix에서 backbone별 eval grid 차이를 명시해야 한다.

4. Result aggregation script의 exact path를 최종 고정해야 한다.
   - 표에 들어가는 AUC/final/best/Hit90 값이 어느 `summary.json` 또는 aggregation script에서 나왔는지 table별 provenance를 붙여야 한다.

5. Random/Shuffled/PER controls의 task coverage가 아직 불균형하다.
   - DrawerOpen controls는 풍부하지만, FPP/FetchSlide main-task controls가 충분한지 다음 단계에서 artifact inventory로 확인해야 한다.

## 9. 최종 논문 Appendix에 넣을 최소 문장

최종 원고 appendix에는 아래 내용을 반드시 포함한다.

```text
All experiments use uniform replay with future-goal HER relabeling
(HER ratio 0.8). Rewards are recomputed after relabeling, and success
states are not treated as terminal for bootstrapping. SABER changes only
the critic loss weight unless explicitly marked otherwise; replay
sampling and actor objectives remain identical to the corresponding
HER backbone.
```

```text
For each sampled transition, SABER constructs a task-specific local
state-goal-horizon context, scans recent replay transitions in the
matching coarse bucket, and marks a sample as support-positive only
when the exact local pool is sufficiently populated and contains the
current policy action bin. The resulting raw weight 1 + lambda_S psi_i
is normalized within the mini-batch and applied to critic TD loss only.
```

```text
We report wall-clock throughput separately from sample-efficiency
metrics because SABER's support lookup is a training-time cost. The
deployed policy network is unchanged from the baseline actor-critic
backbone.
```

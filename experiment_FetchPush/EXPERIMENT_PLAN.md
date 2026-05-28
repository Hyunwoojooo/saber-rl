# EXPERIMENT_PLAN

# MetaCog-Lite v0 Fetch Benchmark Experiment Plan

## 0. Goal

현재 실험의 목적은 MetaCog-Lite v0의 최종 방법론을 바로 확정하는 것이 아니라, Fetch 계열 benchmark에서 B3 practice-weighting 계열이 어떤 조건에서 의미 있게 작동하는지 분리해서 검증하는 것이다.

핵심 질문은 세 가지다.

1. B3 support-only weighting이 DDPG+HER baseline보다 실제로 유리한가?
2. B3의 이득은 support-positive sample을 정확히 강조해서 생기는가, 아니면 단순한 loss-scale / weight-distribution 효과인가?
3. B3의 계산 비용을 줄여도 성능과 해석을 유지할 수 있는가?

모든 실험은 MetaCog-Lite v0 범위 안에서 진행한다.

```
DDPG + HER
+ replay-action support based train-time practice weighting
+ detached, batch-normalized loss weighting
```

사용하지 않는 것:

```
SKIP / EXPLORE / PROBE / Q_F / Q_aug / dynamics / rho / w_C / m_upd
old scalar kNN S_Q support
```

---

## 1. What Is B3?

B3는 MetaCog-Lite v0에서 가장 단순한 replay-action support 기반 practice-weighting 방법이다.

핵심 아이디어:

```
현재 HER train item이 replay buffer 안에서 "비슷한 local context"를 충분히 가지고 있고,
현재 policy action과 같은 action-bin이 그 local context 안에서 실제로 실행된 적이 있으면,
그 train item은 더 믿을 만한 practice target으로 보고 loss weight를 높인다.
```

즉 B3는 Q-gap magnitude를 쓰지 않는다.

```
B3 uses support availability only.
B3 does not use G_sup magnitude.
B3 does not require candidate Q evaluation.
```

### 1.1 Local Replay Support

각 train item에 대해 HER relabeled goal `g_train` 기준으로 local context를 만든다.

```
query = (observation, g_train, horizon)
```

Replay buffer에서 같은 local context에 속하는 transition들을 찾는다.

```
A_Q_replay_local = replay actions executed in matching local context
```

여기서 중요한 점:

```
그 replay action이 현재 g_train을 목표로 실행되었다고 주장하지 않는다.
그 action은 HER relabeling 이후 같은 local training context로 해석되는 state에서 실제로 실행된 action이다.
```

### 1.2 B3 Support Signal

B3의 support-positive 조건은 다음과 같다.

```
local pool size >= N_local_min
sampled/available support size >= M_Q_min
policy action bin exists in full local replay action-bin set
```

이를 하나의 indicator로 쓴다.

```
psi_support_i = 1 if train item i is locally supported
psi_support_i = 0 otherwise
```

코드/로그에서는 보통 다음 값들이 이 support signal을 설명한다.

```
psi_q_rank
psi_A_policy
q_candidate_fallback
len_A_Q_sup
len_A_Q_replay_local
```

### 1.3 B3 Weight

B3 raw practice weight는 다음과 같다.

```
w_i = 1 + lambda_support * psi_support_i
```

그리고 batch-normalization을 적용한다.

```
Omega_i = w_i / mean_batch(w)
```

따라서 B3는 support-positive sample의 학습 비중을 높이지만, support-negative sample을 완전히 버리지는 않는다.

```
support-positive:
  higher weight

support-negative:
  still trained, but relatively lower weight after batch normalization
```

### 1.4 B3 Actor-Critic vs Critic-Only

기존 B3-original은 actor와 critic loss 모두에 같은 weight를 적용한다.

```
L_Q     = mean(Omega_i * TD_error_i^2)
L_actor = -mean(Omega_i * Q(s_i, pi(s_i)))
```

B3-critic-only는 critic loss에만 weight를 적용한다.

```
L_Q     = mean(Omega_i * TD_error_i^2)
L_actor = -mean(Q(s_i, pi(s_i)))
```

해석상 B3-critic-only가 더 깔끔할 수 있다.

```
support signal = critic target reliability signal
```

즉 support가 풍부한 train item은 critic target을 더 신뢰할 수 있다고 보고 critic을 더 강하게 학습시키지만, actor update에는 직접적인 sample weighting을 걸지 않는다.

### 1.6 B3-random / B3-shuffled Controls

B3-random과 B3-shuffled는 B3가 좋은 이유를 검증하기 위한 control이다.

```
B3-random:
  B3와 같은 positive count K를 유지
  하지만 random K개 sample에 high weight 부여

B3-shuffled:
  B3 raw weight vector를 그대로 만든 뒤 batch 안에서 permutation
  weight distribution은 B3와 동일
  support-positive item과 high weight의 alignment만 깨짐
```

이 control들이 필요한 이유:

```
B3가 좋은 이유가 support-positive sample을 정확히 강조해서인지,
아니면 단순히 batch 안에 high weight sample이 생기는 regularization / loss-scale 효과 때문인지 분리하기 위해서다.
```

---

## 2. Benchmark Scope

대상 benchmark:

```
FetchPush-v4
FetchSlide-v4
FetchPickAndPlace-v4
```

공통 설정:

```
reward_type = sparse
base learner = DDPG + HER
HER ratio = 0.8
batch size = 256
updates_per_step = 1
eval metric = success rate
primary comparison metric = Eval AUC + final/best success
runtime metric = wall-clock seconds, steps/sec, AUC/hour
```

비교 정당성을 위해 같은 benchmark 내부의 모든 비교군은 가능한 한 같은 machine / library / code snapshot에서 실행한다.

---

## 3. Machine Allocation Principle

가동 가능한 실험 컴퓨터가 3대이므로 benchmark 단위로 분리한다.

권장 배치:

```
Machine A: FetchPush only
Machine B: FetchPickAndPlace only
Machine C: FetchSlide only
```

이렇게 하면 benchmark 내부 비교의 정당성이 유지된다.

중요:

```
성능 비교: 같은 seed set, 같은 env/library, 같은 code snapshot이면 machine 분리 가능
runtime 비교: 반드시 같은 machine 안에서 비교
```

따라서 `original vs runtime optimization` 비교는 같은 benchmark와 같은 machine에서만 해석한다.

---

## 4. Experiment Family A: B3 Ablation Suite

목적:

```
B3 practice-weighting signal이 왜 작동하는지 분해한다.
```

대상 benchmark:

```
FetchPush
FetchSlide
FetchPickAndPlace
```

실험군:

```
B0
B3-original
B3-critic-only
B3-random
B3-shuffled
B3-original-runtime-optimization
B3-fast-256
B3-fast-1024
```

`B3-original actor-critic` is a required reference, not an optional run.

### 4.1 B0

의미:

```
clean DDPG + HER baseline
```

동작:

```
Omega = 1
MetaCog-Lite diagnostic weighting 없음
local replay support lookup 없음
```

비교 목적:

```
B3 계열이 baseline보다 실제로 성능/학습효율을 개선하는지 확인
```

### 4.2 B3-critic-only

의미:

```
B3 support-only weight를 critic loss에만 적용
actor loss는 uniform하게 유지
```

동작:

```
w_i = 1 + lambda_support * psi_support_i
Omega_critic = normalize_batch(w)
Omega_actor = 1
```

해석:

```
support signal을 "critic target reliability"로 해석한다.
actor까지 직접 가중하지 않으므로 방법론 해석이 더 깔끔할 수 있다.
```

주요 비교:

```
B3-critic-only vs B0
B3-critic-only vs B3-original actor-critic
```

### 4.3 B3-random

의미:

```
B3와 같은 high-weight 개수를 유지하지만, 어떤 sample에 high weight가 붙는지는 random
```

정확한 구현 원칙:

```
1. 원래 B3 support-only diagnostics 계산
2. batch 안의 support-positive count K 계산
3. 같은 batch에서 random K개 sample 선택
4. 선택된 K개에 high weight 부여
5. loss에는 random assigned weight에서 나온 Omega 사용
6. psi_q_rank / psi_A_policy / fallback / len_A_Q_* diagnostics는 원래 B3 기준으로 유지
```

비교 목적:

```
B3의 이득이 단순히 high-weight sample 개수 때문인지 검증
```

기대 해석:

```
B3-critic-only > B3-random이면 support-positive item을 정확히 강조하는 것이 중요하다는 근거
B3-random도 강하면 B3 효과 중 일부가 batch-level loss scaling / regularization 효과일 수 있음
```

### 4.4 B3-shuffled

의미:

```
B3와 같은 raw weight vector를 만들고 batch 안에서 permutation
```

정확한 구현 원칙:

```
1. 원래 B3 support-only raw weight vector 계산
2. batch 안에서 w_raw permutation
3. weight distribution은 B3와 동일
4. support-positive item과 high weight의 alignment만 깨짐
5. loss에는 shuffled assigned weight에서 나온 Omega 사용
```

비교 목적:

```
B3의 이득이 support alignment 때문인지, weight distribution 자체 때문인지 분리
```

필수 로그:

```
mean_w_P_reference
mean_w_P
reference_high_weight_rate
assigned_high_weight_rate
support_weight_alignment_rate
```

여기서:

```
support_weight_alignment_rate = P(reference_support_positive = 1 | assigned_high_weight = 1)
```

### 4.5 B3-original-runtime-optimization

의미:

```
B3-original과 같은 semantics를 유지하면서 lookup 계산만 최적화
```

허용되는 최적화:

```
versioned support cache
deterministic local scan cache
exact negative short-circuit
candidate construction 최적화
```

금지:

```
support 정의 변경
diagnostic subsampling을 default method로 사용
M_B / M_Q_min / N_local_min 의미 변경
policy support rule 변경
```

비교 목적:

```
B3-original과 성능이 유지되는지
runtime이 얼마나 줄어드는지
```

주요 지표:

```
steps/sec
wall_clock_seconds
time_local_context_lookup
time_candidate_construction
support_cache_hit_rate
short_circuit_rate
local_scan_truncated_rate
Eval AUC
```

### 4.6 B3-fast-256 / B3-fast-1024

의미:

```
local_search_max를 바꿔 runtime-performance trade-off를 확인
```

설정:

```
B3-fast-256:
  local_search_max = 256

B3-fast-1024:
  local_search_max = 1024
```

benchmark별 기본값은 별도로 명시한다. 예를 들어 FetchPush에서 기본 비교값이 512라면:

```
B3-original = local_search_max 512
B3-random = local_search_max 512
B3-shuffled = local_search_max 512
B3-fast-256 = local_search_max 256
B3-fast-1024 = local_search_max 1024
```

Naming rule:

```
B3-fast-N always means local_search_max = N.
```

Therefore, if another benchmark needs a different fast budget, use a new explicit name.

```
B3-fast-512
B3-fast-2048
```

Do not use `B3-fast` without the search budget in the run name.

비교 목적:

```
search budget을 줄여도 성능이 유지되는지
search budget을 늘렸을 때 성능이 좋아지는지
runtime 증가가 정당화되는지
```

---

## 5. Experiment Family B: Original vs Runtime Optimization

목적:

```
B3의 의미를 바꾸지 않는 runtime optimization이 실제로 유효한지 검증
```

대상:

```
FetchPush
FetchPickAndPlace
FetchSlide
```

비교군:

```
B3-original
B3-original-runtime-optimization
```

중요:

```
이 비교는 runtime claim을 포함하므로 반드시 같은 machine에서 수행
```

성공 조건:

```
Eval AUC가 B3-original과 동등하거나 큰 차이 없음
final/best success가 크게 무너지지 않음
steps/sec 개선
wall_clock_seconds 감소
support diagnostics가 비슷한 범위 유지
```

필수 로그:

```
time_local_context_lookup
time_candidate_construction
time_actor_critic_update
local_scan_truncated_rate
mean_len_A_Q_replay_local
mean_len_A_Q_sup
psi_q_rank_rate
q_candidate_fallback_rate
support_cache_hit_rate
short_circuit_rate
```

---

## 6. Experiment Family C: Baseline Comparison

목적:

```
B3-critic-only가 단순한 replay prioritization 또는 TD-error weighting과 다른지 비교
```

비교군:

```
B0 Baseline
B3-critic-only
TD-error baseline
PER-style baseline
```

### 6.1 B0 Baseline

```
DDPG + HER
Omega = 1
```

역할:

```
모든 weighted method의 기준선
```

### 6.2 B3-critic-only

```
support-positive sample의 critic loss만 강화
```

역할:

```
MetaCog-Lite v0 최종 후보 method
```

### 6.3 TD-error baseline

의미:

```
현재 batch에서 TD-error가 큰 sample에 더 큰 critic weight를 부여
```

예시 정의:

```
e_i = |Q(s_i, a_i) - y_i|
w_i = f(e_i)
Omega_critic = normalize_batch(w)
Omega_actor = 1
```

주의:

```
TD-error는 critic이 틀린 정도이지, support-rich context를 의미하지 않는다.
```

비교 목적:

```
B3가 단순히 "큰 error sample을 더 학습"하는 방법보다 나은지 확인
```

### 6.4 PER-style baseline

의미:

```
Prioritized Experience Replay와 유사하게 TD-error 기반 priority를 사용
```

가능한 구현 수준:

```
Option A: sampling probability를 TD-error priority로 변경
Option B: uniform sampling은 유지하고 PER-style importance weight만 적용
```

초기 추천:

```
Option B부터 구현
```

이유:

```
기존 DDPG+HER replay sampling rule을 크게 바꾸지 않고, loss weighting baseline으로 비교하기 쉽다.
```

비교 목적:

```
B3 support signal이 TD-error priority와 다른 정보를 주는지 확인
```

---

## 7. Recommended Seed / Step Plan

기본 seed:

```
seeds = 1,2,3,4,5
```

Run steps:

```
1M extension
```

---

## 10. Implementation Hygiene

앞으로 benchmark / method별 코드는 덮어쓰지 않고 폴더를 분리한다.

예시:

```
experiment/
  FetchPush/
    B0/
    B3-critic-only/
    B3-original/
    B3-original-runtime-optimization/
    B3-fast-256/
    B3-fast-1024/
    B3-random/
    B3-shuffled/
    TD-error/
    PER-style/

  FetchPickAndPlace/
    ...

  FetchSlide/
    ...
```

각 run directory에는 최소한 아래 파일을 남긴다.

```
config.json
env_snapshot.json
metrics.jsonl
metrics.csv
summary.json
run.log
best_checkpoint.json
```

각 실험군에는 별도 code snapshot 또는 commit hash를 기록한다.

---

## 12. Required Clarifications Before Implementation

This section is the finalized addendum for code implementation and run-command generation. If any earlier section is ambiguous, this section takes priority.

### 12.1 B3-original Is the Required Reference

`B3-original` is not optional. It must be run and reported for every benchmark where any B3 variant is evaluated.

```
B3-original:
  method = b3_support_only
  weight_scope = actor_critic
  weighting_mode = batch_norm
```

Every B3-family result must be interpreted against the same-benchmark B3-original result.

```
B3-critic-only:
  compare against B0 and B3-original

B3-random:
  compare against B3-original

B3-shuffled:
  compare against B3-original

B3-runtime-optimization:
  compare against B3-original

B3-fast:
  compare against B3-original at the benchmark default local_search_max
```

### 12.2 Benchmark Default `local_search_max`

Default values are fixed as follows.

```
FetchPush:
  default local_search_max = 512
  B3-original = 512
  B3-critic-only = 512
  B3-random = 512
  B3-shuffled = 512
  B3-runtime-optimization = 512
  B3-fast-256 = 256
  B3-fast-1024 = 1024

FetchPickAndPlace:
  default local_search_max = 1024
  B3-original = 1024
  B3-critic-only = 1024
  B3-random = 1024
  B3-shuffled = 1024
  B3-runtime-optimization = 1024
  additional B3-fast-512 = 512
  additional B3-fast-2048 = 2048

FetchSlide:
  default local_search_max = 1024
  B3-original = 1024
  B3-critic-only = 1024
  B3-random = 1024
  B3-shuffled = 1024
  B3-runtime-optimization = 1024
  additional B3-fast-512 = 512
  additional B3-fast-2048 = 2048
```

### 12.3 TD-error Baseline Formula

The TD-error baseline is a critic-only loss-weighting baseline. It does not use replay-action support.

For each sampled train item:

```
y_i = r_i + gamma * (1 - done_i) * Q_target(s_next_i, pi_target(s_next_i))
td_i = stopgrad(abs(Q(s_i, a_i) - y_i))
td_norm_i = td_i / (mean_batch(td) + eps)
u_i = clip(td_norm_i, 0, U_TD_max)
w_i = 1 + lambda_TD * u_i
Omega_critic_i = w_i / mean_batch(w)
Omega_actor_i = 1
```

Default values:

```
lambda_TD = 1.0
U_TD_max = 1.0
eps = 1e-8
```

Interpretation:

```
TD-error baseline asks whether B3 is better than simply emphasizing high Bellman-error samples.
```

### 12.4 PER-style Baseline Formula

The initial PER-style baseline should be implemented as loss weighting, not as full replay sampling replacement.

For each sampled train item:

```
y_i = r_i + gamma * (1 - done_i) * Q_target(s_next_i, pi_target(s_next_i))
td_i = stopgrad(abs(Q(s_i, a_i) - y_i))
p_i = (td_i + eps_PER) ** alpha
p_norm_i = p_i / (mean_batch(p) + eps)
w_i = clip(p_norm_i, 0, U_PER_max)
Omega_critic_i = w_i / mean_batch(w)
Omega_actor_i = 1
```

Default values:

```
alpha = 0.6
eps_PER = 1e-6
U_PER_max = 2.0
eps = 1e-8
```

Naming:

```
Use "PER-style loss-weighting baseline".
Do not call it full PER unless replay sampling probabilities are changed.
```

### 12.5 Runtime Optimization Correctness Test

Runtime optimization must preserve B3 semantics. Before using optimized results as main results, run an exact correctness test.

Split correctness into two levels.

```
Level A: semantic equivalence
  blocking requirement for main results

Level B: diagnostic equivalence
  required for analysis claims and debugging reports
```

### 12.5.1 Semantic equivalence

Semantic equivalence means the optimized implementation produces the same training weights and therefore the same loss semantics as naive B3-original.

Required semantic test:

```
1. Build one replay buffer snapshot.
2. Sample one fixed HER train batch.
3. Compute B3-original weights with the naive lookup path.
4. Compute B3-runtime-optimization weights with the optimized lookup path.
5. Compare the loss-semantic fields:

   psi_q_rank
   psi_A_policy
   q_candidate_fallback
   w_P
   Omega_critic
   Omega_actor

6. Values must match within numerical tolerance.
```

If local scan is truncated:

```
The scan subset must be deterministic for the correctness test.
Otherwise the optimized run is profiling-only, not a main result.
```

### 12.5.2 Diagnostic equivalence

Diagnostic equivalence means the optimized implementation reports the same interpretability diagnostics as the naive implementation.

Required diagnostic audit fields:

```
len_A_Q_sup
len_A_Q_replay_local
num_coarse_candidates
num_scanned_candidates
local_scan_truncated
psi_q_rank
psi_A_policy
q_candidate_fallback
```

For exact final reporting:

```
diagnostic fields should match naive B3-original on fixed audit batches
```

If an optimization uses short-circuiting and intentionally skips expensive diagnostic computation:

```
training semantics can still be valid
but diagnostic fields must be marked as approximate or recomputed in a separate audit pass
```

Do not use approximate diagnostic fields to support claims about local support distribution unless an exact audit pass is provided.

Strictly forbidden in runtime optimization:

```
cache Q values
cache TD targets
cache G_sup
change support key
change action-bin rule
change local_search_max except in B3-fast experiments
```

### 12.6 Stage Priority and Compute Budget

Run stages:

```
Stage 0: document/code freeze
  output = one code snapshot per method/benchmark

Stage 1: smoke/correctness
  steps = 10k or 20k
  seeds = 1
  purpose = CLI, logs, summaries, correctness tests

Stage 2: first-pass result
  steps = 1M
  seeds = 1,2,3,4,5
  purpose = ranking, variance, failure modes

Stage 3: final extension
  steps = 1M
  seeds = 1,2,3,4,5
  purpose = final paper-quality curves for selected methods
```

Compute priority:

```
Priority 1:
  B0
  B3-original
  B3-critic-only

Priority 2:
  B3-random
  B3-shuffled

Priority 3:
  B3-runtime-optimization
  B3-fast-256 / B3-fast-1024

Priority 4:
  TD-error baseline
  PER-style baseline
```

If the paper angle becomes primarily runtime reduction, move Priority 3 above Priority 2.

### 12.7 Metric Definitions

### Eval AUC

Use normalized area under the evaluation success curve.

```
eval points = (t_k, s_k)
t_k = environment step
s_k = eval success rate

Eval AUC = sum_k 0.5 * (s_k + s_{k+1}) * (t_{k+1} - t_k) / total_steps
```

Why:

```
final success is sensitive to the last eval point
best success is sensitive to lucky peaks
Eval AUC better reflects learning efficiency over the full run
```

### Steps-to-threshold

Definition:

```
steps_to_threshold = first eval step where eval_success_rate >= threshold
```

Recommended thresholds:

```
FetchPush: 0.8
FetchPickAndPlace: 0.5
FetchSlide: 0.5 or 0.6
```

If threshold is never reached:

```
steps_to_threshold = null
```

### Runtime aggregate

Required runtime fields:

```
wall_clock_seconds
steps_per_second = total_steps / wall_clock_seconds
AUC_per_hour = Eval_AUC / (wall_clock_seconds / 3600)
runtime_overhead_vs_B0 = wall_clock_seconds_method / wall_clock_seconds_B0
runtime_overhead_vs_B3_original = wall_clock_seconds_method / wall_clock_seconds_B3_original
```

Runtime comparison is valid only within the same benchmark and same machine.

### 12.8 Random/Shuffled Reference Scope

Default random/shuffled controls are controls for actor-critic B3-original.

```
B3-random:
  reference = B3-original actor_critic
  assigned random weights are applied to critic and actor

B3-shuffled:
  reference = B3-original actor_critic
  assigned shuffled weights are applied to critic and actor
```

Therefore:

```
Omega_critic = normalize_batch(assigned_w)
Omega_actor = normalize_batch(assigned_w)
```

If we need critic-only controls, use separate names:

```
B3-critic-random
B3-critic-shuffled
```

Those are not the default random/shuffled experiments.

### 12.9 Execution Matrix

Use this matrix when creating code folders and run commands.

### 12.9.1 Core matrix

| Method | Required? | Reference | Weight scope | Support lookup | local_search_max | Main purpose |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | yes | none | none | no | n/a | DDPG+HER baseline |
| B3-original | yes | B0 | actor_critic | yes | benchmark default | required B3 reference |
| B3-critic-only | yes | B0, B3-original | critic_only | yes | benchmark default | main candidate |
| B3-random | yes | B3-original | actor_critic | yes, reference diagnostics | benchmark default | positive-count control |
| B3-shuffled | yes | B3-original | actor_critic | yes, reference diagnostics | benchmark default | weight-distribution control |
| B3-runtime-optimization | yes for runtime claim | B3-original | same as B3-original | optimized exact path | benchmark default | runtime reduction |
| B3-fast-256 | FetchPush yes | B3-original | chosen B3 scope | yes | 256 | search-budget sensitivity |
| B3-fast-1024 | FetchPush yes | B3-original | chosen B3 scope | yes | 1024 | search-budget sensitivity |
| TD-error | later baseline | B0, B3-critic-only | critic_only | no | n/a | high TD-error baseline |
| PER-style | later baseline | B0, B3-critic-only | critic_only | no | n/a | PER-like weighting baseline |

### 12.9.2 Benchmark matrix

| Benchmark | Machine policy | Default local_search_max | Required first-pass methods | Additional fast sweep |
| --- | --- | --- | --- | --- |
| FetchPush | one machine only | 512 | B0, B3-original, B3-critic-only, B3-random, B3-shuffled, B3-runtime-optimization, B3-fast-256, B3-fast-1024 | already included |
| FetchPickAndPlace | one machine only | 1024 | B0, B3-original, B3-critic-only, B3-random, B3-shuffled, B3-runtime-optimization | B3-fast-512, B3-fast-2048 |
| FetchSlide | one machine only | 1024 | B0, B3-original, B3-critic-only, B3-random, B3-shuffled, B3-runtime-optimization | B3-fast-512, B3-fast-2048 |

### 12.9.3 Run naming convention

Run names must encode benchmark, method, seed, steps, and local search budget.

```
{benchmark}_{method}_s{seed}_{steps}_ls{local_search_max}
```

Examples:

```
fetch_push_b3_original_s1_1000000_ls512
fetch_push_b3_critic_only_s1_1000000_ls512
fetch_push_b3_fast_256_s1_1000000_ls256
fetch_push_b3_fast_1024_s1_1000000_ls1024
fetch_pick_and_place_b3_shuffled_s1_1000000_ls1024
fetch_slide_b3_runtime_opt_s1_1000000_ls1024
```

For B0, either omit `ls` or set it to the benchmark default for bookkeeping. It must not affect training.

---

```markdown
# SABER-RL Experiment Metrics Specification for Codex

This document specifies the **minimal but sufficient metric/logging/analysis design** for the next SABER-RL experiments.

## 0. Experiment Scope

Target tasks:

```text
FetchSlide
FetchPush
FetchPickAndPlace
```

Target methods:

```text
B0
SABER = B3-critic-only
SABER-shuffled
SABER-random
SABER-fast-256
SABER-fast-1024
SABER-runtime-optimized
TD-error critic-only baseline
PER-style baseline
```

Optional reference method if already available:

```text
B3-actor-critic / B3-base
B2 unrestricted gap
B4-hard
```

Core claim to support:

```text
SABER uses local replay support as a critic-side Bellman fitting reliability signal.
It should improve the stability-efficiency tradeoff compared with:
1. no weighting,
2. actor-critic support weighting,
3. shuffled/random support controls,
4. TD-error weighting,
5. PER-style replay prioritization.
```

---

# 1. Logging Philosophy

Separate metrics into three categories.

## 1.1 Online metrics

These should be logged during training into `metrics.csv`.

They are needed to compute learning curves, diagnostics, runtime, and per-phase behavior.

## 1.2 Summary metrics

These should be written into `summary.json` at the end of each run.

They are aggregate values that allow quick experiment comparison without parsing the full curve.

## 1.3 Post-hoc metrics

These should **not** be logged during training.

They should be computed after training from `metrics.csv`, `summary.json`, and optional diagnostic files.

---

# 2. Required Online Metrics: Common Across All Tasks

Add these columns to `metrics.csv` for all methods unless explicitly impossible.

## 2.1 Evaluation / learning curve

```text
step
eval/success_mean
eval/goal_success_mean
eval/env_success_mean
eval/return_mean
```

Notes:

- For Fetch tasks, `eval/goal_success_mean` and `eval/env_success_mean` may be identical or nearly identical.
- Keep both names if the infrastructure already supports them.
- For future non-Fetch tasks, `eval/env_success_mean` should be the official environment success when available.

## 2.2 SABER / support weighting

```text
lite/mean_w_P
lite/max_w_P
lite/std_w_P
lite/entropy_w_P
lite/ESS_w
lite/ESS_w_normalized

lite/support_available_rate
lite/psi_q_rank_rate
lite/q_candidate_fallback_rate
lite/local_scan_truncated_rate

lite/local_support_count_mean
lite/local_support_count_p50
lite/local_support_count_p90

lite/mean_len_A_Q_replay_local
lite/mean_len_A_Q_sup
lite/mean_num_coarse_candidates
lite/mean_num_scanned_candidates
```

Definitions:

```text
ESS_w = (sum_i w_i)^2 / sum_i w_i^2
ESS_w_normalized = ESS_w / batch_size
```

Important:

- For SABER/B3-critic-only, support weights apply to critic loss only.
- For SABER-random and SABER-shuffled, the weight distribution should match SABER as closely as possible.
- `mean_w_P`, `max_w_P`, `entropy_w_P`, and `ESS_w` are essential for proving that shuffled/random controls are distribution-matched.

## 2.3 Loss / critic diagnostics

```text
loss/critic
loss/actor
loss/td_error_abs_mean
loss/td_error_abs_p90
loss/critic_grad_norm
loss/actor_grad_norm
```

If grad norms are expensive or intrusive, implement them as optional config flags:

```text
--log_grad_norms
```

Default may be false if overhead is high, but the code path should exist.

## 2.4 Q-value diagnostics

```text
critic/q_policy_mean
critic/q_policy_std
critic/q_target_mean
critic/q_target_std
critic/q_abs_mean
critic/q_abs_max
```

Purpose:

- Detect critic overestimation, instability, and Q explosion.
- Especially useful for TD-error and B2/B4 comparisons.

## 2.5 Runtime diagnostics

```text
runtime/steps_per_sec
runtime/total_sec

time/local_context_lookup
time/candidate_construction
time/weight_compute
time/actor_critic_update
time/q_candidate_eval
time/q_policy_eval
```

Expected behavior:

- SABER should not need Q candidate evaluation.
- B2 and B4 may use Q candidate evaluation.
- Runtime comparison must include detailed timing breakdown.

---

# 3. Phase-Wise Common Metrics

Every robotic task should expose `task.phase_names`.

For each phase in `task.phase_names`, log:

```text
phase/{phase}/count_rate
phase/{phase}/mean_w_P
phase/{phase}/support_rate
phase/{phase}/fallback_rate
phase/{phase}/local_scan_truncated_rate
phase/{phase}/td_error_abs_mean
```

If existing code uses the `lite/phase_*` naming convention, keep backward compatibility and additionally write the unified names above if possible.

Backward-compatible examples:

```text
lite/phase_{phase}_count_rate
lite/phase_{phase}_mean_w_P
lite/phase_{phase}_psi_q_rank_rate
lite/phase_{phase}_fallback_rate
lite/phase_{phase}_positive_G_sup_rate
```

---

# 4. Method-Specific Online Metrics

## 4.1 SABER-shuffled

Purpose:

```text
Test whether support-weight/sample alignment matters.
```

Required behavior:

```text
1. Compute SABER weights normally.
2. Shuffle weights within the batch before applying to critic loss.
3. Keep the same weight distribution as SABER.
4. Log the same weight/support diagnostics as SABER.
```

No special online metric is required beyond the common metrics.

Post-hoc comparison will compute:

```text
SABER_final - shuffled_final
SABER_AUC - shuffled_AUC
shuffled_drop - SABER_drop
```

## 4.2 SABER-random

Purpose:

```text
Test whether random critic reweighting with similar distribution gives the same effect.
```

Required behavior:

```text
1. Generate random weights with matched mean/variance or matched sampled distribution.
2. Apply to critic loss only.
3. Log weight distribution metrics.
```

No special online metric is required beyond the common metrics.

Post-hoc comparison will compute:

```text
SABER_final - random_final
SABER_AUC - random_AUC
random_drop - SABER_drop
```

## 4.3 TD-error critic-only baseline

Purpose:

```text
Compare support-based reliability weighting against TD-error / surprise-based weighting.
```

Required online metrics:

```text
td_error/abs_mean
td_error/abs_p90
td_error/weight_mean
td_error/weight_max
td_error/weight_entropy
td_error/weight_ESS
```

Expected update:

```text
TD-error weights should apply to critic loss only.
Actor loss should remain unweighted unless explicitly testing an actor-critic TD-error variant.
```

## 4.4 PER-style baseline

Purpose:

```text
Compare SABER against replay sampling prioritization rather than loss reweighting.
```

Required online metrics:

```text
per/priority_entropy
per/priority_ESS
per/IS_weight_mean
per/IS_weight_max
per/replay_sample_age_mean
per/replay_sample_age_p90
per/sample_duplicate_rate
```

Notes:

- PER changes sampling distribution.
- SABER keeps sampling fixed and changes critic loss weights.
- These metrics are needed to detect priority collapse and sample duplication.

## 4.5 SABER-fast variants

Target variants:

```text
SABER-fast-256
SABER-fast-1024
SABER-runtime-optimized
```

Required online metrics:

```text
lite/support_available_rate
lite/q_candidate_fallback_rate
lite/local_scan_truncated_rate
lite/mean_num_coarse_candidates
lite/mean_num_scanned_candidates
runtime/steps_per_sec
time/local_context_lookup
```

Post-hoc metrics:

```text
AUC_retention = AUC_fast / AUC_SABER_1024
final_retention = final_fast / final_SABER_1024
speedup = steps_per_sec_fast / steps_per_sec_SABER_1024
support_retention = support_rate_fast / support_rate_SABER_1024
fallback_delta = fallback_fast - fallback_SABER_1024
truncation_delta = trunc_fast - trunc_SABER_1024
```

---

# 5. Required Summary Metrics

At the end of each run, write these to `summary.json`.

```text
final_success
best_success
best_step

auc_eval_mean
auc_trapz_0_to_total

first_hit_0.25
first_hit_0.50
first_hit_0.75
first_hit_0.90
first_hit_1.00

drop_from_best_to_final
last_quarter_auc

final_window_success_mean
final_window_success_std

runtime_total_sec
steps_per_sec
```

Where:

```text
drop_from_best_to_final = best_success - final_success
last_quarter_auc = AUC over last 25% of training
final_window_success_mean = mean success over last N eval points
final_window_success_std = std success over last N eval points
```

Recommended final window:

```text
last 5 eval points
```

If fewer than 5 eval points exist, use all available eval points and record:

```text
final_window_n
```

---

# 6. Post-Hoc Analysis Metrics

Do not log these during training. Compute them in analysis scripts.

## 6.1 Seed-paired comparison

For each task and each comparator method:

```text
SABER_final - comparator_final
SABER_AUC - comparator_AUC
comparator_drop - SABER_drop
SABER_first_hit - comparator_first_hit
```

Comparators:

```text
B0
SABER-shuffled
SABER-random
TD-error critic-only
PER-style
B3-actor-critic if available
```

Aggregate:

```text
mean paired diff
std paired diff
bootstrap 95% CI
win_count_final
win_count_AUC
win_count_drop
```

For `drop`, lower is better, so define:

```text
drop_advantage = comparator_drop - SABER_drop
```

Positive means SABER is more stable.

## 6.2 Runtime-performance comparison

Compute:

```text
overhead_vs_B0
overhead_vs_B3_base
AUC_retention
final_retention
speedup
```

Recommended figure:

```text
x-axis: runtime overhead vs B0
y-axis: AUC improvement vs B0
point size: final success
```

## 6.3 Weight distribution validation

For SABER, shuffled, and random:

```text
mean_w_P
max_w_P
std_w_P
entropy_w_P
ESS_w
ESS_w_normalized
```

Goal:

```text
SABER, shuffled, and random should have comparable weight distribution.
If SABER outperforms them despite matched distributions, support/sample alignment matters.
```

---

# 7. Optional Diagnostic Files

These are useful for paper-quality mechanism analysis but may be expensive.

Do not log every update. Log at coarse intervals such as every 10k or every eval step.

Recommended files:

```text
diagnostics/support_td_overlap.csv
diagnostics/critic_validation.csv
diagnostics/policy_drift.csv
diagnostics/support_bin_td_error.csv
```

## 7.1 Support vs TD-error overlap

Store periodically:

```text
step
sample_id
support_score
td_error_abs
weight_saber
weight_td_error
```

Post-hoc compute:

```text
support_td_top10_overlap
support_td_top20_overlap
corr_support_td_error
```

Purpose:

```text
Show that support signal is not simply TD-error under another name.
```

## 7.2 Support-bin Bellman diagnostics

Bin samples by support:

```text
low_support
mid_support
high_support
```

Store:

```text
td_error_before_update
td_error_after_update
td_error_reduction
target_y_var_low_support
target_y_var_mid_support
target_y_var_high_support
```

Purpose:

```text
Support-rich samples should behave like more reliable critic fitting targets.
```

## 7.3 Policy drift

Use a fixed validation state set.

Store:

```text
policy_action_drift_t = mean ||pi_t(x_val) - pi_{t-k}(x_val)||
q_drift = mean |Q_t(x_val, pi_t) - Q_{t-k}(x_val, pi_{t-k})|
```

Purpose:

```text
Show that SABER avoids actor instability seen in actor-critic support weighting.
```

---

# 8. Task-Specific Metrics

## 8.1 FetchPush

Phases:

```text
approach
contact_push
near_goal
```

Task metrics:

```text
fetch_push/object_goal_dist_mean
fetch_push/object_goal_dist_p50
fetch_push/object_goal_dist_p90

fetch_push/hand_object_dist_mean
fetch_push/contact_like_rate
fetch_push/object_displacement_mean
fetch_push/progress_rate
```

Important first-hit thresholds:

```text
0.50
0.75
0.90
1.00
```

Interpretation focus:

```text
SABER does not need to beat B2 on FetchPush.
It should beat B0 or remain competitive while being stable and cheaper than B4-hard.
```

## 8.2 FetchPickAndPlace

Phases:

```text
approach
grasp
lift
place
```

Task metrics:

```text
fetch_pick/hand_object_dist_mean
fetch_pick/object_goal_dist_mean
fetch_pick/object_z_mean
fetch_pick/gripper_gap_mean

fetch_pick/lift_success_rate
fetch_pick/grasp_contact_rate
fetch_pick/place_near_goal_rate
```

Important first-hit thresholds:

```text
0.25
0.50
0.75
```

Interpretation focus:

```text
FetchPickAndPlace is the key task for support-aware learning.
Check whether SABER preserves or improves B3 actor-critic performance.
```

## 8.3 FetchSlide

Phases:

```text
approach
strike
slide
near_goal
```

Task metrics:

```text
fetch_slide/object_goal_dist_mean
fetch_slide/object_speed_mean
fetch_slide/object_speed_p90
fetch_slide/object_velocity_alignment_mean
fetch_slide/strike_contact_like_rate
fetch_slide/progress_rate
fetch_slide/overshoot_rate

fetch_slide/phase_distribution_approach
fetch_slide/phase_distribution_strike
fetch_slide/phase_distribution_slide
fetch_slide/phase_distribution_near_goal
```

Important first-hit thresholds:

```text
0.25
0.50
```

Interpretation focus:

```text
FetchSlide is the key task for SABER stability.
SABER should preserve B3 AUC while improving final success and reducing best-to-final drop.
```

---

# 9. Required Analysis Script Outputs

For each task, produce:

```text
by_seed.csv
aggregate.csv
paired_vs_b0.csv
paired_vs_controls.csv
runtime_pareto.csv
report.md
```

## 9.1 by_seed.csv

Rows:

```text
task
method
variant
seed
final_success
best_success
best_step
auc_eval_mean
auc_trapz_0_to_total
first_hit_0.25
first_hit_0.50
first_hit_0.75
first_hit_0.90
first_hit_1.00
drop_from_best_to_final
last_quarter_auc
final_window_success_mean
final_window_success_std
runtime_total_sec
steps_per_sec
```

## 9.2 aggregate.csv

Rows grouped by method:

```text
mean/std for all core metrics
hit_count and mean_hit_step for each first-hit threshold
```

## 9.3 paired_vs_controls.csv

For each seed:

```text
SABER - B0
SABER - shuffled
SABER - random
SABER - TD-error
SABER - PER
```

Metrics:

```text
final_diff
AUC_diff
drop_advantage
first_hit_diff
```

## 9.4 report.md

Should include:

```text
1. Main performance table
2. Stability table
3. Control comparison table
4. Runtime table
5. Phase-wise diagnostics summary
6. Interpretation and flags
```

---

# 10. Success Criteria

SABER is considered strong if the following hold.

## 10.1 Primary criteria

```text
SABER final >= B0 final in at least 2/3 Fetch tasks.
SABER AUC >= B0 AUC in at least 2/3 Fetch tasks.
SABER drop < B3-actor-critic drop on FetchSlide.
SABER > shuffled/random on AUC or final.
SABER >= TD-error/PER on stability-efficiency tradeoff.
```

## 10.2 Runtime criteria

```text
SABER-fast retains at least 90% of SABER-1024 AUC.
SABER-fast retains at least 90% of SABER-1024 final success.
SABER-fast gives meaningful speedup.
```

Recommended target:

```text
AUC_retention >= 0.90
final_retention >= 0.90
speedup >= 1.5x
```

## 10.3 Mechanism criteria

```text
SABER and shuffled/random have matched weight distributions.
SABER outperforms shuffled/random.
Support signal is not identical to TD-error signal.
SABER reduces best-to-final degradation compared with B3 actor-critic.
```

---

# 11. Implementation Priority

Implement in this order.

## Tier 1: must implement now

```text
1. Core online metrics
2. Weight/support metrics
3. Runtime metrics
4. Phase-wise metrics
5. Summary metrics
6. Post-hoc analysis scripts
```

## Tier 2: important for paper

```text
1. TD-error baseline metrics
2. PER-style baseline metrics
3. ESS_w
4. last_quarter_auc
5. final_window_success
6. paired_vs_controls analysis
```

## Tier 3: optional deep diagnostics

```text
1. support vs TD-error overlap
2. support-bin TD-error reduction
3. critic validation TD-error
4. policy drift
5. Q drift
```

---

# 12. Do Not Overcomplicate First Implementation

Do not block training on every optional diagnostic.

The first implementation must ensure:

```text
All methods can run.
metrics.csv has required common columns.
summary.json has final aggregate columns.
analysis scripts produce by_seed, aggregate, and paired comparison tables.
```

Add deep diagnostics behind flags:

```text
--log_deep_diagnostics
--log_grad_norms
--log_policy_drift
--log_support_td_overlap
```

Default:

```text
false
```

This avoids slowing down the main training runs.

```

# exact_support_lookup_optimization

# MetaCog-Lite v0 Exact Support Lookup Optimization

## 요약

MetaCog-Lite v0 B3 계열의 주요 병목은 neural network forward/backward가 아니라, 매 update batch마다 수행되는 replay-action support lookup이다.

현재 B3 support 계산은 각 train item에 대해 다음을 반복한다.

```
coarse_index_key로 coarse candidates 조회
-> full_context_key(s_j, g_train, h_j)로 filtering
-> local action_bin set 구성
-> policy action_bin support 여부 확인
```

이 문서는 B3의 알고리즘 의미를 바꾸지 않고 support lookup 비용을 줄이는 두 가지 exact optimization을 정리한다.

```
1. exact negative short-circuit
2. versioned support cache
```

핵심은 다음이다.

```
support definition unchanged
psi_support unchanged
w_P unchanged
Omega unchanged
```

즉, 방법론 변경이 아니라 같은 support decision을 더 빠르게 계산하는 구현 최적화다.

---

## 배경: B3 Support 계산

B3 support-only weighting은 각 HER-relabeled train item에 대해 support signal을 계산한다.

```
x_i = (s_i, g_train, h_i)
a_policy = pi(x_i)
policy_bin = action_bin(a_policy)
```

B3는 local replay context 안에 policy action과 같은 action bin이 존재하는지 확인한다.

```
psi_support_i = 1 if:
  local replay pool is sufficient
  A_Q_sup candidate count is sufficient
  policy action_bin appears in full local replay action bins
else:
  psi_support_i = 0
```

weight는 다음과 같다.

```
w_i = 1 + lambda_support * psi_support_i
Omega_i = w_i / mean_batch(w)
```

여기서 중요한 구조는 coarse context와 full context의 관계다.

```
full_context candidates are a subset of coarse candidates
```

즉:

```
full_context candidates ⊆ coarse candidates
```

이 관계 때문에 exact negative short-circuit이 가능하다.

---

## 1. Exact Negative Short-Circuit

### 의미

`exact negative short-circuit`은 다음을 의미한다.

```
끝까지 full_context filtering을 해도 support-positive가 절대 될 수 없는 경우를
coarse 단계에서 미리 확인하고 바로 fallback 처리한다.
```

즉:

```
어차피 negative가 확정된 query는 끝까지 계산하지 않는다.
```

여기서 `exact`라는 말이 중요한데, 이는 approximation이 아니라 원래 naive lookup과 같은 결과를 낸다는 뜻이다.

---

### 안전한 short-circuit 조건

다음 조건들은 exact negative로 안전하다.

### 조건 1. coarse candidate 수가 N_local_min보다 작음

```
if len(coarse_candidates) < N_local_min:
    psi_support = 0
```

이유:

```
len(full_context_candidates) <= len(coarse_candidates)
```

따라서 coarse 후보가 이미 부족하면 full_context filtering 후에도 충분해질 수 없다.

---

### 조건 2. coarse candidate 수만으로 M_Q_min을 만족할 수 없음

```
if min(len(coarse_candidates), M_B) < M_Q_min:
    psi_support = 0
```

이유:

```
len(A_Q_sup) <= min(len(full_context_candidates), M_B)
len(full_context_candidates) <= len(coarse_candidates)
```

따라서 coarse 후보 수가 너무 작으면 최종 candidate set도 충분할 수 없다.

---

### 조건 3. policy action_bin이 coarse bucket action bins에 없음

```
if policy_action_bin not in coarse_action_bins:
    psi_A_policy = 0
    psi_support = 0
```

이유:

```
full_context_action_bins ⊆ coarse_action_bins
```

coarse bucket 전체에도 없는 action bin은 full_context subset 안에도 존재할 수 없다.

---

### 하면 안 되는 short-circuit

다음은 안전하지 않다.

```
coarse bucket에 policy action_bin이 있음
=> support-positive라고 판단
```

이는 틀릴 수 있다. coarse bucket에는 해당 action bin이 있어도, full_context filtering 후에는 사라질 수 있다.

즉:

```
coarse positive 가능성 있음
does not imply full_context positive
```

short-circuit은 negative 확정에만 사용해야 한다.

---

## 2. Versioned Support Cache

### 의미

`versioned cache`는 다음을 의미한다.

```
같은 support query가 반복되고,
해당 replay bucket이 바뀌지 않았다면,
이전에 계산한 support 결과를 재사용한다.
```

cache key는 다음과 같이 잡는다.

```
cache_key = (
    full_context_key,
    policy_action_bin,
    replay_bucket_version,
    local_search_max,
)
```

같은 key가 다시 들어오면 full_context filtering을 다시 하지 않고 저장된 결과를 사용한다.

---

### 왜 version이 필요한가?

replay buffer는 계속 변한다. 새로운 transition이 들어오거나 circular buffer overwrite가 일어나면 support 결과가 달라질 수 있다.

따라서 coarse bucket마다 version counter를 둔다.

```
bucket_version[coarse_key] += 1
```

새 transition이 해당 coarse bucket에 들어가거나 기존 transition이 제거되면 version을 증가시킨다.

cache에 저장된 version과 현재 bucket version이 다르면 cache miss로 처리하고 다시 계산한다.

---

### Versioned cache와 stale cache의 차이

```
versioned cache:
  replay bucket이 바뀌지 않았을 때만 재사용
  exact optimization
  의미 변화 없음

stale cache:
  replay가 바뀌었는데도 예전 support 결과를 재사용
  approximation
  의미 변화 있음
```

따라서 versioned cache는 제대로 구현하면 MetaCog-Lite v0의 support semantics를 바꾸지 않는다.

---

### action_bin 기반 cache가 가능한 이유

B3는 policy의 raw continuous action을 그대로 support test에 쓰지 않는다.

```
policy_action_bin = action_bin(pi(x))
```

따라서 actor output이 조금 변해도 action bin이 같으면 support query는 동일하다.

```
raw action: 0.21 -> 0.23 -> 0.24
action_bin: move-right-small
```

이 경우 cache reuse가 가능하다.

---

## 두 방법을 함께 쓰는 순서

권장 lookup 순서는 다음과 같다.

```
1. query coarse_key 계산
2. policy_action_bin 계산
3. coarse bucket metadata 조회
4. exact negative short-circuit 적용
5. cache key 생성
6. versioned cache 조회
7. cache miss이면 full_context filtering 수행
8. support result 계산
9. cache에 저장
```

pseudocode:

```python
coarse_key = spec.coarse_index_key(obs, horizon)
policy_bin = spec.action_bin(a_policy)

coarse_indices = replay_index.get(coarse_key)
coarse_bins = replay_index.action_bins_for(coarse_key)
bucket_version = replay_index.version(coarse_key)

# exact negative short-circuit
if len(coarse_indices) < cfg.N_local_min:
    return support_negative()

if min(len(coarse_indices), cfg.M_B) < cfg.M_Q_min:
    return support_negative()

if policy_bin not in coarse_bins:
    return support_negative()

query_full_key = spec.full_context_key(obs, g_train, horizon)
cache_key = (
    query_full_key,
    policy_bin,
    bucket_version,
    cfg.local_search_max,
)

if cache_key in support_cache:
    return support_cache[cache_key]

local_indices = []
local_bins = set()

for idx in scan(coarse_indices, cfg.local_search_max):
    transition = replay[idx]
    if spec.full_context_key(transition.obs, g_train, transition.horizon) == query_full_key:
        local_indices.append(idx)
        local_bins.add(replay_index.action_bin(idx))

result = compute_support_result(local_indices, local_bins, policy_bin, cfg)
support_cache[cache_key] = result
return result
```

---

## 중요한 구현 주의점

### 1. local_search_max가 random scan이면 cache가 exact가 아닐 수 있음

현재 B3가 `local_search_max`로 coarse candidates 중 일부만 scan한다면, scan subset이 매 query마다 random하게 달라질 수 있다.

이 경우 같은 cache key라도 이전 query와 현재 query가 서로 다른 scanned subset을 볼 수 있다.

따라서 versioned cache를 exact하게 쓰려면 다음이 필요하다.

```
same key/version/local_search_max
=> same scanned candidate subset
```

즉 deterministic scan이 필요하다.

예:

```
scan_seed = hash(full_context_key, policy_action_bin, bucket_version, local_search_max)
```

또는 index order 기반 deterministic truncation을 사용한다.

만약 매번 새 random subset을 뽑는다면 cache는 exact optimization이 아니라 stochastic semantics를 바꾸는 approximation이 된다.

---

### 2. diagnostics 처리

action-bin short-circuit은 training weight 관점에서는 exact하다.

예:

```
policy_action_bin not in coarse_action_bins
=> psi_support = 0
=> w = 1
```

하지만 이 경우 `len_A_Q_replay_local` 같은 diagnostic을 정확히 기록하려면 full_context filtering이 필요하다.

따라서 구현 시 정책을 정해야 한다.

권장:

```
training path:
  short-circuit으로 빠르게 support_negative 반환

logging path:
  short_circuited = 1 기록
  local_count_exact는 null 또는 log_interval에서만 lazy 계산
```

논문/분석에서 exact local_count가 필요하면 log interval마다 별도 full diagnostic을 수행한다.

---

## 빠른 synthetic 검증 결과

현재 학습 중인 실험을 건드리지 않고 synthetic frozen replay setup에서 빠른 검증을 수행했다.

설정:

```
replay_size = 30,000
queries = 5,000
local_search_max = 512
N_local_min = 8
M_Q_min = 4
M_B = 16
```

비교:

```
naive B3 lookup
vs
exact negative short-circuit + versioned cache lookup
```

결과:

```
semantic_mismatches(psi/fallback/psi_A_policy) = 0
diagnostic_mismatches_checked = 0 / 3719

naive_time_sec = 0.0633
optimized_time_sec = 0.0333
speedup = 1.90x

naive_full_key_calcs = 1,043,084
optimized_full_key_calcs = 418,423
full_key_calc_reduction = 59.9%

short_circuit_bin = 1,281 / 5,000
short_circuit_count = 0
short_circuit_mq = 0

cache_hits = 1,713
cache_misses = 2,006
cache_hit_rate = 0.461
```

해석:

```
support decision은 naive lookup과 동일했다.
full_context_key 계산량은 약 60% 감소했다.
synthetic setting에서 lookup time은 약 1.9배 개선됐다.
```

주의:

```
이 결과는 synthetic frozen replay correctness / feasibility check이다.
실제 FetchPickAndPlace / FetchPush replay에서의 speedup은 별도 microbenchmark가 필요하다.
```

---

## 회의용 핵심 메시지

```
Exact negative short-circuit과 versioned cache는 B3 알고리즘을 바꾸는 것이 아니라,
동일한 replay-action support decision을 더 빠르게 계산하는 exact support lookup optimization입니다.
```

```
Synthetic frozen replay 검증에서는 support decision mismatch 없이
full_context_key 계산량을 약 60% 줄였고,
lookup time을 약 1.9배 개선했습니다.
```

```
따라서 runtime을 줄이기 위해 periodic/stale approximation으로 바로 가기보다,
먼저 exact support lookup optimization을 구현하고,
그 위에서 B3-Fast-512 또는 B3-Fast-512 critic-only를 평가하는 것이 가장 방어적입니다.
```

---

## 다음 검증 계획

### 1. Frozen-buffer correctness test

실제 FetchPickAndPlace / FetchPush replay snapshot에서:

```
naive lookup
optimized lookup
```

을 비교한다.

확인:

```
psi_support mismatch rate = 0
psi_A_policy mismatch rate = 0
fallback mismatch rate = 0
w_P mismatch rate = 0
```

### 2. Frozen-buffer speed microbenchmark

측정:

```
query/sec
mean lookup latency
p95 lookup latency
full_context_key calculation count
short-circuit rate
cache hit rate
cache miss rate
support lookup time share
```

### 3. Learning run

최적화된 exact lookup engine을 사용해:

```
B3-Fast-512
B3-Fast-512 critic-only
```

를 FetchPickAndPlace와 FetchPush에서 비교한다.

보고:

```
Eval AUC
Final success
Best success
wall-clock
steps/sec
overhead vs B0
AUC/hour
```

---

## 결론

현재 가장 방어적인 runtime 절감 전략은 다음이다.

```
1. exact negative short-circuit
2. versioned support cache
3. deterministic bounded scan
4. B3-Fast-512
5. critic-only는 해석/안정성 ablation으로 확인
```

`periodic support update`나 `stale reuse`는 runtime 절감 효과가 클 수 있지만, support를 매 update마다 계산한다는 원래 의미를 일부 바꾼다. 따라서 exact optimization 이후에도 overhead가 너무 클 때 별도 fast approximation으로 고려하는 것이 좋다.
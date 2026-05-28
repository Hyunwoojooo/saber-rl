# FetchPush Runtime Optimization Snapshot 차이 정리

이 문서는 FetchPush 환경에서 B3 critic-only runtime optimization을 비교하기 위한 코드 스냅샷들의 차이를 정리한다.

세 폴더는 모두 같은 목적을 가진다.

- FetchPush
- B3 critic-only
- sparse reward
- local support lookup runtime 최적화

차이는 주로 **local replay support lookup을 어떻게 계산하느냐**에 있다.

---

## 1. `B3-Critic-only-runtime-optimization-sticky-v2`

이 폴더는 JIT를 적용하기 전의 runtime optimization 버전이다.

핵심 구현:

- `ReplayActionIndex v2`를 사용한다.
- coarse bucket별 live slot metadata를 유지한다.
- 같은 replay slot이 같은 coarse bucket에 반복 append되는 문제를 줄인다.
- local support lookup 시 coarse bucket 전체 candidate를 매번 만들지 않는다.
- replay index의 live slot list에서 최대 `local_search_max`개 transition을 직접 선택한다.
- bounded scan은 deterministic pseudo-random 방식이다.
  - 같은 `(coarse_key, bucket_version, local_search_max, scan_seed)`에 대해서는 항상 같은 subset을 선택한다.
- local scan cache를 사용한다.
- versioned support context cache를 사용한다.
- HER goal-aware cache key를 사용한다.
- coarse action-bin metadata를 incremental하게 유지한다.
- exact negative short-circuit은 유지한다.
- batch query coalescing은 끈다.
- Numba/JIT full-context scanner는 사용하지 않는다.

예상되는 diagnostic pattern:

```text
runtime/replay_index_direct_scan_count > 0
runtime/replay_index_candidates_materialized_count ~= 0
runtime/numba_support_scan_enabled = 0
runtime/numba_support_queries = 0
```

목적:

JIT 없이 `ReplayActionIndex v2`, deterministic bounded scan, local scan cache, support context cache만으로 runtime이 얼마나 줄어드는지 확인하기 위한 버전이다.

---

## 2. `B3-Critic-only-runtime-optimization-jit-sticky-v2`

이 폴더는 현재 가장 강한 FetchPush runtime optimization 버전이다.

핵심 구현:

- `B3-Critic-only-runtime-optimization-sticky-v2`의 모든 요소를 포함한다.
- 추가로 FetchPush 전용 Numba/JIT exact support scanner를 사용한다.
- JIT scanner는 Python/NumPy 기반 full-context filtering path를 대체한다.
- 기존 방식에서는 다음 값들을 매번 materialize했다.
  - `ref_obs`
  - HER goal 반복 배열
  - `ref_hash`
  - `local_idx`
  - `local_action_bins`
- JIT 방식에서는 bounded scan으로 선택된 replay slot을 직접 loop 돌면서 아래 값만 계산한다.
  - local pool size
  - available support size
  - 현재 policy action bin이 local support set 안에 있는지 여부
  - support-positive 여부
  - fallback 여부
- JIT 결과는 현재 policy action bin에 의존하므로, support cache key에 policy action bin과 scanner identity를 포함한다.
- `ReplayActionIndex v2` direct bounded scan을 그대로 사용한다.
- coarse bucket 전체 candidate materialization을 피한다.
- deterministic pseudo-random bounded scan을 사용한다.
- local scan cache와 versioned support context cache를 사용한다.
- coarse action-bin metadata를 incremental하게 유지한다.
- exact negative short-circuit은 유지한다.
- batch query coalescing은 끈다.

예상되는 diagnostic pattern:

```text
runtime/replay_index_direct_scan_count > 0
runtime/replay_index_candidates_materialized_count ~= 0
runtime/numba_support_scan_enabled = 1
runtime/numba_support_queries > 0
runtime/numba_support_scan_fallback_rate = 0
time/numba_support_scan > 0
```

목적:

다음 요소들이 결합됐을 때 runtime이 얼마나 줄어드는지 확인하기 위한 최종 후보 버전이다.

- `ReplayActionIndex v2`
- direct bounded replay scan
- deterministic pseudo-random subset selection
- cache reuse
- exact JIT full-context support scanning

세 스냅샷 중 가장 강한 runtime optimization 구현이다.

---

## 3. `B3-Critic-only-runtime-optimization-jit-fetchslide-style`

이 폴더는 FetchSlide runtime optimization 구조와 최대한 비슷하게 만든 비교용 버전이다.

핵심 구현:

- FetchPush 전용 Numba/JIT exact support scanner는 사용한다.
- support context cache를 사용한다.
- local scan cache를 사용한다.
- deterministic bounded scan을 사용한다.
- batch query coalescing을 기본적으로 켠다.
- 하지만 `ReplayActionIndex v2`의 direct bounded scan path를 핵심 경로로 사용하지 않는다.
- 대신 FetchSlide 방식처럼 먼저 coarse bucket 전체 candidate를 materialize한다.

```python
candidates = replay.action_index.candidates(coarse_key, replay.size)
```

- 그 다음 materialized candidate array에서 최대 `local_search_max`개를 선택한다.
- subset 선택 방식은 deterministic evenly-spaced 방식이다.
- local scan cache key도 FetchSlide 구현에 가깝게 단순하다.
  - `(coarse_key, local_search_max, bucket_version)`
- exact negative short-circuit은 강제로 켜지 않는다.

예상되는 diagnostic pattern:

```text
runtime/replay_index_candidates_materialized_count > 0
runtime/replay_index_direct_scan_count ~= 0
runtime/numba_support_scan_enabled = 1
runtime/numba_support_queries > 0
runtime/numba_support_scan_fallback_rate = 0
```

목적:

- JIT scanner의 효과와 `ReplayActionIndex v2` direct bounded scan의 효과를 분리해서 보기 위한 버전이다.
- 이 버전이 `B3-Critic-only-runtime-optimization-jit-sticky-v2`보다 훨씬 느리다면, runtime 개선은 JIT만으로 설명되지 않는다.
- 그 경우 coarse bucket 전체 candidate materialization을 피하는 `ReplayActionIndex v2` direct bounded scan이 중요한 역할을 한다고 해석할 수 있다.
- 반대로 이 버전과 `B3-Critic-only-runtime-optimization-jit-sticky-v2`의 속도가 비슷하다면, runtime 개선의 대부분은 JIT full-context scanner에서 나온다고 해석할 수 있다.

---

## Summary Table

| Snapshot | coarse bucket 전체 materialization | direct bounded scan | bounded subset 선택 방식 | JIT full-context scanner | batch query coalescing | exact negative short-circuit |
|---|---:|---:|---|---:|---:|---:|
| `B3-Critic-only-runtime-optimization-sticky-v2` | No | Yes | bucket version 기반 deterministic pseudo-random | No | No | Yes |
| `B3-Critic-only-runtime-optimization-jit-sticky-v2` | No | Yes | bucket version 기반 deterministic pseudo-random | Yes | No | Yes |
| `B3-Critic-only-runtime-optimization-jit-fetchslide-style` | Yes | No | deterministic evenly-spaced subset | Yes | Yes | Not forced |

---

## 비교 해석 가이드

### 비교 1

```text
B3-Critic-only-runtime-optimization-sticky-v2
vs
B3-Critic-only-runtime-optimization-jit-sticky-v2
```

이 비교는 FetchPush JIT full-context support scanner의 효과를 분리해서 보기 위한 비교다.

해석:

- 두 버전 모두 `ReplayActionIndex v2` direct bounded scan을 사용한다.
- 차이는 JIT scanner 사용 여부다.
- 따라서 `B3-Critic-only-runtime-optimization-jit-sticky-v2`가 더 빠르면, 그 차이는 주로 JIT full-context scanner에서 나온다.

### 비교 2

```text
B3-Critic-only-runtime-optimization-jit-fetchslide-style
vs
B3-Critic-only-runtime-optimization-jit-sticky-v2
```

이 비교는 coarse bucket 전체 candidate materialization을 피하는 것이 얼마나 중요한지 보기 위한 비교다.

해석:

- 두 버전 모두 JIT scanner를 사용한다.
- 차이는 local scan candidate를 고르는 방식이다.
- `B3-Critic-only-runtime-optimization-jit-fetchslide-style`은 coarse bucket 전체 candidate를 먼저 만든다.
- `B3-Critic-only-runtime-optimization-jit-sticky-v2`는 coarse bucket 전체를 만들지 않고, live slot metadata에서 바로 bounded subset을 선택한다.
- 따라서 `B3-Critic-only-runtime-optimization-jit-sticky-v2`가 더 빠르면, `ReplayActionIndex v2` direct bounded scan이 runtime 개선에 중요하다고 볼 수 있다.

### 비교 3

```text
B3-Critic-only-runtime-optimization-sticky-v2
vs
B3-Critic-only-runtime-optimization-jit-fetchslide-style
```

이 비교는 JIT 없이 direct bounded scan을 쓰는 방식과, JIT는 있지만 FetchSlide-style candidate materialization을 쓰는 방식을 비교한다.

해석:

- `B3-Critic-only-runtime-optimization-sticky-v2`는 JIT는 없지만 coarse bucket 전체 materialization을 피한다.
- `B3-Critic-only-runtime-optimization-jit-fetchslide-style`은 JIT는 있지만 coarse bucket 전체 candidate를 먼저 만든다.
- 이 비교를 통해 “JIT scanner”와 “candidate materialization 제거” 중 어느 쪽이 더 큰 runtime 이득을 주는지 간접적으로 볼 수 있다.

---

## 중요 Runtime Diagnostics

속도 비교에서 가장 중요한 지표는 다음과 같다.

- `runtime/steps_per_sec`
- `time/local_context_lookup`
- `time/numba_support_scan`
- `runtime/full_context_scan_rate`
- `runtime/support_cache_hit_rate`
- `runtime/local_scan_cache_hit_rate`
- `runtime/replay_index_direct_scan_count`
- `runtime/replay_index_candidates_materialized_count`
- `lite/mean_num_coarse_candidates`
- `lite/mean_num_scanned_candidates`
- `lite/local_scan_truncated_rate`

특히 핵심은 다음 두 지표다.

```text
runtime/replay_index_direct_scan_count
runtime/replay_index_candidates_materialized_count
```

해석:

- `direct_scan_count > 0`이고 `candidates_materialized_count ~= 0`이면 `ReplayActionIndex v2` direct bounded scan이 작동하고 있는 것이다.
- `candidates_materialized_count > 0`이면 coarse bucket 전체 candidate를 먼저 만들고 있다는 뜻이다.
- replay buffer가 커질수록 후자의 비용이 커질 가능성이 높다.

---

## 중요 Behavior / Correctness Diagnostics

성능과 support behavior가 무너지지 않았는지 확인할 때 중요한 지표는 다음과 같다.

- `lite/support_available_rate`
- `lite/psi_q_rank_rate`
- `lite/psi_A_policy_rate`
- `lite/q_candidate_fallback_rate`
- `lite/mean_len_A_Q_replay_local`
- `lite/mean_len_A_Q_sup`
- `eval/success_mean`
- `eval/return_mean`

runtime이 빨라져도 아래 현상이 나타나면 문제가 있을 수 있다.

- `psi_A_policy_rate`가 거의 0에 가까워짐
- `psi_q_rank_rate`가 거의 0에 가까워짐
- `q_candidate_fallback_rate`가 1에 가까워짐
- `mean_len_A_Q_replay_local`이 비정상적으로 낮아짐
- eval success가 B0보다 명확히 나빠짐

따라서 runtime optimization은 단순히 `steps/sec`만 보고 판단하지 않고, support diagnostic과 eval 성능을 함께 봐야 한다.

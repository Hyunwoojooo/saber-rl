# SABER AAAI-27 Random-Control Experiment Plan

작성일: 2026-06-24
목적: SABER의 성능 향상이 structured local support 때문인지, 아니면 단순 random reweighting regularization 때문인지 구분하기 위한 FPP random-control 실험 계획을 고정한다.

## 1. 결론

FPP main task에서 random-control은 새 실험이 필요하다.

현재 repository 상태:

| Control | Existing FPP artifact | 판단 |
| --- | --- | --- |
| Random-Support | `experiment_FPP/runs/B3-critic-only-random` 1k smoke 수준 | 1M main/control evidence로 불가 |
| Shuffled-Weight | `experiment_FPP/runs/B3-critic-only-shuffled` 1k smoke 수준 | 1M main/control evidence로 불가 |
| Random-Fixed-Count | FPP artifact 없음 | 새 구현 또는 port 필요 |

DrawerOpen과 FetchPush에는 random/shuffled/fixed-count 계열이 있지만, FPP main claim을 방어하려면 FPP에서 직접 control을 보여야 한다.

## 2. 왜 필요한가

현재 reviewer 공격은 다음이다.

> SABER가 support-aware라서 좋은 것이 아니라, critic loss에 random하게 큰 weight를 주는 regularization이 좋아진 것 아닌가?

이 공격은 DrawerOpen에서 Random-Support가 강하게 나온 결과 때문에 더 중요하다. 따라서 main FPP에서 다음을 보여야 한다.

- SABER > Random-Support
- SABER > Shuffled-Weight
- 가능하면 SABER > Random-Fixed-Count
- weight distribution은 최대한 비슷하게 맞춘다.

## 3. 실험 우선순위

### Tier 1: 반드시 먼저

FPP DDPG, 1M steps, seeds 1--5:

| Row | Method | 목적 |
| --- | --- | --- |
| Existing SABER | `b3_support_runtime_opt`, local_search 512 | reference |
| Random-Support | `b3_support_random`, local_search 512 | positive-count random control |
| Shuffled-Weight | `b3_support_shuffled`, local_search 512 | weight-distribution shuffled control |

이 세 줄이 최소 방어선이다.

### Tier 2: 가능하면 추가

FPP DDPG Random-Fixed-Count:

- 현재 FPP trainer에는 `b3_fully_random_fixed_count`가 없다.
- FetchPush/DrawerOpen에는 fixed-count 구현이 있다.
- FPP에 port한다면 `random_fixed_count_k`를 추가해야 한다.
- 추천 \(k\): 기존 FPP DDPG SABER의 평균 support-positive count를 사용한다.

현재 FPP DDPG SABER mean support rate는 약 `0.367`이고 batch size는 256이므로:

\[
k \approx \mathrm{round}(256 \times 0.367)=94
\]

따라서 first fixed-count control은 `random_fixed_count_k=94`가 합리적이다.

### Tier 3: 여력이 있으면

FPP TD3 또는 SAC 중 하나에 Random-Support/Shuffled-Weight 추가.

우선순위는 TD3다. 이유는 TD3가 deterministic actor-critic이고 DDPG와 구조적으로 가까워서 DDPG 결과를 보조하기 좋다.

## 4. 중요한 설정 결정

### local_search_max

기존 `B3-critic-only-random/config.json`과 `B3-critic-only-shuffled/config.json`은 `local_search_max=1024`로 되어 있다. 하지만 main FPP DDPG SABER row는 `b3_critic_only_runtime_opt_v4_ls512`를 source of truth로 사용한다.

따라서 FPP main control은 `local_search_max=512`로 맞춰야 한다. 그렇지 않으면 SABER와 control의 lookup budget이 달라진다.

### trainer path

초기 상태에서는 `experiment_FPP/B3-critic-only-random/run.py --dry-run`이 다음 legacy trainer를 가리켰다.

```text
/Users/joo/version_v0_lite/src/train_fetch_pick_and_place_lite_v0.py
```

현재 checkout에는 이 경로가 없어서 `experiment_FPP/shared/run_experiment.py`에 fallback을 추가했다. 이제 wrapper dry-run은 현재 repo 안의 trainer를 가리킨다.

```text
experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/train_fetch_pick_and_place_lite_v0.py
```

이 trainer는 `b3_support_random`과 `b3_support_shuffled`를 지원한다.

주의: wrapper registry의 random/shuffled 기본값은 여전히 `local_search_max=1024`다. Main FPP DDPG SABER row와 공정하게 비교하려면 아래 direct command처럼 `local_search_max=512`로 실행하거나, 별도 ls512 wrapper/config를 만들어야 한다.

## 5. 실행 명령 후보

아래 명령은 FPP DDPG random/shuffled control을 1M, seeds 1--5, local_search 512로 돌리는 기준이다.

### Random-Support

```bash
TRAINER="experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/train_fetch_pick_and_place_lite_v0.py"
OUT="experiment_FPP/runs/fpp_ddpg_random_support_ls512_1m"

for seed in 1 2 3 4 5; do
  python3 "$TRAINER" \
    --task_name fetch_pick_and_place \
    --run_name "fetch_pick_and_place_b3_support_random_ls512_s${seed}_1000000_ls512" \
    --output_root "$OUT" \
    --method b3_support_random \
    --reward_type sparse \
    --total_steps 1000000 \
    --batch_size 256 \
    --her_ratio 0.8 \
    --updates_per_step 1 \
    --eval_freq 10000 \
    --eval_episodes 20 \
    --M_B 16 \
    --M_Q_min 4 \
    --N_local_min 4 \
    --N_gap_min 128 \
    --local_search_max 512 \
    --lambda_support 1.0 \
    --weighting_mode batch_norm \
    --weight_scope critic_only \
    --policy_support_mode bin \
    --device auto \
    --env_seed "$seed" \
    --torch_seed "$seed" \
    --numpy_seed "$seed" \
    --action_noise_seed "$seed" \
    --weight_seed "$((20000 + seed))" \
    --eval_seed "$((10000 + seed))" \
    --no_save_step_checkpoints
done
```

### Shuffled-Weight

```bash
TRAINER="experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/train_fetch_pick_and_place_lite_v0.py"
OUT="experiment_FPP/runs/fpp_ddpg_shuffled_weight_ls512_1m"

for seed in 1 2 3 4 5; do
  python3 "$TRAINER" \
    --task_name fetch_pick_and_place \
    --run_name "fetch_pick_and_place_b3_support_shuffled_ls512_s${seed}_1000000_ls512" \
    --output_root "$OUT" \
    --method b3_support_shuffled \
    --reward_type sparse \
    --total_steps 1000000 \
    --batch_size 256 \
    --her_ratio 0.8 \
    --updates_per_step 1 \
    --eval_freq 10000 \
    --eval_episodes 20 \
    --M_B 16 \
    --M_Q_min 4 \
    --N_local_min 4 \
    --N_gap_min 128 \
    --local_search_max 512 \
    --lambda_support 1.0 \
    --weighting_mode batch_norm \
    --weight_scope critic_only \
    --policy_support_mode bin \
    --device auto \
    --env_seed "$seed" \
    --torch_seed "$seed" \
    --numpy_seed "$seed" \
    --action_noise_seed "$seed" \
    --weight_seed "$((20000 + seed))" \
    --eval_seed "$((10000 + seed))" \
    --no_save_step_checkpoints
done
```

## 6. 예상 runtime

FPP DDPG SABER local_search 512 기준 steps/sec는 약 `41.72`다. Random/Shuffled도 support lookup을 수행하므로 비슷하거나 조금 빠른 수준일 가능성이 높다.

1M step run 1개 예상:

\[
1,000,000 / 41.72 / 3600 \approx 6.66\text{ hours}
\]

2 controls x 5 seeds이면 대략 66 GPU-hours 수준이다. 병렬 실행 여부는 GPU 수와 메모리에 따라 결정해야 한다.

## 7. 결과 집계 기준

각 control run 완료 후 다음 metric을 기존 FPP table과 동일하게 집계한다.

- Final success
- Best success
- Success AUC
- Median Hit90 Step with hit count
- Tail10 success
- Tail50 success
- Steps/sec
- `lite/mean_w_P`
- `lite/ESS_w_normalized`
- `lite/reference_high_weight_rate`
- `lite/assigned_high_weight_rate`
- `lite/support_weight_alignment_rate`

Random/Shuffled control에서 특히 중요한 것은 weight distribution matching이다.

Expected:

- Random-Support: assigned high-weight rate는 reference와 유사하지만 alignment는 낮아야 한다.
- Shuffled-Weight: weight distribution은 reference와 같지만 sample alignment는 깨져야 한다.

## 8. 논문 해석 규칙

### SABER가 control보다 좋을 때

주장 가능:

> Structured support alignment matters in the main FPP setting.

### SABER와 Random-Support가 비슷할 때

주장 축소:

> Support-conditioned reweighting and reweighting-induced regularization both contribute; structured support is not uniquely isolated by current evidence.

### Shuffled-Weight가 실패할 때

주장 가능:

> Weight distribution alone is insufficient; alignment between high weights and support-positive samples matters.

## 9. 현재 action item

1. FPP DDPG Random-Support 1M seeds 1--5 실행.
2. FPP DDPG Shuffled-Weight 1M seeds 1--5 실행.
3. 완료 후 existing `analyze_results.py` 또는 별도 aggregate script로 table 생성.
4. Random-Fixed-Count는 FPP trainer에 method를 port할지 결정.
5. 논문 본문에는 random/shuffled 결과가 생기기 전까지 DrawerOpen random-control을 limitation으로만 유지.

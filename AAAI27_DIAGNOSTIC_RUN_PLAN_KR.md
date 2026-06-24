# SABER AAAI-27 Sample-level Diagnostic Run Plan

작성일: 2026-06-24

목적: SABER가 단순 TD-error/PER-style weighting이 아니라는 점과, support-positive sample이 critic Bellman fitting에서 어떤 차이를 보이는지 sample-level로 확인한다.

## 1. 결론

#5 diagnostic은 기존 artifact만으로 닫을 수 없다. 기존 `metrics.csv`에는 checkpoint-level 평균만 있고, reviewer가 요구한 `psi_i`와 `|delta_i|`의 sample-level 관계가 없다.

따라서 FPP DDPG SABER를 diagnostic logging 옵션으로 다시 실행해야 한다. 이번 구현으로 trainer는 학습 알고리즘을 바꾸지 않고, 일정 간격으로 `diagnostic_samples.csv`만 추가 저장한다.

## 2. 추가된 로깅 옵션

대상 trainer:

```text
experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/train_fetch_pick_and_place_lite_v0.py
```

새 CLI 옵션:

| Option | 의미 | 추천값 |
| --- | --- | --- |
| `--diagnostic_log_samples` | sample-level diagnostic CSV 저장 | 켜기 |
| `--diagnostic_freq` | 몇 env step마다 저장할지 | `10000` |
| `--diagnostic_samples_per_log` | 저장 시 batch에서 몇 sample을 남길지 | `128` |

출력 파일:

```text
<run_dir>/diagnostic_samples.csv
<run_dir>/diagnostic_config.json
```

## 3. 저장되는 핵심 field

| Field | 의미 |
| --- | --- |
| `psi_q_rank` | SABER support-positive indicator |
| `td_abs_before` | critic update 전 fixed-target TD residual |
| `td_abs_after` | critic update 후 같은 Bellman target에 대한 residual |
| `td_abs_reduction` | `td_abs_before - td_abs_after` |
| `w_raw`, `omega_critic` | SABER raw/normalized critic weight |
| `local_pool_size` | local replay context size |
| `support_candidate_count` | support candidate count |
| `policy_action_bin` | current policy action bin |
| `replay_action_bin` | replay action bin |
| `her_relabel` | sample별 HER relabel 여부 |

주의: `td_abs_after`는 critic update 직후, actor update와 target soft update 전에 같은 target으로 계산한다. 따라서 critic fitting 변화만 보는 진단에 더 가깝다.

## 4. Smoke run

먼저 seed 1로 10k 또는 50k smoke run을 돌려 `diagnostic_samples.csv`가 제대로 생기는지 확인한다.

```bash
TRAINER="experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/train_fetch_pick_and_place_lite_v0.py"
OUT="experiment_FPP/runs/fpp_ddpg_saber_diagnostic_ls512_smoke"

python3 "$TRAINER" \
  --task_name fetch_pick_and_place \
  --run_name "fetch_pick_and_place_b3_saber_diagnostic_ls512_s1_50000" \
  --output_root "$OUT" \
  --method b3_support_runtime_opt \
  --reward_type sparse \
  --total_steps 50000 \
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
  --env_seed 1 \
  --torch_seed 1 \
  --numpy_seed 1 \
  --action_noise_seed 1 \
  --weight_seed 20001 \
  --eval_seed 10001 \
  --diagnostic_log_samples \
  --diagnostic_freq 10000 \
  --diagnostic_samples_per_log 128 \
  --no_save_step_checkpoints
```

Smoke run 이후 확인:

```bash
python3 analysis/support_td_error/analyze_sample_diagnostics.py \
  --input experiment_FPP/runs/fpp_ddpg_saber_diagnostic_ls512_smoke
```

## 5. Main diagnostic run

논문용 최소 설정:

- Task: FetchPickAndPlace
- Backbone: DDPG
- Method: SABER critic-only, `b3_support_runtime_opt`
- Steps: 1M
- Seeds: 1--3 최소, 1--5 권장
- `local_search_max=512`
- `diagnostic_freq=10000`
- `diagnostic_samples_per_log=128`

권장 full run command:

```bash
TRAINER="experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/train_fetch_pick_and_place_lite_v0.py"
OUT="experiment_FPP/runs/fpp_ddpg_saber_diagnostic_ls512_1m"

for seed in 1 2 3 4 5; do
  python3 "$TRAINER" \
    --task_name fetch_pick_and_place \
    --run_name "fetch_pick_and_place_b3_saber_diagnostic_ls512_s${seed}_1000000" \
    --output_root "$OUT" \
    --method b3_support_runtime_opt \
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
    --diagnostic_log_samples \
    --diagnostic_freq 10000 \
    --diagnostic_samples_per_log 128 \
    --no_save_step_checkpoints
done
```

## 6. 분석 명령

Full run 완료 후:

```bash
python3 analysis/support_td_error/analyze_sample_diagnostics.py \
  --input experiment_FPP/runs/fpp_ddpg_saber_diagnostic_ls512_1m
```

출력:

```text
analysis/support_td_error/sample_diagnostic_report.md
analysis/support_td_error/sample_diagnostic_summary.json
analysis/support_td_error/sample_diagnostic_summary.csv
analysis/support_td_error/sample_diagnostic_per_step.csv
analysis/support_td_error/sample_diagnostic_group_stats.csv
```

## 7. 논문에서 볼 핵심 숫자

| Metric | 해석 |
| --- | --- |
| `corr_psi_td_before_pearson` | support가 TD-error와 같은 신호인지 |
| `top_td_same_k_support_rate` | PER-style top TD-error sample과 support-positive sample overlap |
| `td_reduction_support_mean` vs `td_reduction_unsupported_mean` | support-positive sample의 critic fitting residual 감소 차이 |
| `target_var_support` vs `target_var_unsupported` | support group별 Bellman target variance |

원하는 결과는 다음이다.

- `corr(psi, |delta|)`가 너무 높지 않다.
- top-k overlap이 1에 가깝지 않다.
- support-positive sample에서 residual reduction이 더 크거나 안정적이다.
- target variance 차이는 reliability 해석의 보조 증거로만 사용한다.

반대로 Random-control처럼 diagnostic 결과가 애매하게 나오면, SABER claim은 “structured support가 유일한 원인”이 아니라 “support-conditioned reweighting과 regularization이 함께 작동할 수 있다”로 낮춰야 한다.

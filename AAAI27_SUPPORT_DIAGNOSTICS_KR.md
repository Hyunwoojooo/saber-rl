# SABER AAAI-27 Support vs TD-error Diagnostic

작성일: 2026-06-24
최신 업데이트: 2026-07-10
목적: SABER가 단순 TD-error weighting 또는 PER-style weighting의 변형이라는 반론을 줄이기 위해, 현재 artifact로 가능한 support-vs-TD-error 분석 범위와 추가로 필요한 diagnostic을 정리한다.

## 1. 결론

기존 FPP artifact만으로는 reviewer가 요구할 만한 핵심 diagnostic을 닫을 수 없었지만, 이후 FPP DDPG/TD3/SAC SABER instrumented diagnostic 1M runs가 모두 완료되었다. 본 문서의 상세 표는 DDPG를 중심으로 유지하고, cross-backbone aggregate는 논문 diagnostic tables를 source of truth로 둔다.

DDPG diagnostic source:

- Source checkout: `/private/tmp/metacog-v0-lite-a306605-sparse`
- Local artifact copy: `experiment_FPP/runs/fpp_ddpg_saber_diagnostic_ls512_1m/`
- Upstream commit: `metacog-rl` `v0-lite` commit `a306605`
- Seeds: 1--5
- Steps: 1M
- Diagnostic frequency: 10k
- Batch-level diagnostic: 500 batches, each full batch size 256
- Logged sample rows: 64,000 rows, 128 samples/checkpoint

SAC performance sanity check는 legacy/native AUC 대신 exact shared 50k--1M grid를 사용한다. Diagnostic row와 main SABER row는 이 grid에서 동일한 curve를 가지며 final success `0.990 +/- 0.022`, AUC `0.583 +/- 0.030`, median Hit90 `750k (5/5)`다. Common-grid source는 `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/common_grid_50k/`다.

최신 결론:

- FPP DDPG에서 support indicator와 TD-error magnitude는 동일한 신호가 아니다.
- 다만 완전히 독립도 아니다. \(\psi_i\)와 \(|\delta_i|\)의 Pearson/Spearman correlation은 약 0.12로 약한 positive relation이고, top-10% high-TD-error samples에는 support-positive samples가 base rate보다 enrich된다.
- Support-positive samples는 update 전 \(|\delta|\)가 더 크고 raw residual reduction도 더 크다. 하지만 \(\Delta|\delta|/\Omega\)로 보면 support-positive와 support-negative가 비슷하므로, residual reduction 차이는 weight scale 효과를 상당히 포함한다.
- 따라서 이 diagnostic은 “SABER = PER-style high-TD-error selection”이라는 공격을 약화하지만, “support-positive sample이 본질적으로 더 reliable하다”는 pure mechanism proof는 아니다.

논문에 넣을 수 있는 가장 안전한 문장:

> FPP DDPG sample-level diagnostics show that SABER's support signal is only weakly correlated with TD-error magnitude, so it is not identical to PER-style surprise prioritization. However, support-positive samples are enriched among top high-TD-error samples and show larger raw residual reduction, so the diagnostic supports a mixed critic-side reweighting interpretation rather than a pure reliability proof.

## 2. FPP DDPG instrumented diagnostic 결과

집계 기준:

- `diagnostic_metrics.jsonl`: 각 10k step checkpoint의 full batch 256개 sample에서 계산한 batch-level metric
- 총 5 seeds \(\times\) 100 checkpoints = 500 diagnostic batches
- 아래 값은 seed별 평균을 먼저 구한 뒤, seeds 1--5 평균과 sample standard deviation을 보고한다.

| Metric | Mean +/- seed std | 해석 |
| --- | ---: | --- |
| Support-positive rate | `0.367 +/- 0.011` | batch의 약 37%가 high raw weight |
| Pearson corr(`psi`, `abs_td_before`) | `0.116 +/- 0.016` | TD-error와 약한 positive relation |
| Spearman corr(`psi`, `abs_td_before`) | `0.125 +/- 0.017` | rank 기준으로도 약한 positive relation |
| Top-10% TD-error support overlap | `0.516 +/- 0.022` | base support rate 0.367보다 enriched |
| Top-20% TD-error support overlap | `0.487 +/- 0.016` | enriched but not identical |
| Support-positive TD abs before | `0.686 +/- 0.044` | support-positive가 더 큰 residual |
| Support-negative TD abs before | `0.484 +/- 0.052` | support-negative가 더 작은 residual |
| Support-positive TD reduction | `0.019 +/- 0.003` | raw residual reduction이 더 큼 |
| Support-negative TD reduction | `0.008 +/- 0.002` | raw residual reduction이 더 작음 |
| Support-positive TD reduction / omega | `0.013 +/- 0.002` | per-weight reduction은 비슷함 |
| Support-negative TD reduction / omega | `0.011 +/- 0.003` | per-weight reduction은 비슷함 |
| Target variance support-positive | `230.4 +/- 28.2` | lower-variance target이라는 증거는 아님 |
| Target variance support-negative | `115.6 +/- 16.2` | support-negative target variance가 더 낮음 |

Seed별 diagnostic mean:

| Seed | Support rate | Pearson | Spearman | Top10 overlap | Top20 overlap | TD before + | TD before - | TD reduction + | TD reduction - |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.368 | 0.095 | 0.097 | 0.479 | 0.463 | 0.745 | 0.561 | 0.021 | 0.010 |
| 2 | 0.366 | 0.112 | 0.122 | 0.518 | 0.484 | 0.666 | 0.482 | 0.016 | 0.006 |
| 3 | 0.375 | 0.126 | 0.139 | 0.536 | 0.500 | 0.719 | 0.494 | 0.023 | 0.009 |
| 4 | 0.349 | 0.138 | 0.133 | 0.523 | 0.485 | 0.641 | 0.415 | 0.020 | 0.010 |
| 5 | 0.376 | 0.111 | 0.132 | 0.522 | 0.502 | 0.658 | 0.468 | 0.016 | 0.005 |

Performance sanity check:

| Metric | Mean +/- seed std |
| --- | ---: |
| Final success | `0.960 +/- 0.042` |
| Best success | `1.000 +/- 0.000` |
| Success AUC | `0.609 +/- 0.038` |
| Steps/sec | `41.438 +/- 0.244` |

이 row는 기존 FPP DDPG SABER main result와 일치한다.

## 3. 기존 artifact diagnostic 한계

가능한 것은 checkpoint-level aggregate 분석이다.

- `lite/psi_q_rank_rate`: batch 평균 support-positive rate
- `loss/td_error_abs_mean`: batch 평균 absolute TD-error
- `loss/td_error_abs_p90`: batch TD-error p90
- evaluation success curve

불가능한 것은 sample-level mechanism 분석이다.

- sample별 `psi_i`
- sample별 `|delta_i|`
- support-positive sample과 top-k TD-error sample overlap
- support-positive vs support-negative TD-error distribution
- update 전후 residual reduction by support group
- support-positive vs support-negative target variance

따라서 기존 artifact diagnostic은 sanity check로만 쓰고, 논문의 mechanism closure에는 instrumented rerun이 필요하다.

## 4. 생성한 기존 artifact diagnostic

생성 파일:

- `analysis/support_td_error/analyze_existing_fpp_support_td_error.py`
- `analysis/support_td_error/fpp_support_td_error_existing_report.md`
- `analysis/support_td_error/fpp_support_td_error_existing_summary.csv`
- `analysis/support_td_error/fpp_support_td_error_existing_per_seed.csv`
- `analysis/support_td_error/fpp_support_td_error_existing_summary.json`

분석 대상:

| Method | Source |
| --- | --- |
| FPP-DDPG-SABER | `experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512` seeds 1--5 |
| FPP-TD3-SABER | `experiment_FPP/runs/td3_saber_critic_only_v4_ls512` seeds 1--5 |
| FPP-SAC-SABER | `experiment_FPP/runs/sac_saber_critic_only_v4_ls512` seeds 1--5 |

각 run에서 `metrics_selected_10k.csv`를 읽고, step별 마지막 finite row만 사용했다. 같은 step의 train/eval row 중복을 줄이기 위한 처리다.

## 5. 기존 artifact 결과

Checkpoint-level correlation 결과:

| Method | n | Support rate | TD abs mean | corr(support, TD mean) | Spearman | corr(support, TD p90) | corr(support, eval success) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FPP-DDPG-SABER | 5 | `0.367 +/- 0.011` | `0.560 +/- 0.050` | `0.670 +/- 0.078` | `0.679 +/- 0.060` | `0.682 +/- 0.064` | `0.442 +/- 0.123` |
| FPP-TD3-SABER | 5 | `0.219 +/- 0.009` | `0.356 +/- 0.027` | `0.290 +/- 0.092` | `0.325 +/- 0.168` | `0.362 +/- 0.086` | `0.678 +/- 0.152` |
| FPP-SAC-SABER | 5 | `0.197 +/- 0.013` | `0.276 +/- 0.019` | `-0.463 +/- 0.138` | `-0.359 +/- 0.104` | `-0.453 +/- 0.126` | `0.758 +/- 0.045` |

해석:

- DDPG에서는 support rate와 TD-error aggregate가 같이 증가하는 경향이 있다.
- TD3에서는 약한 positive correlation이다.
- SAC에서는 negative correlation이다.
- 세 backbone에서 방향이 일관되지 않으므로, checkpoint-level aggregate만 보면 support와 TD-error가 같은 신호라고 보기는 어렵다.
- 하지만 이 분석은 training progress confound가 크다. support rate와 TD-error 모두 시간에 따라 변하므로, 이 결과를 mechanism proof로 쓰면 안 된다.

논문에 쓸 수 있는 안전한 문장:

> Existing checkpoint-level logs show that support coverage and TD-error magnitude are not interchangeable aggregate signals across backbones, but they do not provide sample-level evidence. We therefore treat this as a diagnostic sanity check and use instrumented logging for the final mechanism analysis.

## 6. 추가로 필요한 diagnostic

FPP DDPG/TD3/SAC sample-level diagnostics는 완료되었다. 다음 확장 우선순위는 DrawerOpen large-effect setting이다.

### 6.1 Sample-level support vs TD-error correlation

각 diagnostic checkpoint에서 mini-batch sample별로 다음을 저장한다.

| Column | 의미 |
| --- | --- |
| `global_step` | training step |
| `seed` | seed |
| `sample_id` | diagnostic row id |
| `replay_index` | sampled replay index |
| `phase` | task phase |
| `psi` | support-positive indicator |
| `local_pool_size` | exact local context pool size |
| `support_count` | available support count |
| `w_raw` | raw SABER weight |
| `omega` | batch-normalized critic weight |
| `td_abs_before` | critic update 전 absolute TD-error |
| `td_abs_after` | critic update 후 absolute TD-error |
| `action_exec_bin` | replay action bin |
| `policy_action_bin` | current policy action bin |
| `her_relabel` | HER relabel 여부 |

필수 통계:

\[
\operatorname{corr}(\psi_i, |\delta_i|)
\]

\[
\text{Top-}k\text{ overlap}(\psi_i, |\delta_i|)
\]

\[
\mathbb E[|\delta|_{\text{after}}-|\delta|_{\text{before}}\mid \psi=1]
\quad \text{vs}\quad
\mathbb E[|\delta|_{\text{after}}-|\delta|_{\text{before}}\mid \psi=0]
\]

### 6.2 권장 logging budget

FPP DDPG는 완료되었으므로, 다음 우선순위는 FPP TD3 또는 FPP SAC다.

- Seeds: 1--3 먼저, 가능하면 1--5
- Horizon: 1M
- Diagnostic interval: 10k 또는 20k steps
- Samples per checkpoint: 64 또는 128
- 예상 row 수:
  - 1M, interval 10k, 64 samples, 5 seeds = 약 32k rows
  - 충분히 작고 논문 artifact로 관리 가능하다.

### 6.3 성공 기준

가장 좋은 결과:

- `corr(psi_i, |delta_i|)`가 낮거나 backbone/task별로 불안정하다.
- Top-k TD-error sample과 support-positive sample overlap이 낮다.
- support-positive sample의 residual reduction이 더 안정적이거나 variance가 낮다.

이 경우 주장:

> SABER does not simply prioritize high-error transitions; it selects locally supported, policy-relevant Bellman supervision.

나쁜 결과:

- `psi_i`와 `|delta_i|`가 매우 높게 correlate된다.
- top-k overlap이 높다.
- residual reduction 차이가 없다.

이 경우 claim을 낮춰야 한다.

> SABER behaves as a support-conditioned critic regularizer whose empirical benefit may partly overlap with TD-error-based reweighting.

## 7. 현재 판단

기존 artifact만으로는 C11을 닫지 못했지만, FPP DDPG instrumented diagnostic으로 PER-style equivalence 공격은 상당히 약해졌다. 다만 pure support-reliability mechanism은 아직 닫히지 않는다.

- support와 TD-error는 sample-level에서도 동일한 신호가 아니다.
- 그러나 support-positive samples는 high TD-error에 enriched되어 있고 raw residual reduction도 더 크다.
- 따라서 논문 claim은 “TD-error와 구별되는 support-conditioned signal을 사용하는 critic-side reweighting”까지는 가능하지만, “support-positive가 본질적으로 lower-risk Bellman target”이라는 강한 claim은 피해야 한다.
- 다음 작업은 FPP TD3/SAC diagnostic 또는 DrawerOpen diagnostic으로 이 패턴이 backbone/task별로 어떻게 달라지는지 확인하는 것이다.

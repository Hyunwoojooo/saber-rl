# SABER AAAI-27 Support vs TD-error Diagnostic

작성일: 2026-06-24
목적: SABER가 단순 TD-error weighting 또는 PER-style weighting의 변형이라는 반론을 줄이기 위해, 현재 artifact로 가능한 support-vs-TD-error 분석 범위와 추가로 필요한 diagnostic을 정리한다.

## 1. 결론

현재 repository에 있는 기존 FPP artifact만으로는 reviewer가 요구할 만한 핵심 diagnostic을 닫을 수 없다.

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

## 2. 생성한 기존 artifact diagnostic

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

## 3. 기존 artifact 결과

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

## 4. 최종 논문에 필요한 진짜 diagnostic

Mechanism closure에 필요한 최소 diagnostic은 sample-level이다.

### 4.1 Sample-level support vs TD-error correlation

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

### 4.2 권장 logging budget

FPP DDPG SABER부터 최소로 시작한다.

- Seeds: 1--3 먼저, 가능하면 1--5
- Horizon: 1M
- Diagnostic interval: 10k 또는 20k steps
- Samples per checkpoint: 64 또는 128
- 예상 row 수:
  - 1M, interval 10k, 64 samples, 5 seeds = 약 32k rows
  - 충분히 작고 논문 artifact로 관리 가능하다.

### 4.3 성공 기준

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

## 5. 현재 판단

기존 artifact만으로는 C11을 닫지 못한다. 다만 기존 checkpoint-level diagnostic은 다음 판단을 가능하게 한다.

- support와 TD-error는 적어도 aggregate level에서 backbone별로 같은 방향으로 움직이지 않는다.
- sample-level proof는 아직 없다.
- 따라서 다음 작업은 instrumented FPP DDPG SABER rerun 설계다.

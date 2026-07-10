# SABER AAAI-27 Result Matrix / Artifact Inventory

작성일: 2026-06-24
최근 업데이트: 2026-07-10
목적: GitHub remote branches와 local working tree에 남아 있는 SABER 관련 실험 결과를 논문 사용 가능성 기준으로 분류한다.

## 1. 조사 범위

이번 inventory는 local checkout만 보지 않고 GitHub remote branch를 함께 확인했다.

| Source | Commit | 역할 |
| --- | --- | --- |
| `origin/main` | `4b84853` / FPP SAC common-grid reanalysis with pinned dependency | FPP, FetchPush, DrawerOpen/DrawerClose 및 common-grid artifact source |
| `origin/fetchslide-runtime-artifacts` | `2a8d34e` / `Add FetchSlide runtime experiment artifacts` | FetchSlide runtime/control 결과의 primary source |
| `origin/aaai27-core-artifacts` | `1142c97` / `Add AAAI27 paper draft` | 논문 초안과 core artifact branch |
| `Hyunwoojooo/metacog-rl@v0-lite` | `51fb99f` / `Add FPP TD3 random control results` | FPP TD3 Random-support/Random-fixed-count artifact source |
| `Hyunwoojooo/metacog-rl@v0-lite` | `d5439b2` / `Add FPP SAC random control results` | FPP SAC Random-support/Random-fixed-count artifact source |
| `Hyunwoojooo/metacog-rl@v0-lite` | `783a34c` / `Add FPP TD3 high-RR x4 results` | FPP TD3 high replay-ratio \(4\times\) artifact source |
| `Hyunwoojooo/metacog-rl@v0-lite` | `62de0f7` / `Add FPP SAC high replay-ratio x4 results` | FPP SAC high replay-ratio \(4\times\) artifact source |
| `Hyunwoojooo/metacog-rl@v0-lite` | `8fd6f8e` / `Add FPP REDQ-HER TQB full results` | FPP REDQ-HER UTD4 N10 M2 target-Q-bound artifact source |
| local `aaai27-core-artifacts` | `34d7a69` + uncommitted integration | 현재 작업 중인 논문, pipeline, common-grid package |

조사는 dirty worktree를 보호하기 위해 branch checkout 없이 `/private/tmp/saber_wt_origin_main`, `/private/tmp/saber_wt_fetchslide` worktree에서 수행했다.

## 2. 결론 요약

가장 중요한 결론은 다음이다.

- FPP main table은 이미 충분하다. DDPG/TD3/SAC HER baseline과 critic-only SABER가 모두 1M, seeds 1--5, summary 기반으로 존재한다.
- FPP SAC의 legacy 25k main rows와 newer 10k controls는 exact 50k--1M common grid로 정식 재계산됐다. Interpolation/forward-fill 없이 20 checkpoints를 사용하며, 70개 source file hash와 7개 analyzer test가 검증됐다.
- FPP PER-style/PER-full baseline도 1M, seeds 1--5로 존재한다.
- FPP TD-error는 log 기반 partial 630k 재집계를 완료했다. Run directory는 1M으로 되어 있으나 실제 모든 seed가 약 633k train steps에서 종료되었고, `metrics_selected_10k.csv`의 evaluation columns는 비어 있어 `run.log`에서 evaluation curve를 복원했다. 따라서 1M main row로 쓰면 안 되고 partial diagnostic 또는 rerun 판단용으로만 쓴다.
- FPP DDPG Random-Support/Shuffled-Weight/Random-Fixed-Count는 1M, seeds 1--5로 완료되어 main mechanism control로 사용할 수 있다.
- FPP TD3 Random-Support와 Random-Fixed-Count k73도 1M, seeds 1--5로 완료되었다. 이 결과는 Random-Support/Random-Fixed가 HER 수준에 머물고 SABER와 분리된다는 점에서 DDPG보다 강한 mechanism evidence를 제공한다.
- FPP SAC Random-Support와 Random-Fixed-Count k76도 1M, seeds 1--5로 완료되었다. Random-Support는 HER와 거의 같은 AUC에 머물고, Random-Fixed-Count는 final success를 포화시키지만 SABER보다 낮은 AUC를 보인다.
- FPP TD3 high replay-ratio \(4\times\)도 1M, seeds 1--5로 완료되었다. AUC는 TD3 SABER와 비슷하지만 final success와 Hit90 seed count는 낮고 runtime/update budget은 더 크다.
- FPP SAC high replay-ratio \(4\times\)는 common-grid AUC 0.768로 SABER 0.583보다 높지만, recorded runtime은 약 47.0h로 SABER의 약 9.6h보다 훨씬 길다. AUC/hour는 hardware-normalized metric이 아니므로 main claim에서 제거한다.
- FPP REDQ-HER target-Q-bound \(N=10,M=2,\mathrm{UTD}=4\)는 common-grid final 1.000, AUC 0.867, median Hit90 300k로 가장 강한 FPP learning curve지만, 10-critic ensemble과 약 89.5h recorded runtime을 사용하는 architecture/compute-scaled baseline이다.
- FetchSlide는 `origin/fetchslide-runtime-artifacts`에 DDPG/TD3/SAC baseline/SABER 및 TD-error/PER-style 1M 결과가 있다.
- FetchSlide Random/Shuffled는 1M 결과가 없고 1k smoke 수준만 있다.
- FetchPush는 `origin/main`에 DDPG와 TD3 결과가 생각보다 많다. 다만 main paper가 아니라 appendix evidence로 쓰는 것이 안전하다.
- DrawerOpen은 DDPG 2M HER/SABER/Random-Support/PER-Style/Random-Fixed/Shuffled 결과와 TD3 2M controls가 풍부하다. DDPG는 HER 대비 large-effect/seed-consistency evidence로 유용하지만 Random-Support/PER-Style/Random-Fixed도 강하므로 structured support mechanism의 primary proof로만 쓰면 안 된다.
- 기존 DrawerOpen `HGR` row는 canonical HGR의 importance-sampling correction을 끈 sampler-only ablation이다. 논문과 pipeline에서는 `HGR-style sampler-only`로만 표기하며 canonical HGR direct baseline claim에는 사용하지 않는다.
- DrawerClose TD3 2M HER/SABER/Random-Support/Random-Fixed/Shuffled/PER-Style results도 `origin/main`에 있다. Shuffled-Weight가 SABER를 match하므로 appendix/stress-test와 mixed-mechanism limitation evidence로 쓰는 것이 안전하다.

## 3. Main Paper에 바로 쓸 수 있는 결과

### 3.1 FetchPickAndPlace primary table

Source: `origin/main`, local에도 동일 계열 artifact 존재.

| Row | Path | Steps | Seeds | 상태 | 권장 사용 |
| --- | --- | ---: | --- | --- | --- |
| DDPG HER | `experiment_FPP/runs/B0` and `experiment_FPP/results/fpp_b3_v4_ls512_1m_summary.json` | 1M | 1--5 | ready | main |
| DDPG HER+SABER | `experiment_FPP/results/fpp_b3_v4_ls512_1m_summary.json` | 1M | 1--5 | ready | main |
| TD3 HER | `experiment_FPP/runs/td3_her_baseline_ls512` | 1M | 1--5 | ready | main |
| TD3 HER+SABER | `experiment_FPP/runs/td3_saber_critic_only_v4_ls512` | 1M | 1--5 | ready | main |
| SAC HER | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m` | 1M | 1--5 | common-grid ready | main |
| SAC HER+SABER | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m` | 1M | 1--5 | common-grid ready | main |
| REDQ-HER TQB | `experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m` | 1M | 1--5 | common-grid ready | strong architecture/compute baseline |

주의: `experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512`에는 seeds 1--10와 중간 500k run이 함께 있어 단순 recursive 평균을 내면 논문 값이 틀어진다. DDPG SABER main row는 `experiment_FPP/results/fpp_b3_v4_ls512_1m_summary.json`의 `B3_v4_512` summary를 source of truth로 둔다.

현재 paper table의 DDPG row와 일치하는 aggregate:

| Method | Final | Best | AUC | Steps/sec |
| --- | ---: | ---: | ---: | ---: |
| DDPG HER | `0.940 ± 0.042` | `1.000 ± 0.000` | `0.539 ± 0.125` | `65.02 ± 0.10` |
| DDPG HER+SABER | `0.960 ± 0.042` | `1.000 ± 0.000` | `0.609 ± 0.038` | `41.72 ± 0.20` |

Strong compute/backbone reference:

| Method | Final | Best | AUC | Hit90 | Runtime h | Steps/sec | 해석 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| REDQ-HER TQB \(N=10,M=2,\mathrm{UTD}=4\) | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.867 ± 0.052` | `300k (5/5)` | `89.52` | `3.10` | Exact 50k common-grid에서 가장 강한 learning curve. Architecture/compute-scaled reference로 사용. |

FPP SAC exact common-grid source of truth:

| Method | Final | AUC | Hit90 | Paired AUC 해석 |
| --- | ---: | ---: | ---: | --- |
| HER | `0.790 ± 0.288` | `0.471 ± 0.079` | `800k (3/5)` | reference |
| HER+SABER | `0.990 ± 0.022` | `0.583 ± 0.030` | `750k (5/5)` | vs HER `+0.112 [0.064, 0.161]` |
| Random-Support | `0.690 ± 0.263` | `0.482 ± 0.069` | `725k (2/5)` | SABER minus control `+0.101 [0.034, 0.164]` |
| Random-Fixed-Count | `1.000 ± 0.000` | `0.526 ± 0.059` | `900k (5/5)` | SABER minus control `+0.058 [0.003, 0.113]` |
| HER high-RR \(4\times\) | `0.960 ± 0.042` | `0.768 ± 0.097` | `450k (5/5)` | SABER minus high-RR `-0.184 [-0.235, -0.123]` |
| REDQ-HER TQB | `1.000 ± 0.000` | `0.867 ± 0.052` | `300k (5/5)` | REDQ minus SABER `+0.283 [0.213, 0.329]` |

### 3.2 FPP PER / weighting comparison

| Row | Path | Steps | Seeds | 상태 | 권장 사용 |
| --- | --- | ---: | --- | --- | --- |
| DDPG HER+PER-style loss weighting | `experiment_FPP/runs/per_style_loss_weighting_ls512` | 1M | 1--5 | ready | main/appendix |
| DDPG HER+PER-full | `experiment_FPP/runs/ddpg_her_per_full_ls512` | 1M | 1--5 | ready | main/appendix |
| DDPG HER+TD-error weighting | `experiment_FPP/runs/TD-error` | partial 630k | 1--5 | log-reaggregated | diagnostic only, 1M rerun 권장 |

TD-error는 각 seed에 `metrics_final_row.json`, `metrics_selected_10k.csv`, `config.json`, `env_snapshot.json`, `run.log`가 있다. 확인 결과 `metrics_selected_10k.csv`의 evaluation columns는 비어 있고, `run.log`에는 evaluation line이 남아 있다. 따라서 `experiment_FPP/shared/aggregate_td_error_from_logs.py`로 log 기반 재집계를 수행했다.

Generated files:

- `experiment_FPP/results/fpp_td_error_partial_630k_analysis.md`
- `experiment_FPP/results/fpp_td_error_partial_630k_per_seed.csv`
- `experiment_FPP/results/fpp_td_error_partial_630k_method_summary.csv`
- `experiment_FPP/results/fpp_td_error_partial_630k_summary.json`

주의: 위 generated files는 현재 `.gitignore`의 `results/` 패턴에 걸리므로, commit할 때는 `git add -f experiment_FPP/results/fpp_td_error_partial_630k_*`가 필요하다.

Partial TD-error aggregate:

| Method | Actual final step | Last eval | Final | Best | AUC | Tail10 | Hit90 median | Steps/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DDPG HER+TD-error weighting | `634k ± 1k` | `630k` | `0.940 ± 0.042` | `0.960 ± 0.042` | `0.383 ± 0.044` | `0.793 ± 0.102` | `550k (5/5)` | `61.26 ± 0.12` |

해석: TD-error partial run은 630k 기준 final success는 높지만, AUC span이 10k--630k로 제한되어 1M FPP rows와 직접 비교할 수 없다. 본문 PER/TD-error 방어선으로 쓰려면 TD-error를 1M까지 rerun하거나, B0/SABER/PER-style/PER-full을 모두 같은 630k horizon으로 잘라 별도 same-horizon diagnostic table을 만들어야 한다.

### 3.3 FetchSlide secondary table

Primary source: `origin/fetchslide-runtime-artifacts`.

| Row | Path | Steps | Seeds | 상태 | 권장 사용 |
| --- | --- | ---: | --- | --- | --- |
| DDPG HER | `experiment_FetchSlide/experiment/FetchSlide/b0` | 1M | 1--5 | ready | secondary main |
| DDPG HER+SABER | `experiment_FetchSlide/experiment/FetchSlide/b3_critic_only_runtime_opt_v5` | 1M | 1--5 | ready | secondary main |
| TD3 HER | `experiment_FetchSlide/experiment/FetchSlide/td3` | 1M | 1--5 | ready | secondary main |
| TD3 HER+SABER | `experiment_FetchSlide/experiment/FetchSlide/td3_saber` | 1M | 1--5 | ready | secondary main |
| TD3 HER+SABER v4 ls512 | `experiment_FetchSlide/experiment/FetchSlide/td3_saber_critic_only_v4_ls512` | 1M | 1--5 | ready | appendix or alternate main row |
| SAC HER | `experiment_FetchSlide/experiment/FetchSlide/sac` | 1M | 1--5 | ready | secondary main |
| SAC HER+SABER | `experiment_FetchSlide/experiment/FetchSlide/sac_saber` | 1M | 1--5 | ready | secondary main |
| SAC HER+SABER v4 ls512 | `experiment_FetchSlide/experiment/FetchSlide/sac_saber_critic_only_v4_ls512` | 1M | 1--5 | ready | appendix or alternate main row |
| DDPG TD-error | `experiment_FetchSlide/experiment/FetchSlide/td_error` | 1M | 1--5 | ready | appendix |
| DDPG PER-style | `experiment_FetchSlide/experiment/FetchSlide/per_style` | 1M | 1--5 | ready | appendix |

추천: 본문 FetchSlide table은 현재 draft처럼 `b0`, `b3_critic_only_runtime_opt_v5`, `td3`, `td3_saber`, `sac`, `sac_saber`를 쓰고, v4 ls512 variants는 appendix에 둔다. v4 ls512가 더 좋은 숫자를 일부 보이지만, main claim은 좁고 일관된 method naming이 더 중요하다.

## 4. Appendix 또는 Stress-Test로 쓸 수 있는 결과

### 4.1 FetchPush

Source: `origin/main`.

FetchPush는 결과가 많지만 main paper의 primary evidence로 크게 밀기보다는 appendix/supporting evidence로 쓰는 것이 안전하다.

| Family | Row | Path | Steps | Seeds | 상태 | 비고 |
| --- | --- | --- | ---: | --- | --- | --- |
| DDPG | HER | `FetchPush_DDPG/DDPG-HER-B0/results` | 1M | 1--5 | ready | appendix 가능 |
| DDPG | HER+SABER FPP-style | `FetchPush_DDPG/DDPG-HER-B3-Critic-Only-FPP-Style/results` | 1M | 1--5 | ready | appendix 가능 |
| DDPG | PER-style | `FetchPush_DDPG/DDPG-HER-PER-Style/results` | 1M | 1--5 | ready | appendix 가능 |
| TD3 | HER baseline | `FetchPush_TD3/TD3-HER-B0` | - | - | no results | code/scripts only |
| TD3 | Critic-only SABER | `FetchPush_TD3/TD3-HER-B3-Critic-Only/results` | 1M | 1--5 | ready | no paired TD3 HER baseline in same folder |
| TD3 | Actor-only | `FetchPush_TD3/TD3-HER-B3-Actor-Only/results` | 1M | 1--5 | ready | ablation |
| TD3 | Actor+Critic | `FetchPush_TD3/TD3-HER-B3-Actor-Critic/results` | 1M | 1--5 | ready | ablation |
| TD3 | Random-Support | `FetchPush_TD3/FetchPush_Random_Shuffled_Uniform_Ablation/TD3-HER-B3-Critic-Only-Random-Support/results_td3_b3_critic_only_random_support` | 1M | 1--5 | ready | mechanism control, appendix |
| TD3 | Shuffled-Weight | `FetchPush_TD3/FetchPush_Random_Shuffled_Uniform_Ablation/TD3-HER-B3-Critic-Only-Shuffled-Weight/results_td3_b3_critic_only_shuffled_weight` | 1M | 1--5 | ready | mechanism control, appendix |
| SAC | HER/SABER folders | `FetchPush_SAC/*` | - | - | no result summaries found | code only or missing results |

Observed aggregate examples:

| Row | Final | AUC | Last-quarter AUC |
| --- | ---: | ---: | ---: |
| FetchPush DDPG HER | `0.990 ± 0.022` | `0.727 ± 0.033` | `0.969 ± 0.017` |
| FetchPush DDPG HER+SABER FPP-style | `1.000 ± 0.000` | `0.725 ± 0.027` | `0.967 ± 0.013` |
| FetchPush DDPG PER-style | `0.990 ± 0.022` | `0.738 ± 0.038` | `0.964 ± 0.016` |
| FetchPush TD3 critic-only SABER | `1.000 ± 0.000` | `0.662 ± 0.039` | `0.990 ± 0.008` |
| FetchPush TD3 random-support | `1.000 ± 0.000` | `0.654 ± 0.041` | `0.989 ± 0.004` |
| FetchPush TD3 shuffled-weight | `0.970 ± 0.067` | `0.679 ± 0.053` | `0.988 ± 0.010` |

해석: FetchPush는 DDPG에서 baseline이 이미 거의 saturated라 SABER improvement claim이 약하다. TD3 controls는 mechanism discussion에는 유용하지만 HER baseline row가 없어서 main cross-backbone table에 넣기 어렵다.

### 4.2 DrawerOpen

Source: `origin/main`.

| Family | Row | Path | Steps | Seeds | 상태 | 권장 사용 |
| --- | --- | --- | ---: | --- | --- | --- |
| TD3 | HER baseline | `metaworld_Drawer_Open_TD3/results_2m/TD3-HER-B0-2M` | 2M | 1--5 | ready | stress-test |
| TD3 | Critic-only SABER | `metaworld_Drawer_Open_TD3/results_2m/TD3-HER-B3-Critic-Only-2M` | 2M | 1--5 | ready | stress-test |
| TD3 | Actor-only | `metaworld_Drawer_Open_TD3/results_2m/TD3-HER-B3-Actor-Only-2M` | 2M | 1--5 | ready | appendix/control |
| TD3 | Actor+Critic | `metaworld_Drawer_Open_TD3/results_2m/TD3-HER-B3-Actor-Critic-2M` | 2M | 1--5 | ready | appendix/control |
| TD3 | Random-Support | `metaworld_Drawer_Open_TD3/results_2m/TD3-HER-B3-Random-Support-2M` | 2M | 1--5 | ready | limitation |
| TD3 | Random-Fixed-Count | `metaworld_Drawer_Open_TD3/results_2m/TD3-HER-B3-Random-Fixed-Count-2M` | 2M | 1--5 | ready | control |
| TD3 | Shuffled-Weight | `metaworld_Drawer_Open_TD3/results_2m/TD3-HER-B3-Shuffled-Weight-2M` | 2M | 1--5 | ready | control |
| DDPG | HER baseline | `metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B0-2M` | 2M | 1--5 | ready | appendix only |
| DDPG | Critic-only SABER | `metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Critic-Only-2M` | 2M | 1--5 | ready | appendix only |
| DDPG | HGR-style sampler-only | `metaworld_Drawer_Open_DDPG_HGR/runs/ddpg_hgr_tdpriority_2m_opt` | 2M | 1--5 | ready ablation | sampling control only; not canonical HGR |
| DDPG | Random-Support | `metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Random-Support-2M` | 2M | 1--5 | ready | control/limitation |
| DDPG | PER-Style | `metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-PER-Style-2M` | 2M | 1--5 | ready | control/limitation |
| DDPG | Random-Fixed-Count | `metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Random-Fixed-Count-2M` | 2M | 1--5 | ready | control/limitation |
| DDPG | Shuffled-Weight | `metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Shuffled-Weight-2M` | 2M | 1--5 | ready | control/limitation |
| SAC | HER baseline | `metaworld_Drawer_Open_SAC/results_2m/SAC-HER-B0-2M` | 2M | 1--5 | ready | negative appendix |

Important DrawerOpen numbers:

| Row | Final | Best | AUC |
| --- | ---: | ---: | ---: |
| TD3 HER | `0.190 ± 0.425` | `0.210 ± 0.442` | `0.127 ± 0.283` |
| TD3 Critic-only SABER | `0.590 ± 0.539` | `0.620 ± 0.522` | `0.362 ± 0.338` |
| TD3 Actor+Critic | `0.600 ± 0.548` | `0.610 ± 0.534` | `0.432 ± 0.395` |
| TD3 Random-Support | `0.600 ± 0.548` | `0.600 ± 0.548` | `0.426 ± 0.392` |
| TD3 Shuffled-Weight | `0.000 ± 0.000` | `0.010 ± 0.022` | `0.000 ± 0.000` |
| DDPG HER | `0.390 ± 0.534` | `0.400 ± 0.548` | `0.207 ± 0.287` |
| DDPG Critic-only SABER | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.754 ± 0.044` |
| DDPG HGR-style sampler-only | `0.400 ± 0.548` | `0.400 ± 0.548` | `0.311 ± 0.425` |
| DDPG Random-Support | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.706 ± 0.122` |
| DDPG PER-Style | `0.800 ± 0.447` | `1.000 ± 0.000` | `0.715 ± 0.090` |
| DDPG Random-Fixed-Count | `0.800 ± 0.447` | `0.800 ± 0.447` | `0.621 ± 0.352` |
| DDPG Shuffled-Weight | `0.400 ± 0.548` | `0.420 ± 0.531` | `0.400 ± 0.547` |
| SAC HER | `0.000 ± 0.000` | `0.000 ± 0.000` | `0.000 ± 0.000` |

해석: DrawerOpen TD3는 Random-Support가 Critic-only와 비슷하거나 더 높다. 따라서 structured support mechanism의 주증거가 아니라 limitation/stress-test로만 써야 한다. DDPG DrawerOpen은 매우 강한 positive result지만 Random-Support/PER-Style/Random-Fixed도 강하므로 HER 대비 large-effect evidence로 쓰고 causal mechanism proof로 쓰면 안 된다.

### 4.3 DrawerClose

Source: `origin/main`.

| Family | Row | Path | Steps | Seeds | 상태 | 권장 사용 |
| --- | --- | --- | ---: | --- | --- | --- |
| TD3 | HER baseline | `metaworld_Drawer_Close_TD3/results_2m/TD3-HER-B0-2M` | 2M | 1--5 | ready | stress-test |
| TD3 | Critic-only SABER | `metaworld_Drawer_Close_TD3/results_2m/TD3-HER-B3-Critic-Only-2M` | 2M | 1--5 | ready | stress-test |
| TD3 | Random-Fixed-Count | `metaworld_Drawer_Close_TD3/results_2m/TD3-HER-B3-Random-Fixed-Count-2M` | 2M | 1--5 | ready | control |
| TD3 | Random-Support | `metaworld_Drawer_Close_TD3/results_2m/TD3-HER-B3-Random-Support-2M` | 2M | 1--5 | ready | control |
| TD3 | Shuffled-Weight | `metaworld_Drawer_Close_TD3/results_2m/TD3-HER-B3-Shuffled-Weight-2M` | 2M | 1--5 | ready | stress/control |
| TD3 | PER-Style | `metaworld_Drawer_Close_TD3/results_2m/TD3-HER-PER-Style-2M` | 2M | 1--5 | ready | stress/control |

Important DrawerClose numbers:

| Row | Final | Best | AUC | 해석 |
| --- | ---: | ---: | ---: | --- |
| TD3 HER | `0.350 ± 0.480` | `1.000 ± 0.000` | `0.329 ± 0.348` | high variance, late collapse |
| TD3 Critic-only SABER | `0.590 ± 0.539` | `1.000 ± 0.000` | `0.549 ± 0.422` | mean은 높지만 HER 대비 CI가 넓음 |
| TD3 Random-Fixed-Count | `0.000 ± 0.000` | `1.000 ± 0.000` | `0.133 ± 0.026` | final collapse, SABER가 명확히 높음 |
| TD3 Random-Support | `0.200 ± 0.447` | `1.000 ± 0.000` | `0.296 ± 0.356` | HER와 유사/낮음 |
| TD3 Shuffled-Weight | `0.590 ± 0.539` | `1.000 ± 0.000` | `0.586 ± 0.295` | SABER final과 같고 AUC는 약간 높음 |
| TD3 PER-Style | `0.200 ± 0.447` | `1.000 ± 0.000` | `0.369 ± 0.275` | random-support보다 AUC는 높지만 high variance |

해석: DrawerClose TD3는 main positive result가 아니라 high-variance final-stability stress-test다. 모든 row가 early Hit90에 도달할 수 있으므로 final success와 best-final drop을 함께 봐야 한다. Shuffled-Weight가 SABER를 match하므로 structured support의 causal proof로 쓰면 안 되고, mixed-mechanism limitation evidence로만 사용하는 것이 안전하다.

## 5. 사용하면 안 되거나 추가 작업이 필요한 결과

### 5.1 FPP mechanism controls

| Row | 현재 상태 | 판단 |
| --- | --- | --- |
| FPP DDPG Random-Support | `experiment_FPP/runs/b3_support_random_critic_only_ls512` | 1M, seeds 1--5, main control 가능 |
| FPP DDPG Shuffled-Weight | `experiment_FPP/runs/b3_support_shuffled_critic_only_ls512` | 1M, seeds 1--5, main control 가능 |
| FPP DDPG Random-Fixed-Count k94 | `experiment_FPP/runs/b3_random_fixed_count_k94_critic_only_ls512` | 1M, seeds 1--5, main control 가능 |
| FPP TD3 Random-Support | `experiment_FPP/runs/td3_support_random_critic_only_ls512` | 1M, seeds 1--5, main control 가능 |
| FPP TD3 Random-Fixed-Count k73 | `experiment_FPP/runs/td3_random_fixed_count_k73_critic_only_ls512` | 1M, seeds 1--5, main control 가능 |
| FPP SAC Random-Support | `experiment_FPP/runs/sac_support_random_critic_only_ls512` | 1M, seeds 1--5, main/appendix control 가능 |
| FPP SAC Random-Fixed-Count k76 | `experiment_FPP/runs/sac_random_fixed_count_k76_critic_only_ls512` | 1M, seeds 1--5, main/appendix control 가능 |
| FPP SAC HER high-RR \(4\times\) | `experiment_FPP/runs/sac_her_high_replay_ratio_x4` | 1M, seeds 1--5, compute-scaling control 가능 |
| FPP TD-error | log 기반 partial 630k summary 생성 완료 | 1M main row 불가, rerun 또는 same-horizon diagnostic 필요 |

FPP DDPG controls는 Random-Support가 SABER와 거의 같아 exact sample-level alignment claim을 약화한다. 반대로 FPP TD3 controls는 Random-Support/Random-Fixed-Count가 HER 수준에 머물고 SABER와 분리되어 random reweighting만으로 모든 gain이 설명되지 않음을 보여준다. FPP SAC random controls는 중간적이다. Random-Support는 HER 수준에 가깝고, Random-Fixed-Count는 final success를 포화시키지만 AUC에서는 SABER보다 낮다. 반면 SAC high-RR \(4\times\)는 AUC에서 SABER보다 높으므로, compute scaling dominance claim은 금지하고 one-update-per-step matched-backbone claim으로 제한해야 한다.

### 5.2 FetchSlide mechanism controls

| Row | 현재 상태 | 판단 |
| --- | --- | --- |
| FetchSlide TD-error | 1M seeds 1--5 ready | appendix 가능 |
| FetchSlide PER-style | 1M seeds 1--5 ready | appendix 가능 |
| FetchSlide Random-Support | old `experiment/FetchSlide/b3_critic_only_random`, 1k only | main/appendix 불가 |
| FetchSlide Shuffled-Weight | old `experiment/FetchSlide/b3_critic_only_shuffled`, 1k only | main/appendix 불가 |
| FetchSlide Random-Fixed-Count | 발견 못함 | 필요하면 새 실험 |

### 5.3 FetchPush gaps

| Row | 상태 |
| --- | --- |
| FetchPush TD3 HER baseline | code/scripts only, result summaries not found |
| FetchPush SAC HER/SABER | folders exist, result summaries not found |
| FetchPush full cross-backbone table | 현재 artifact만으로는 불완전 |

FetchPush를 appendix에 넣는다면 DDPG subset과 TD3 ablation subset으로 분리해야 한다. Full DDPG/TD3/SAC backbone table로 쓰면 안 된다.

## 6. 추천 논문 배치

### Main paper

1. FPP DDPG/TD3/SAC HER vs HER+SABER.
2. FPP DDPG HER vs SABER vs PER-style vs PER-full.
3. FetchSlide DDPG/TD3/SAC HER vs HER+SABER as secondary evidence and limitation.
4. DrawerOpen TD3 as stress-test/limitation only, with Random-Support discussion.

### Appendix

1. FPP additional variants: v4 ls1024, v6 anneal, v7/v8 loss variants, seed6--10 sensitivity.
2. FetchSlide TD-error/PER-style and v4 ls512 variants.
3. FetchPush DDPG subset and TD3 ablation/control subset.
4. DrawerOpen DDPG/SAC/TD3 controls.

### 새 실험 필요

우선순위는 다음이다.

1. FPP DDPG canonical HGR 1M, seeds 1--5를 importance correction과 documented tuning으로 완료한다.
2. 진행 중인 DrawerOpen/DrawerClose HGR와 REDQ rows가 끝나면 HGR importance correction/config fidelity를 먼저 audit한다. Correction이 없으면 canonical이 아니라 sampler-only row로 등록한다.
3. FPP TD-error 1M rerun 또는 same-horizon 630k diagnostic table 작성 여부를 결정한다.
4. 최종 원고에서는 DDPG/TD3/SAC control evidence를 main/appendix 배치에 맞춰 압축한다.

## 7. 다음 액션

FPP SAC common-grid artifact는 pipeline과 두 manuscript source에 연결됐다. 다음 결과 업데이트는 FPP canonical HGR와 audit를 통과한 Drawer HGR/REDQ runs를 기존 sampler-only rows와 분리해 추가하는 것이다. FPP TD-error partial 630k는 계속 main table에서 제외한다.

# SABER AAAI-27 Result Matrix / Artifact Inventory

작성일: 2026-06-24
목적: GitHub remote branches와 local working tree에 남아 있는 SABER 관련 실험 결과를 논문 사용 가능성 기준으로 분류한다.

## 1. 조사 범위

이번 inventory는 local checkout만 보지 않고 GitHub remote branch를 함께 확인했다.

| Source | Commit | 역할 |
| --- | --- | --- |
| `origin/main` | `7a7992b` / `Add DrawerOpen DDPG B3 critic-only results` | FPP, FetchPush, DrawerOpen 결과가 가장 많음 |
| `origin/fetchslide-runtime-artifacts` | `2a8d34e` / `Add FetchSlide runtime experiment artifacts` | FetchSlide runtime/control 결과의 primary source |
| `origin/aaai27-core-artifacts` | `1142c97` / `Add AAAI27 paper draft` | 논문 초안과 core artifact branch |
| local `aaai27-core-artifacts` | `1142c97` + uncommitted docs | 현재 작업 중인 논문/계획 문서 |

조사는 dirty worktree를 보호하기 위해 branch checkout 없이 `/private/tmp/saber_wt_origin_main`, `/private/tmp/saber_wt_fetchslide` worktree에서 수행했다.

## 2. 결론 요약

가장 중요한 결론은 다음이다.

- FPP main table은 이미 충분하다. DDPG/TD3/SAC HER baseline과 critic-only SABER가 모두 1M, seeds 1--5, summary 기반으로 존재한다.
- FPP PER-style/PER-full baseline도 1M, seeds 1--5로 존재한다.
- FPP TD-error는 log 기반 partial 630k 재집계를 완료했다. Run directory는 1M으로 되어 있으나 실제 모든 seed가 약 633k train steps에서 종료되었고, `metrics_selected_10k.csv`의 evaluation columns는 비어 있어 `run.log`에서 evaluation curve를 복원했다. 따라서 1M main row로 쓰면 안 되고 partial diagnostic 또는 rerun 판단용으로만 쓴다.
- FPP Random-Support/Shuffled-Weight는 1k smoke만 있고, 1M main-task mechanism control로는 사용할 수 없다.
- FetchSlide는 `origin/fetchslide-runtime-artifacts`에 DDPG/TD3/SAC baseline/SABER 및 TD-error/PER-style 1M 결과가 있다.
- FetchSlide Random/Shuffled는 1M 결과가 없고 1k smoke 수준만 있다.
- FetchPush는 `origin/main`에 DDPG와 TD3 결과가 생각보다 많다. 다만 main paper가 아니라 appendix evidence로 쓰는 것이 안전하다.
- DrawerOpen은 TD3 2M controls가 풍부하지만 Random-Support가 강하므로 structured support mechanism의 primary evidence로 쓰면 안 된다.

## 3. Main Paper에 바로 쓸 수 있는 결과

### 3.1 FetchPickAndPlace primary table

Source: `origin/main`, local에도 동일 계열 artifact 존재.

| Row | Path | Steps | Seeds | 상태 | 권장 사용 |
| --- | --- | ---: | --- | --- | --- |
| DDPG HER | `experiment_FPP/runs/B0` and `experiment_FPP/results/fpp_b3_v4_ls512_1m_summary.json` | 1M | 1--5 | ready | main |
| DDPG HER+SABER | `experiment_FPP/results/fpp_b3_v4_ls512_1m_summary.json` | 1M | 1--5 | ready | main |
| TD3 HER | `experiment_FPP/runs/td3_her_baseline_ls512` | 1M | 1--5 | ready | main |
| TD3 HER+SABER | `experiment_FPP/runs/td3_saber_critic_only_v4_ls512` | 1M | 1--5 | ready | main |
| SAC HER | `experiment_FPP/runs/sac_her_baseline_ls512` | 1M | 1--5 | ready | main |
| SAC HER+SABER | `experiment_FPP/runs/sac_saber_critic_only_v4_ls512` | 1M | 1--5 | ready | main |

주의: `experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512`에는 seeds 1--10와 중간 500k run이 함께 있어 단순 recursive 평균을 내면 논문 값이 틀어진다. DDPG SABER main row는 `experiment_FPP/results/fpp_b3_v4_ls512_1m_summary.json`의 `B3_v4_512` summary를 source of truth로 둔다.

현재 paper table의 DDPG row와 일치하는 aggregate:

| Method | Final | Best | AUC | Steps/sec |
| --- | ---: | ---: | ---: | ---: |
| DDPG HER | `0.940 ± 0.042` | `1.000 ± 0.000` | `0.539 ± 0.125` | `65.02 ± 0.10` |
| DDPG HER+SABER | `0.960 ± 0.042` | `1.000 ± 0.000` | `0.609 ± 0.038` | `41.72 ± 0.20` |

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
| SAC HER | `0.000 ± 0.000` | `0.000 ± 0.000` | `0.000 ± 0.000` |

해석: DrawerOpen TD3는 Random-Support가 Critic-only와 비슷하거나 더 높다. 따라서 structured support mechanism의 주증거가 아니라 limitation/stress-test로만 써야 한다. DDPG DrawerOpen은 매우 강한 positive result지만 FPP/FetchSlide와 다른 implementation branch이므로 appendix-only가 안전하다.

## 5. 사용하면 안 되거나 추가 작업이 필요한 결과

### 5.1 FPP mechanism controls

| Row | 현재 상태 | 판단 |
| --- | --- | --- |
| FPP Random-Support | `experiment_FPP/runs/B3-critic-only-random`, 1k smoke only | main/appendix 불가, 새 1M 필요 |
| FPP Shuffled-Weight | `experiment_FPP/runs/B3-critic-only-shuffled`, 1k smoke only | main/appendix 불가, 새 1M 필요 |
| FPP Random-Fixed-Count | 발견 못함 | 새 실험 필요 |
| FPP TD-error | log 기반 partial 630k summary 생성 완료 | 1M main row 불가, rerun 또는 same-horizon diagnostic 필요 |

AAAI 리뷰 리스크를 닫으려면 FPP DDPG에서 Random-Support, Shuffled-Weight, Random-Fixed-Count는 최소 1M seeds 1--5로 새로 돌리는 것이 좋다.

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

1. FPP DDPG Random-Support 1M seeds 1--5.
2. FPP DDPG Shuffled-Weight 1M seeds 1--5.
3. FPP DDPG Random-Fixed-Count 1M seeds 1--5.
4. FPP TD-error 1M rerun 또는 same-horizon 630k diagnostic table 작성.
5. 가능하면 FPP TD3 또는 SAC 중 하나에도 Random/Shuffled control 추가.

## 7. 다음 액션

FPP TD-error raw metrics는 `run.log` 기반으로 partial 630k 재집계를 완료했다. 다만 1M result가 아니므로 main table에는 넣지 않는다. 다음 작업은 5번 `Support vs TD-error / Reliability Diagnostic 분석`으로 넘어가면서, TD-error partial result를 같은 horizon diagnostic으로 쓸지 아니면 1M rerun 계획으로 넘길지 결정하는 것이다.

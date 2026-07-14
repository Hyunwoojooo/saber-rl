# SABER AAAI-27 Result Matrix / Artifact Inventory

작성일: 2026-06-24
최근 업데이트: 2026-07-13
목적: GitHub remote branches와 local working tree에 남아 있는 SABER 관련 실험 결과를 논문 사용 가능성 기준으로 분류한다.

## 1. 조사 범위

이번 inventory는 local checkout만 보지 않고 GitHub remote branch를 함께 확인했다.

| Source | Commit | 역할 |
| --- | --- | --- |
| `origin/main` | `19233d5` current; query-ablation artifact commit `9255a2c` | FPP, FetchPush, DrawerOpen/DrawerClose, common-grid, HGR, D8, seed extensions, DrawerOpen REDQ, paired TD3 n=10, and TD3 query-ablation source |
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

- 중심 positioning은 `새 최고성능 RL algorithm`이 아니라 `post-HER critic supervision allocation을 relabeling/sampling/targets/actor/architecture/update rate와 분리한 design axis`다. Abstract와 Introduction은 design axis, architecture/replay/update-budget-preserving intervention, cross-backbone+n=10 evidence, DDPG/TD3 update-budget trade-off를 먼저 제시하고 boundary는 마지막에 둔다.
- FPP main table은 이미 충분하다. DDPG/TD3/SAC HER baseline과 critic-only SABER가 모두 1M, seeds 1--5, summary 기반으로 존재한다.
- FPP SAC의 legacy 25k main rows와 newer 10k controls는 exact 50k--1M common grid로 정식 재계산됐다. Interpolation/forward-fill 없이 20 checkpoints를 사용하며, 70개 source file hash와 7개 analyzer test가 검증됐다.
- FPP PER-style/PER-full baseline도 1M, seeds 1--5로 존재한다.
- FPP TD-error는 log 기반 partial 630k 재집계를 완료했다. Run directory는 1M으로 되어 있으나 실제 모든 seed가 약 633k train steps에서 종료되었고, `metrics_selected_10k.csv`의 evaluation columns는 비어 있어 `run.log`에서 evaluation curve를 복원했다. 따라서 1M main row로 쓰면 안 되고 partial diagnostic 또는 rerun 판단용으로만 쓴다.
- FPP DDPG Random-Support/Shuffled-Weight/Random-Fixed-Count는 1M, seeds 1--5로 완료되어 main mechanism control로 사용할 수 있다.
- FPP TD3 Random-Support와 Random-Fixed-Count k73도 1M, seeds 1--5로 완료되었다. 이 결과는 Random-Support/Random-Fixed가 HER 수준에 머물고 SABER와 분리된다는 점에서 DDPG보다 강한 mechanism evidence를 제공한다.
- FPP SAC Random-Support와 Random-Fixed-Count k76도 1M, seeds 1--5로 완료되었다. Random-Support는 HER와 거의 같은 AUC에 머물고, Random-Fixed-Count는 final success를 포화시키지만 SABER보다 낮은 AUC를 보인다.
- FPP TD3 high replay-ratio \(4\times\)도 1M, seeds 1--5로 완료되었다. AUC는 TD3 SABER와 비슷하지만 final success와 Hit90 seed count는 낮고 runtime/update budget은 더 크다.
- FPP DDPG/TD3에서 SABER는 high-RR \(4\times\)와 comparable AUC를 약 4분의 1 gradient-update budget으로 얻는다. 이는 `update-budget trade-off`이지 support lookup까지 포함한 wall-clock-efficiency claim이 아니다. TD3에서는 SABER final success와 Hit90 coverage도 더 높다.
- FPP SAC high replay-ratio \(4\times\)는 common-grid AUC 0.768로 SABER 0.583보다 높지만, recorded runtime은 약 47.0h로 SABER의 약 9.6h보다 훨씬 길다. AUC/hour는 hardware-normalized metric이 아니므로 main claim에서 제거한다.
- FPP REDQ-HER target-Q-bound \(N=10,M=2,\mathrm{UTD}=4\)는 common-grid final 1.000, AUC 0.867, median Hit90 300k로 가장 강한 FPP learning curve지만, 10-critic ensemble과 약 89.5h recorded runtime을 사용하는 architecture/compute-scaled baseline이다.
- FPP DDPG에는 original pair-priority table, episode/pair sampling, and importance correction을 포함한 `HGR (matched DDPG+HER)` 1M, seeds 1--5 row가 추가됐다. Final/AUC는 `0.540/0.338`, Hit90은 2/5이며, HGR--HER AUC delta는 `-0.201 [-0.357,-0.031]`, HGR--SABER는 `-0.271 [-0.405,-0.197]`이다. 이는 original HGR paper configuration의 exact reproduction이 아니라 현재 matched backbone에서의 algorithmically faithful direct baseline이다.
- FPP TD3 D8 action-diversity sensitivity는 main-positive samples의 98.4%를 유지하는 weak gate다. AUC는 `0.357 ± 0.045`로 main SABER `0.442 ± 0.035`보다 낮지만 HER 및 count-matched random control과의 paired intervals는 0을 포함한다. Diversity가 해롭다는 근거가 아니라 predicate sensitivity가 아직 열려 있다는 appendix 결과로 사용한다.
- FPP TD3 paired confirmatory artifact는 HER와 SABER seeds 1--10을 exact 10k--1M grid에서 비교한다. HER/SABER는 final `0.860 ± 0.191`/`0.895 ± 0.186`, AUC `0.383 ± 0.041`/`0.429 ± 0.058`, Hit90 `7/10` (median 970k)/`8/10` (median 835k)다. Paired AUC delta는 `+0.0467 [0.0085, 0.0783]`, positive seeds `8/10`, exact two-sided sign-flip `p=0.0527`, leave-one-out mean range `[0.0405, 0.0616]`이다. 이는 TD3 mean-AUC 결과를 보강하지만 every-seed improvement나 exact-test `p<0.05` claim은 허용하지 않는다.
- FPP TD3 support-query ablation은 HER/Main/Replay-query/Shuffled-policy/Context-count-only 5 methods, seeds 1--5, exact 10k--1M grid로 완료됐다. Main AUC `0.442 ± 0.035`는 Replay `0.368 ± 0.042`, Shuffled `0.347 ± 0.051`보다 높고 paired deltas는 `+0.0746 [0.0593,0.0923]`, `+0.0958 [0.0631,0.1276]`로 모든 seed에서 양수다. Context-only는 `0.396 ± 0.090`, Main-minus-context `+0.0467 [-0.0409,0.1219]`로 닫히지 않는다. 따라서 FPP TD3에서는 current-policy query identity와 sample alignment가 replay/shuffled alternatives보다 낫다는 mechanism evidence로 사용하되, context eligibility 위 action membership의 독립 기여는 주장하지 않는다.
- FetchSlide는 `origin/fetchslide-runtime-artifacts`에 DDPG/TD3/SAC baseline/SABER 및 TD-error/PER-style 1M 결과가 있다.
- FetchSlide Random/Shuffled는 1M 결과가 없고 1k smoke 수준만 있다.
- FetchPush는 `origin/main`에 DDPG와 TD3 결과가 생각보다 많다. 다만 main paper가 아니라 appendix evidence로 쓰는 것이 안전하다.
- DrawerOpen은 DDPG 2M HER/SABER/Random-Support/PER-Style/Random-Fixed/Shuffled 결과와 TD3 2M controls가 풍부하다. DDPG는 HER 대비 large-effect/seed-consistency evidence로 유용하지만 Random-Support/PER-Style/Random-Fixed도 강하므로 structured support mechanism의 primary proof로만 쓰면 안 된다.
- 기존 DrawerOpen `HGR` row는 canonical HGR의 importance-sampling correction을 끈 sampler-only ablation이다. 논문과 pipeline에서는 `HGR-style sampler-only`로만 표기하며 canonical HGR direct baseline claim에는 사용하지 않는다.
- DrawerOpen SABER seeds 6--10 독립 확장은 4/5 Hit90, final `0.790 ± 0.442`, AUC `0.550 ± 0.329`다. Seeds 1--10 pooled summary는 9/10 Hit90, final `0.895 ± 0.315`, AUC `0.652 ± 0.246`다. Matched methods는 seeds 1--5만 있으므로 pooled row는 robustness summary이며 paired method comparison에 사용하지 않는다.
- DrawerOpen의 별도 diagnostic cohort는 seeds 1--5를 진단 로깅으로 재실행하고 diagnostic seeds 6--10과 합친다. 재실행 AUC `0.755`는 원래 seeds 1--5의 `0.754`를 aggregate 수준에서 재현했다. 성공한 9개 diagnostic seed에서 support-positive/negative pre-update TD error는 `0.3204/0.2739` (`1.17x`), same-minibatch residual reduction은 `0.00843/0.00403` (`2.1x`)이며 모두 9/9 seed에서 양의 차이다. Seed 7은 수치 폭발 없이 contact exploration 실패 뒤 마지막 400k support rate가 `0.993`으로 포화한다. 동일 weighted minibatch의 in-sample 측정이고 Python scanner의 scan truncation이 약 98%이므로 causal/generalization 또는 full-replay-equivalence 증거로 쓰지 않는다. 이 diagnostic cohort와 원래 pooled performance cohort를 합치지 않는다.
- DrawerOpen REDQ-HER TQB는 2M steps, seeds 1--5에서 final `1.000 ± 0.000`, AUC `0.672 ± 0.314`, Hit90 `5/5` (median 370k)다. REDQ는 SABER보다 Hit90이 빠르고 paired seed AUC 3/5에서 우세하다. SABER는 mean AUC와 stability에서 우세하지만 SABER--REDQ AUC `+0.082 [-0.145, 0.360]`은 닫히지 않는다. `N=10, M=2, UTD=4`, mean runtime 111.05h/seed의 architecture/compute-scaled reference로 사용한다.
- Exact paired robustness 분석을 추가했다. FPP SABER--HER AUC의 original positive seed 수와 exact two-sided sign-flip 값은 DDPG `3/5, p=0.375`, TD3 `4/5, p=0.125`, SAC `5/5, p=0.0625`다. Leave-one-seed-out mean delta는 세 backbone 모두 양수이며, DrawerOpen SABER--HER도 `5/5, p=0.0625`와 positive leave-one-out range를 보인다. TD3 n=10 extension은 `8/10, p=0.0527`와 positive leave-one-out range를 보인다. Bootstrap CI와 exact sign-flip은 서로 다른 finite-seed diagnostics로 보고하며 significance claim으로 합치지 않는다.
- DrawerClose TD3 2M HER/SABER/Random-Support/Random-Fixed/Shuffled/PER-Style results도 `origin/main`에 있다. Shuffled-Weight가 SABER를 match하므로 appendix/stress-test와 mixed-mechanism limitation evidence로 쓰는 것이 안전하다.

## 3. Main Paper에 바로 쓸 수 있는 결과

### 3.1 FetchPickAndPlace primary table

Source: `origin/main`, local에도 동일 계열 artifact 존재.

| Row | Path | Steps | Seeds | 상태 | 권장 사용 |
| --- | --- | ---: | --- | --- | --- |
| DDPG HER | `experiment_FPP/runs/B0` and `experiment_FPP/results/fpp_b3_v4_ls512_1m_summary.json` | 1M | 1--5 | ready | main |
| DDPG HER+SABER | `experiment_FPP/results/fpp_b3_v4_ls512_1m_summary.json` | 1M | 1--5 | ready | main |
| DDPG HGR (matched DDPG+HER) | `experiment_FPP/paper_artifacts/fpp_ddpg_hgr_pair_priority_is_1m` | 1M | 1--5 | verified artifact | strong goal-prioritization baseline |
| TD3 HER | `experiment_FPP/runs/td3_her_baseline_ls512` | 1M | 1--5 | ready | main |
| TD3 HER vs SABER confirmatory | `experiment_FPP/paper_artifacts/fpp_td3_her_vs_saber_n10_1m` | 1M | 1--10 | verified paired artifact | confirmatory AUC evidence; original n=5 rows remain cross-backbone source |
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
| DDPG HGR (matched DDPG+HER) | `0.540 ± 0.365` | `0.810 ± 0.188` | `0.338 ± 0.149` | `16.93 ± 0.06` |

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

FPP TD3 D8 action-diversity sensitivity:

| Method | Final | AUC | Hit90 | 해석 |
| --- | ---: | ---: | ---: | --- |
| TD3 HER | `0.840 ± 0.188` | `0.372 ± 0.032` | `980k (3/5)` | reference |
| TD3 SABER | `0.970 ± 0.045` | `0.442 ± 0.035` | `840k (5/5)` | main predicate |
| TD3 SABER-D8 | `0.800 ± 0.184` | `0.357 ± 0.045` | `990k (3/5)` | D8 minus main `-0.085 [-0.112,-0.059]` |
| Random-Diversity-Count | `0.790 ± 0.266` | `0.361 ± 0.052` | `920k (3/5)` | D8 minus control `-0.004 [-0.051,0.065]` |

Gate activity source: `experiment_FPP/paper_artifacts/fpp_td3_saber_action_diversity_d8/`. D8 pass given main is `0.984199`; threshold target interval was not reached and the preregistered fallback selected D8.

FPP TD3 support-query ablation source of truth:

| Query / control | Final | AUC | Hit90 | Main-minus-row AUC | Support-positive rate | 해석 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TD3 HER | `0.840 ± 0.188` | `0.372 ± 0.032` | `980k (3/5)` | `+0.071 [0.032,0.095]` | `0.000` | baseline |
| Main current-policy | `0.970 ± 0.045` | `0.442 ± 0.035` | `840k (5/5)` | reference | `0.219` | main predicate |
| Replay-query | `0.660 ± 0.219` | `0.368 ± 0.042` | `920k (1/5)` | `+0.0746 [0.0593,0.0923]` | `0.426` | sampled transition remains in pool; self-support limitation |
| Shuffled-policy | `0.550 ± 0.226` | `0.347 ± 0.051` | `890k (1/5)` | `+0.0958 [0.0631,0.1276]` | `0.154` | policy-action distribution preserved, sample alignment removed |
| Context-count-only | `0.800 ± 0.242` | `0.396 ± 0.090` | `850k (3/5)` | `+0.0467 [-0.0409,0.1219]` | `0.956` | not count matched; action-membership increment unresolved |

Source: `experiment_FPP/paper_artifacts/fpp_td3_saber_query_ablation_ls512/` at `origin/main`. Exact sign-flip values are `0.0625/0.0625/0.3125` for Main minus Replay/Shuffled/Context-only. The first two are the minimum attainable two-sided value for five nonzero pairs and are reported as finite-seed consistency checks, not separate significance claims.

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
| DDPG | Critic-only SABER, paired cohort | `metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Critic-Only-2M` | 2M | 1--5 | ready | large-effect paired row |
| DDPG | Critic-only SABER, independent extension | `metaworld_Drawer_Open_DDPG/paper_artifacts/drawer_open_ddpg_saber_seed_extension_2m` | 2M | 6--10 | verified artifact | unmatched robustness/diagnostic only |
| DDPG | HGR-style sampler-only | `metaworld_Drawer_Open_DDPG_HGR/paper_artifacts/drawer_open_ddpg_hgr_sampler_only_2m` | 2M | 1--5 | verified artifact | sampling control only; not canonical HGR |
| DDPG | Random-Support | `metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Random-Support-2M` | 2M | 1--5 | ready | control/limitation |
| DDPG | PER-Style | `metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-PER-Style-2M` | 2M | 1--5 | ready | control/limitation |
| DDPG | Random-Fixed-Count | `metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Random-Fixed-Count-2M` | 2M | 1--5 | ready | control/limitation |
| DDPG | Shuffled-Weight | `metaworld_Drawer_Open_DDPG/results_2m/DDPG-HER-B3-Shuffled-Weight-2M` | 2M | 1--5 | ready | control/limitation |
| REDQ-SAC | REDQ-HER TQB | `metaworld_Drawer_Open_REDQ/paper_artifacts/drawer_open_redq_her_tqb_2m` | 2M | 1--5 | verified artifact | architecture/compute-scaled reference |
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
| DDPG Critic-only SABER, extension 6--10 | `0.790 ± 0.442` | `0.800 ± 0.447` | `0.550 ± 0.329` |
| DDPG Critic-only SABER, pooled 1--10 | `0.895 ± 0.315` | `0.900 ± 0.316` | `0.652 ± 0.246` |
| DDPG HGR-style sampler-only | `0.400 ± 0.548` | `0.400 ± 0.548` | `0.311 ± 0.425` |
| DDPG Random-Support | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.706 ± 0.122` |
| DDPG PER-Style | `0.800 ± 0.447` | `1.000 ± 0.000` | `0.715 ± 0.090` |
| DDPG Random-Fixed-Count | `0.800 ± 0.447` | `0.800 ± 0.447` | `0.621 ± 0.352` |
| DDPG Shuffled-Weight | `0.400 ± 0.548` | `0.420 ± 0.531` | `0.307 ± 0.420` |
| REDQ-HER TQB (N=10, M=2, UTD=4) | `1.000 ± 0.000` | `1.000 ± 0.000` | `0.672 ± 0.314` |
| SAC HER | `0.000 ± 0.000` | `0.000 ± 0.000` | `0.000 ± 0.000` |

해석: DrawerOpen TD3는 Random-Support가 Critic-only와 비슷하거나 더 높다. 따라서 structured support mechanism의 주증거가 아니라 limitation/stress-test로만 써야 한다. DDPG DrawerOpen은 매우 강한 positive result지만 Random-Support/PER-Style/Random-Fixed도 강하므로 HER 대비 large-effect evidence로 쓰고 causal mechanism proof로 쓰면 안 된다. DDPG Shuffled-Weight의 corrected Hit90은 `460k (2/5)`, observed throughput은 `22.95 ± 1.42 steps/s`다. 이전 `1735k`와 `141.62 ± 53.69`는 interrupted/resumed run의 resume-only fields를 사용한 오류였다. Additional seeds는 한 complete failure를 포함하므로 기존 5/5를 과장하지 않게 만들지만 pooled 9/10 Hit90으로 broader seed robustness를 보강한다. REDQ-HER TQB도 5/5 Hit90과 더 이른 median 370k를 달성하지만 seed AUC 범위가 `0.196--0.936`으로 넓고, ten-critic/UTD-4 budget을 사용한다. SABER mean AUC가 더 높아도 paired interval이 0을 포함하므로 이 row는 compute/backbone-scaling boundary로만 사용한다.

### 4.3 DrawerClose

Source: `origin/main`.

DrawerClose DDPG에는 compact artifact 기반 HGR-style sampler-only row가 추가됐다. 이 row는 episode/future-goal TD-error sampling만 바꾸고 critic/actor loss weights를 모두 1로 유지하며 canonical HGR importance correction을 사용하지 않는다.

| Family | Row | Path | Steps | Seeds | 상태 | 권장 사용 |
| --- | --- | --- | ---: | --- | --- | --- |
| DDPG | HGR-style sampler-only | `metaworld_Drawer_Close_DDPG_HGR/paper_artifacts/drawer_close_ddpg_her_hgr_2m` | 2M | 1--5 | ready ablation | appendix stress-test; not canonical HGR |

DrawerClose DDPG의 기존 HER/SABER와 descriptive comparison은 다음과 같다.

| Row | Final | Best | AUC | Hit90 | 해석 |
| --- | ---: | ---: | ---: | ---: | --- |
| DDPG HER | `0.160 ± 0.182` | `1.000 ± 0.000` | `0.251 ± 0.030` | 70k (5/5) | late collapse |
| DDPG SABER | `0.280 ± 0.251` | `1.000 ± 0.000` | `0.324 ± 0.082` | 20k (5/5) | higher AUC, earliest Hit90 |
| DDPG HGR-style sampler-only | `0.530 ± 0.390` | `1.000 ± 0.000` | `0.262 ± 0.054` | 590k (5/5) | higher final, HER-like AUC, much later Hit90 |

해석: sampler-only row는 mean final stability가 높지만 AUC는 HER와 가깝고 SABER보다 낮다. 이는 late stability와 early sample efficiency의 descriptive tradeoff이며 canonical HGR comparison이나 SABER superiority claim에 사용할 수 없다.

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

### 4.4 FPP TD3 runtime profile

Matched seed-1 TD3+HER와 TD3+HER+SABER를 같은 source snapshot과 NVIDIA GB10 stack에서 100k steps까지 실행하고, `(20k,100k]` 구간을 synchronized profile했다.

| Metric | TD3+HER | TD3+HER+SABER |
| --- | ---: | ---: |
| Training-loop ms/env-step | `8.063` | `24.133` |
| Steps/sec excluding eval | `124.02` | `41.44` |
| Training seconds in window | `645.0` | `1930.7` |

SABER의 상대 training-time overhead는 `199.3%`, throughput 감소는 `66.6%`다. Support/weighting은 `15.028 ms/step`으로 SABER loop의 `62.3%`와 measured loop-time 증가분의 `93.5%`를 차지한다. 그중 local replay lookup이 `12.651 ms/step`, support block의 `84.2%`다. 이 row는 hardware-specific bottleneck diagnosis이며 1M runtime extrapolation이나 wall-clock superiority claim에 사용하지 않는다.

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
2. Matched HGR와 high-RR/REDQ를 포함한 compact strong-baseline boundary.
3. Actor--critic input과 support-only context를 분리한 support-construction 정의와 핵심 lookup 설정.
4. FPP TD3 query ablation: Main versus Replay/Shuffled closure와 Context-only uncertainty.
5. DrawerOpen DDPG paired large-effect row, Random-support/PER-style rows, scaled REDQ-HER TQB row, unmatched 9/10 seed-extension 한 문장.
6. FetchSlide DDPG failure 및 DrawerOpen TD3를 limitation으로 짧게 유지.

### Appendix

1. FPP additional variants: v4 ls1024, v6 anneal, v7/v8 loss variants, TD3 D8 sensitivity.
2. FPP TD3 paired n=10 confirmatory extension; original n=5 cross-backbone table과 분리해 aggregate, paired CI, exact sign-flip을 보고.
3. FetchSlide TD-error/PER-style and v4 ls512 variants.
4. FetchPush DDPG subset and TD3 ablation/control subset.
5. DrawerOpen DDPG/SAC/TD3 controls와 SABER seeds 6--10 diagnostics.

### 새 실험 필요

우선순위는 다음이다.

1. DrawerOpen REDQ-HER TQB audit와 반영은 완료했다. 진행 중인 DrawerClose REDQ가 끝나면 같은 config/evaluation-grid fidelity audit를 수행한다.
2. FPP TD3 current-policy-query ablation은 완료했다. 다음 mechanism priority는 phase-free, horizon-free, or generic distance-only context ablation으로 task-specific context 의존성을 직접 검사하는 것이다.
3. Blind code/data archive를 묶고 clean-environment smoke test와 table-regeneration test를 수행한다. OpenReview paper ID가 배정되면 첫 페이지에 추가한다.
4. D8보다 더 selective한 diversity threshold는 context ablation보다 우선하지 않는다.
5. FPP TD-error 1M rerun은 secondary diagnostic이므로 context ablation과 artifact 검증 뒤에 결정한다.
6. 최종 원고에서는 matched HGR, DDPG/TD3/SAC controls, scaling references를 main/appendix 배치에 맞춰 압축하고 두 PDF를 교차검증한다.

## 7. 다음 액션

FPP SAC common-grid, matched HGR, TD3 D8, FPP TD3 paired n=10 confirmatory, FPP TD3 support-query ablation, DrawerOpen HGR sampler-only, DrawerOpen SABER seed extension, and DrawerOpen REDQ-HER TQB artifacts는 inventory와 두 manuscript source에 연결됐다. Primary 원고에는 actor--critic input과 support-only context의 분리, compact support construction, 핵심 lookup 설정, query-identity closure, DrawerOpen Random-support/PER-style/scaled-REDQ rows, bounded conclusion이 반영됐다. 다음 scientific update는 task-specific context ablation과 completion-audited DrawerClose REDQ를 서로 다른 역할로 추가하는 것이다. FPP TD-error partial 630k는 계속 main table에서 제외한다.

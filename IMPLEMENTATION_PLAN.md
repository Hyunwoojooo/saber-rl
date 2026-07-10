# SABER AAAI-27 Implementation Plan

작성일: 2026-06-24
목표: SABER AAAI-27 논문 초안을 top-tier review 기준에서 방어 가능한 제출 원고로 보강한다.

## 0. 현재 판단

현재 원고는 아이디어와 framing은 괜찮지만, 리뷰 기준으로는 아직 위험하다. 특히 다음 문제가 크다.

- C3: 실험 프로토콜과 세팅 정의 부족
- C8: 재현성 appendix 부족
- C11: support-conditioned mechanism closure 부족
- Method gap: current policy action support를 측정하지만 critic loss는 replay action에서 계산됨
- Random-Support: structured support가 핵심 원인이라는 claim을 약화함

작업의 핵심은 논문을 더 길게 쓰는 것이 아니라, reviewer가 reject할 지점을 하나씩 닫는 것이다.

## 1. Method 정의 보강

상태: 3차 완료. `AAAI27_METHOD_FIXES_KR.md`를 만들고, `paper/saber_aaai27_draft_kr.tex`의 Method/Preliminaries에 replay action vs policy action 구분, FPP exact context, support predicate, TD3/SAC target, PER-style baseline 수식을 반영했다. 이후 `saber_aaai27_review_error_check_full.md`의 지적을 반영해 `M_Q_min`을 Q-value/diversity threshold가 아니라 capped local transition count threshold로 정리하고, 논문 표기는 \(M_{\mathrm{avail}}\)로 바꿨다. 또한 `r_i`, bootstrapping terminal \(d_i\), timeout convention, TD-error baseline의 `stopgrad`/no-gradient weight 계산을 명시했다. 추가로 FPP, FetchSlide, DrawerOpen의 task-specific context features, phase predicates, bin boundaries, action binning, support hyperparameters를 본문 표와 reproducibility appendix에 추가했다.

### 목적

SABER의 technical definition을 재현 가능하고 논리적으로 닫힌 형태로 만든다.

### 해야 할 일

- current policy action support를 보지만 critic loss는 replay action에서 계산된다는 점을 명확히 설명한다.
- 이를 `policy-relevant local-support proxy`로 정의한다.
- `A_i^sup`, `B_i`, `M_B`, `M_{\mathrm{avail}}`/`M_Q_min`, `N_gap_min`을 정의한다.
- task별 local context와 binning을 정확히 표로 정리한다.
- TD3 Bellman target 수식을 추가한다.
- SAC Bellman target 수식을 추가한다.
- PER-style baseline 수식을 추가한다.

### 산출물

- `AAAI27_METHOD_FIXES_KR.md`
- 이후 `paper/saber_aaai27_draft_kr.tex` Method section 수정

### 완료 기준

- 논문 수식만 보고 SABER support predicate와 critic objective를 구현할 수 있다.
- DDPG/TD3/SAC 각각에서 SABER weight가 어디에 적용되는지 명확하다.
- policy-action support와 replay-action critic loss 사이의 proxy assumption이 명시되어 있다.

## 2. Metric 및 Table 오류 정리

상태: 3차 완료. `AAAI27_METRIC_DEFINITIONS_KR.md`를 만들고, `paper/saber_aaai27_draft_kr.tex`의 metric 정의, FPP Hit90 표기, FetchSlide method naming, DrawerOpen Hit90/AUC/method naming을 정리했다. 이후 success curve notation, final/best 수식, normalized trapezoidal AUC 식, last-quarter AUC denominator, Hit90 first-hit/unreached 처리, seeds 1--5 sample standard deviation, steps/sec 해석을 본문과 `AAAI27_REPRODUCIBILITY_APPENDIX_KR.md`에 반영했다. Round 3 error check에서 지적된 FetchSlide Table과 result pipeline 불일치는 `analysis/result_pipeline/output/aggregate_metrics.csv`를 source-of-truth로 고정해 원고 Table과 해석 문장을 재생성했다.

### 목적

결과 표의 신뢰도를 회복하고 metric 정의 혼동을 없앤다.

### 해야 할 일

- `success AUC` 계산 방식을 정의한다.
- `Last-quarter AUC` 계산 방식을 정의한다.
- `First Hit >= 0.90` 정의를 재확인한다.
- `Hit90` 정의를 통일한다.
- `mean +/- std`인지 `stderr`인지 명시한다.
- Table 1의 TD3 HER `Best Success 0.880` vs `First Hit 980k` 모순 여부를 재검산한다.
- Table 4/5의 DrawerOpen AUC `0.362` vs `0.361`을 통일한다.
- `HER+SABER`, `SABER`, `B3-Critic-Only` 표기를 통일한다.
- `B3` prefix를 제거하거나 정의한다.

### 산출물

- `AAAI27_METRIC_DEFINITIONS_KR.md`
- 수정된 result table draft

### 완료 기준

- 모든 표의 metric 정의가 본문 또는 caption에서 추적 가능하다.
- 동일 결과가 서로 다른 표에서 다른 숫자로 등장하지 않는다.
- method naming이 table 전체에서 일관된다.

## 3. Reproducibility Appendix 작성

상태: 2차 완료. `AAAI27_REPRODUCIBILITY_APPENDIX_KR.md`를 만들고, FPP/FetchSlide/DrawerOpen의 source-of-truth 파일, 공통 HER/training protocol, environment/reward convention, SABER support lookup, task-specific context, metric/runtime 정의, 남은 artifact gap을 정리했다. 이후 `paper/saber_aaai27_draft_kr.tex`에서 로컬 md 파일명 참조를 제거하고, `Appendix A: Reproducibility Details`로 실제 LaTeX appendix를 통합했다. Appendix A에는 environment version/threshold, reward convention, seed mapping, eval episodes/frequency, model/optimizer settings, SABER support settings, runtime/hardware metadata limitation을 포함했다.

### 목적

논문만 보고 실험 세팅과 SABER 구현을 재구성할 수 있게 한다.

### 해야 할 일

- env name/version을 정리한다.
- reward convention을 정리한다.
- success threshold를 정리한다.
- train steps를 정리한다.
- seed list를 정리한다.
- eval episodes와 evaluation frequency를 정리한다.
- network architecture를 정리한다.
- optimizer/lr을 정리한다.
- batch size를 정리한다.
- replay buffer size를 정리한다.
- HER ratio/strategy를 정리한다.
- update frequency를 정리한다.
- target network update rule을 정리한다.
- DDPG/TD3/SAC hyperparameters를 정리한다.
- hardware / GPU / CPU 정보를 정리한다.
- steps/sec 측정 기준을 정리한다.
- SABER hyperparameters를 정리한다.
- task별 support context feature를 정리한다.
- bin width / action binning을 정리한다.
- lookup size를 정리한다.
- support predicate pseudocode를 작성한다.

### 산출물

- `AAAI27_REPRODUCIBILITY_APPENDIX_KR.md`

### 완료 기준

- reviewer가 환경, model, training loop, HER, SABER support lookup을 재현할 수 있다.
- task-specific context가 "can include"가 아니라 exact definition으로 제시된다.
- runtime claim을 해석할 수 있는 hardware 및 측정 조건이 명시된다.

## 4. 기존 Artifact Inventory 재정리

상태: 1차 완료. `AAAI27_RESULT_MATRIX_KR.md`를 만들고, local checkout뿐 아니라 GitHub remote의 `origin/main`, `origin/fetchslide-runtime-artifacts`, `origin/aaai27-core-artifacts`를 모두 확인했다. FPP main rows, FetchSlide rows, FetchPush appendix candidates, DrawerOpen controls, 부족한 FPP/FetchSlide random/shuffled controls를 분리했다. 추가로 FPP TD-error raw artifact는 `run.log` 기반 partial 630k 재집계를 완료했으며, 1M main row가 아니라 diagnostic/rerun 판단용으로 분류했다.

### 목적

새 실험을 돌리기 전에 이미 repo에 있는 결과를 최대한 회수한다.

### 찾아야 할 결과

- FPP Random-Support
- FPP Shuffled-Weight
- FPP Random-Fixed-Count
- FPP TD-error
- FPP PER-style
- FPP PER-full
- FetchSlide random/shuffled/PER 결과
- FetchPush complete row
- DrawerOpen control 결과 원본 metrics

### 확인할 항목

- seeds
- train steps
- environment
- backbone
- result path
- source code path
- metric file 존재 여부
- `metrics.csv`, `summary.json`, `run.log` 중 어떤 파일을 기준으로 할지
- main paper에 넣을 수 있는 row인지 appendix-only row인지

### 산출물

- `AAAI27_RESULT_MATRIX_KR.md`
- `experiment_FPP/results/fpp_td_error_partial_630k_analysis.md`
- `experiment_FPP/results/fpp_td_error_partial_630k_summary.json`
- `experiment_FPP/results/fpp_td_error_partial_630k_per_seed.csv`
- `experiment_FPP/results/fpp_td_error_partial_630k_method_summary.csv`

### 완료 기준

- 기존 artifact만으로 만들 수 있는 main result table과 appendix table이 분리된다.
- FPP main task에서도 structured SABER가 random/shuffled보다 낫다는 증거가 있는지 판단할 수 있다.
- 새 실험이 필요한 row와 이미 충분한 row가 분리된다.
- TD-error는 partial 630k로만 존재한다는 점이 명확히 표시된다.

## 5. Support vs TD-error / Reliability Diagnostic 분석

상태: FPP DDPG instrumented diagnostic 1M seeds 1--5 완료 및 원고 반영. 기존 artifact 기반 1차 진단은 `analysis/support_td_error/analyze_existing_fpp_support_td_error.py`로 정리했고, 이후 FPP trainer에 optional sample-level diagnostic logger를 추가했다. `metacog-rl@v0-lite` commit `a306605`의 `experiment_FPP/runs/fpp_ddpg_saber_diagnostic_ls512_1m/` 결과를 로컬 작업트리에 복사했으며, 500 full-batch diagnostic checkpoints와 64k logged sample rows를 확인했다. 결과는 `AAAI27_SUPPORT_DIAGNOSTICS_KR.md`와 `paper/saber_aaai27_draft_kr.tex`의 `FPP DDPG Sample-Level Diagnostic` 절에 반영했다. 핵심 결과는 \(\mathrm{corr}(\psi_i,|\delta_i|)\approx 0.12\), top-10% high-TD-error support overlap 0.516, support-positive raw residual reduction 0.019 vs support-negative 0.008이다. 이는 PER-style equivalence를 약화하지만 pure reliability proof는 아니므로, 남은 과제는 TD3/SAC 또는 DrawerOpen으로 diagnostic을 확장하는 것이다.

### 목적

SABER가 PER/TD-error weighting의 변형이 아니라는 점과 support가 reliability signal이라는 점을 직접 분석한다.

### 분석 항목

- `corr(psi_i, |delta_i|)`
- support-positive sample과 top-k TD-error sample overlap
- support-positive vs support-negative TD-error distribution
- update 전후 Bellman residual 감소량
- support-positive vs support-negative target variance
- current policy action 근방 Q-error 또는 calibration
- support coverage over training
- weight ESS / support rate curve

### 산출물

- `analysis/support_td_error/`
- `analysis/support_td_error/analyze_existing_fpp_support_td_error.py`
- `analysis/support_td_error/fpp_support_td_error_existing_report.md`
- `analysis/support_td_error/fpp_support_td_error_existing_summary.csv`
- `analysis/support_td_error/fpp_support_td_error_existing_per_seed.csv`
- `analysis/support_td_error/fpp_support_td_error_existing_summary.json`
- `analysis/support_td_error/analyze_sample_diagnostics.py`
- `AAAI27_SUPPORT_DIAGNOSTICS_KR.md`
- `AAAI27_DIAGNOSTIC_RUN_PLAN_KR.md`
- `experiment_FPP/runs/fpp_ddpg_saber_diagnostic_ls512_1m/`
- FPP trainer diagnostic logger:
  - `experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/train_fetch_pick_and_place_lite_v0.py`
  - `experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/metacog_lite_v0/trainer.py`
  - `experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/metacog_lite_v0/agent.py`
  - `experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/metacog_lite_v0/replay.py`
  - `experiment_FPP/b3_critic_only_runtime_opt_v4_ls512/code/src/metacog_lite_v0/config.py`
- figure 후보:
  - support vs TD-error scatter 또는 bin plot
  - top-k overlap bar plot
  - residual reduction by support group
  - support coverage over training curve

### 완료 기준

- 1차 완료: 기존 artifact만으로 가능한 checkpoint-level sanity check를 끝낸다.
- 준비 완료: sample-level `psi_i`, fixed-target `|delta_i|` before/after, support count, weight, action bin, HER relabel mask를 남기는 optional logger를 구현한다.
- 준비 완료: `diagnostic_samples.csv`를 읽어 corr, top-k overlap, residual reduction, target variance를 집계하는 분석 스크립트를 작성한다.
- 완료: smoke/full diagnostic run으로 logger output을 확인했다.
- 완료: full diagnostic run 결과로 support-positive가 단순 high TD-error sample과 동일하지 않음을 sample-level로 보였다.
- 완료: support-positive와 support-negative 사이의 critic fitting behavior 차이를 제시했다.
- 완료: reliability interpretation을 theorem이 아니라 empirical diagnostic으로 낮춰 원고에 반영했다.
- 남음: 같은 diagnostic을 FPP TD3/SAC 또는 DrawerOpen으로 확장해 backbone/task dependence를 확인한다.

## 6. Random-Support 문제 닫기

상태: FPP DDPG Shuffled-Weight, Random-Support, Random-Fixed-Count k94 1M seeds 1--5 결과를 모두 반영했다. `experiment_FPP/runs/b3_support_shuffled_critic_only_ls512/`, `experiment_FPP/runs/b3_support_random_critic_only_ls512/`, `experiment_FPP/runs/b3_random_fixed_count_k94_critic_only_ls512/` artifact를 result pipeline에 연결했고, `paper/saber_aaai27_draft_kr.tex`에는 FPP DDPG control table과 updated learning curve를 반영했다. Random-Fixed-Count는 AUC가 HER와 거의 같아 fixed-count random weighting만으로는 SABER의 early/mid gain을 설명하지 못한다. 반면 Random-Support는 SABER와 거의 같은 AUC를 보여 exact sample-level support alignment가 유일한 원인이라는 claim은 배제한다. 현재 claim은 "support-conditioned critic reweighting + generic reweighting regularization"으로 낮춘다.

### 목적

structured support가 아니라 random reweighting regularization 때문이라는 반론을 줄인다.

### 해야 할 일

- 완료: FPP에서 Random-Support 결과를 반영한다.
- 완료: FPP에서 Shuffled-Weight 결과를 반영한다.
- 완료: FPP에서 Random-Fixed-Count 결과를 반영한다.
- FetchSlide에서 random/shuffled/fixed-count 결과가 있는지 확인한다.
- 없으면 최소 새 실험 목록을 작성한다.

### 최소 새 실험 후보

- FPP DDPG: SABER vs Random-Support vs Shuffled-Weight vs Random-Fixed-Count
- 가능하면 FPP TD3 또는 FPP SAC 중 하나 추가
- FetchSlide는 DDPG failure 분석용으로 최소화

### 산출물

- `AAAI27_RANDOM_CONTROL_PLAN_KR.md`
- `experiment_FPP/shared/run_experiment.py` 경로 fallback 수정
- 가능한 경우 updated mechanism table

### 완료 기준

- DrawerOpen에서는 random이 강하더라도, main FPP에서는 structured support가 random/shuffled/fixed-count와 구분된다는 근거를 확보한다.
- 구분되지 않는다면 claim을 더 낮춘다.

## 7. Related Work 보강

상태: 2차 완료. `AAAI27_RELATED_WORK_NOTES_KR.md`를 만들고, HER relabeling, HGG/CHER, EBP/HGR, MHER, PER/TD-error weighting, REDQ+HER/high replay-ratio, offline RL support constraint를 `intervention point` 축으로 비교했다. 핵심 framing은 SABER를 새로운 HER relabeling/prioritization method가 아니라, HER 이후 critic Bellman residual에 적용되는 support-conditioned loss reweighting objective로 두는 것이다. 이후 `paper/saber_aaai27_draft_kr.tex` Related Work에 `Hindsight Goal Selection and Prioritization` 절, intervention-point 비교표, REDQ+HER/high replay-ratio 논의를 추가했고, `paper/saber_refs.bib`에 HGG, CHER, EBP, HGR, MHER, REDQ, sparse-goal REDQ references를 추가했다. CHER metadata는 MHER arXiv source의 `main.bbl`에서 확인했다.

### 목적

closest prior coverage 부족과 missing strong baseline 공격을 줄인다.

### 추가할 축

- CHER
- Hindsight Goal Ranking / HGR
- Energy-Based Hindsight Experience Prioritization
- MHER / MHER(lambda)
- REDQ+HER / high replay ratio HER
- 최신 sparse goal-conditioned robotics baseline
- HER prioritization vs TD-error prioritization vs support-conditioned critic loss weighting 비교

### 산출물

- `AAAI27_RELATED_WORK_NOTES_KR.md`
- related work comparison table

### 완료 기준

- 완료: SABER가 HER relabeling, HER prioritization, PER, offline RL support constraint와 어떻게 다른지 formal axis로 정리된다.
- 완료: missing baseline에 대한 limitation 또는 추가 실험 계획이 명시된다.
- 완료: 원고 `Related Work` 본문에 압축 반영했다.
- 완료: CHER metadata와 BibTeX를 MHER arXiv source 기준으로 확인했다.

## 8. Claim 축소 및 Abstract/Introduction 재작성

상태: 2차 완료. `AAAI27_CLAIM_REVISION_KR.md`를 만들고, `paper/saber_aaai27_draft_kr.tex`의 title, abstract, introduction, contribution list, limitations, conclusion을 현재 evidence 범위에 맞게 수정했다. 이후 round-3 review와 FPP Shuffled-Weight 결과를 반영해 FPP claim을 "mean final success / mean AUC가 높다"로 낮추고, paired CI가 0을 포함하는 row를 명시했다. 핵심 claim은 "SABER is a critic-side Bellman error reweighting objective that can improve environment-step sample efficiency in support-aligned sparse HER actor-critic settings"로 좁혔다. Universal improvement, HER replacement, general-purpose RL, wall-clock efficiency, structured support unique cause claim은 제거하거나 caveat로 낮췄다.

### 목적

현재 evidence가 지지하는 범위 안에서만 주장한다.

### 버릴 claim

- structured support가 유일한 원인이다.
- 모든 backbone/task에서 일반적으로 개선된다.
- SABER가 extrapolation error를 해결한다.
- SABER는 새로운 HER method다.
- SABER는 general-purpose RL algorithm이다.

### 안전한 핵심 claim

> SABER is a critic-side Bellman error reweighting objective that can improve environment-step sample efficiency in support-aligned sparse HER settings.

### 반드시 넣을 caveat

- Random-Support 결과는 reweighting-induced regularization 가능성을 시사한다.
- SABER는 sample-efficient할 수 있지만 wall-clock efficient하지 않을 수 있다.
- support predicate는 task-specific context design에 의존한다.
- FetchSlide DDPG는 failure case다.

### 산출물

- `AAAI27_CLAIM_REVISION_KR.md`
- revised title
- revised abstract
- revised introduction
- revised contribution list
- revised limitations
- revised conclusion

### 완료 기준

- 완료: Abstract와 Conclusion이 Random-Support 및 FPP Shuffled-Weight 결과와 충돌하지 않는다.
- 완료: main claim이 FPP/FetchSlide/DrawerOpen evidence 범위 안에 있다.
- 남음: #7 related work framing을 원고 Related Work에 압축 반영한다.
- 완료: FPP DDPG Random-Support/Random-Fixed-Count, FPP TD3 Random-Support/Random-Fixed-Count, FPP SAC Random-Support/Random-Fixed-Count, FPP TD3/SAC high-RR \(4\times\), DrawerOpen DDPG Random-Support/PER-Style/Random-Fixed-Count/Shuffled-Weight, DrawerClose TD3 controls를 Abstract/Conclusion caveat에 반영했다.
- 완료: FPP DDPG sample-level diagnostic 결과를 반영해 support-vs-TD-error 및 reliability wording을 업데이트했다.

## 9. Figure/Table 재구성

상태: 통계/그림 파이프라인 9차 완료. `analysis/result_pipeline/`에 FPP DDPG Shuffled-Weight, Random-Support, Random-Fixed-Count k94, FPP TD3 Random-Support/Random-Fixed-Count k73, FPP TD3 high-RR \(4\times\), FPP SAC Random-Support/Random-Fixed-Count k76, FPP SAC high-RR \(4\times\), DrawerOpen DDPG HER/SABER/Random-Support/PER-Style/Random-Fixed-Count/Shuffled-Weight, DrawerClose TD3 HER/SABER/Random-Support/Random-Fixed-Count rows를 추가했다. 현재 입력 기준 per-seed table 200 rows, aggregate table 40 rows, paired seed-difference 1140 rows, learning curve points 125297개를 생성했다. `/opt/homebrew/bin/python3.11`의 Pillow fallback으로 22개 figure PDF/PNG를 재생성했고, 최신 PDF가 이 figure들을 정상 include한다. 새 실험 결과가 나오면 `analysis/result_pipeline/config.json`의 `rows`/`pairs`에 summary glob만 추가해서 같은 table/figure를 다시 만들 수 있다.

### 목적

표만 있는 draft에서 diagnostic evidence가 보이는 paper로 전환한다.

### Main paper 후보

- Figure 1: SABER conceptual diagram
- Table 1: FPP DDPG/TD3/SAC main result
- Figure 2: FPP learning curves
- Table 2: mechanism ablation
- Figure 3: support vs TD-error diagnostic
- Table 3: FetchSlide positive/negative result
- Table 4 또는 appendix: DrawerOpen stress-test

### 해야 할 일

- Table 3-5의 float 공백 문제를 줄인다.
- DrawerOpen control은 main text에서 compact하게 정리하거나 appendix로 이동한다.
- 완료: 기존 결과와 FPP DDPG Shuffled-Weight 결과 기반 per-seed table, paired seed difference, bootstrap CI, learning curve aggregate CSV를 생성한다.
- 완료: 기존 결과 기반 learning curve figure를 자동 생성한다.
- 완료: bootstrap resampling seed를 고정해 같은 artifact에서는 같은 CI가 나오게 한다.
- 완료: FPP DDPG Shuffled-Weight row/pair를 config에 추가하고 main draft에 반영했다.
- 완료: FPP Random-Support/Random-Fixed-Count 결과를 config row/pair와 main draft에 반영했다.
- 완료: DrawerOpen DDPG Shuffled-Weight 2M artifact를 `origin/main`에서 가져오고, config row/pair, aggregate table, learning curve, main draft DrawerOpen DDPG table에 반영했다.
- 완료: FPP TD3 Random-Support/Random-Fixed-Count k73 artifact를 `metacog-rl@v0-lite` commit `51fb99f`에서 가져오고, config row/pair, aggregate table, updated TD3 learning curve, main draft TD3 control table에 반영했다.
- 완료: FPP SAC Random-Support/Random-Fixed-Count k76 artifact를 `metacog-rl@v0-lite` commit `d5439b2`에서 가져오고, config row/pair, aggregate table, updated SAC learning curve, main draft SAC control table에 반영했다.
- 완료: FPP TD3 high-RR \(4\times\) artifact를 `metacog-rl@v0-lite` commit `783a34c`에서 가져오고, config row/pair, aggregate table, updated TD3 learning curve, main draft high replay-ratio table에 반영했다.
- 완료: FPP SAC high-RR \(4\times\) artifact를 `metacog-rl@v0-lite` commit `62de0f7`에서 가져오고, config row/pair, aggregate table, main draft high replay-ratio table과 claim-boundary/limitation 문구에 반영했다.
- 완료: DrawerOpen DDPG Random-Support/PER-Style과 DrawerClose TD3 HER/SABER/Random-Support/Random-Fixed-Count artifact를 `origin/main`에서 가져오고, config row/pair, aggregate table, DrawerOpen table, DrawerClose stress-test table에 반영했다.
- 완료: Figure 1--2를 review용 publication-readable style로 재생성했다. 내부 파일명 title을 제거하고, 큰 axis/legend, seed별 thin lines, mean/std band, PDF+PNG export를 지원한다.
- diagnostic table을 원고에 추가했다. 필요하면 diagnostic plot을 추가한다.

### 산출물

- `analysis/result_pipeline/config.json`
- `analysis/result_pipeline/build_result_artifacts.py`
- `analysis/result_pipeline/README.md`
- `analysis/result_pipeline/output/per_seed_metrics.csv`
- `analysis/result_pipeline/output/aggregate_metrics.csv`
- `analysis/result_pipeline/output/aggregate_bootstrap_ci.csv`
- `analysis/result_pipeline/output/paired_seed_differences.csv`
- `analysis/result_pipeline/output/paired_bootstrap_ci.csv`
- `analysis/result_pipeline/output/learning_curves_long.csv`
- `analysis/result_pipeline/output/learning_curves_aggregate.csv`
- `analysis/result_pipeline/output/figures/*.png`
- `analysis/result_pipeline/output/report.md`

### 완료 기준

- main paper 7-page 제한 안에서 성능, mechanism, limitation이 모두 보인다.
- appendix/supplement가 없어도 주요 claim을 이해할 수 있다.

## 9.5 Source-of-Truth 관리

상태: 1차 완료. 현재 한국어 authoring source는 `paper/saber_aaai27_draft_kr.tex`로 고정한다. `gpt_review_upload/` 아래 파일들은 GPT 리뷰를 위한 round별 snapshot이며, 이후 수정 원본으로 사용하지 않는다. 최신 PDF는 `output/pdf/saber_aaai27_draft_kr_latest.pdf`이고, 이 PDF는 `paper/build_pdf.sh`로 생성한다.

### 목적

리뷰 round가 늘어나면서 TeX, PDF, result report, upload packet이 서로 다른 상태가 되는 문제를 막는다. 앞으로는 원고, 수치, 리뷰 패킷의 source-of-truth를 분리해서 관리한다.

### Authoring source

- 원고 원본: `paper/saber_aaai27_draft_kr.tex`
- 빌드 스크립트: `paper/build_pdf.sh`
- 최신 PDF: `output/pdf/saber_aaai27_draft_kr_latest.pdf`
- 리뷰 스냅샷: `gpt_review_upload/<round>/`

규칙: 원고 수정은 항상 `paper/saber_aaai27_draft_kr.tex`에서 한다. `gpt_review_upload/<round>/`에 들어간 TeX/PDF/문서는 해당 round의 첨부본이므로 직접 수정하지 않는다.

### Numeric source

- result pipeline 설정: `analysis/result_pipeline/config.json`
- aggregate metrics: `analysis/result_pipeline/output/aggregate_metrics.csv`
- paired CI: `analysis/result_pipeline/output/paired_bootstrap_ci.csv`
- 자동 리포트: `analysis/result_pipeline/output/report.md`
- 사람이 읽는 해석 matrix: `AAAI27_RESULT_MATRIX_KR.md`

규칙: 논문 표의 숫자는 `aggregate_metrics.csv`와 `paired_bootstrap_ci.csv`를 기준으로 확인한다. `AAAI27_RESULT_MATRIX_KR.md`는 해석과 artifact provenance를 관리하지만, 최종 numeric source는 pipeline output이다.

### Review packet 생성 순서

1. 새 실험 artifact를 로컬 작업트리에 복사하거나 pull한다.
2. `analysis/result_pipeline/config.json`의 `rows`와 `pairs`를 갱신한다.
3. `analysis/result_pipeline/build_result_artifacts.py`를 실행해 CSV/report를 재생성한다.
4. `paper/saber_aaai27_draft_kr.tex`의 표/본문 숫자와 claim boundary를 갱신한다.
5. `paper/build_pdf.sh`로 최신 PDF를 빌드한다.
6. PDF에서 table overflow, float warning, undefined citation/reference, 핵심 숫자 노출 여부를 확인한다.
7. 필요한 최소 파일만 새 `gpt_review_upload/<round>/`에 복사한다.

### 현재 반영 상태

- 반영 완료: FPP DDPG/TD3/SAC main rows, random-support/fixed/shuffled/PER controls, DDPG/TD3/SAC high-RR, REDQ-HER TQB, DDPG/TD3/SAC diagnostics, DrawerOpen DDPG controls, and DrawerClose stress rows.
- FPP SAC의 cross-method AUC/final/Hit90는 exact 50k common grid로 재계산해 pipeline과 두 원고에 반영했다. Native metrics는 provenance audit용으로 별도 보존한다.
- 기존 DrawerOpen `HGR` row는 canonical HGR가 아니라 importance correction이 없는 `HGR-style sampler-only` ablation으로만 사용한다.
- 진행 중이라 본문 수치로 쓰지 않음: FPP DDPG canonical HGR rerun, DrawerOpen/DrawerClose HGR runs, and DrawerOpen/DrawerClose REDQ-TQB runs. Drawer HGR rows는 완료 후 importance correction과 tuning fidelity를 확인하기 전까지 canonical로 부르지 않는다.
- figure 상태: result pipeline이 22개 curve figures를 만들고, submission용 method overview와 3000x1700 FPP aggregate evidence figure를 별도로 유지한다.

### 완료 기준

- 어떤 파일을 수정해야 최신 원고가 바뀌는지 명확하다.
- 어떤 CSV/report를 기준으로 표 숫자를 검증하는지 명확하다.
- 리뷰 업로드 폴더가 authoring source로 오인되지 않는다.
- 진행 중 실험과 원고 반영 완료 실험이 구분되어 있다.

## 10. 실행 순서

1. `AAAI27_RESULT_MATRIX_KR.md` 만들기: 기존 결과부터 확인한다.
2. `AAAI27_REPRODUCIBILITY_APPENDIX_KR.md` 만들기: 실험 세팅을 복구한다.
3. Method 수식/정의 수정 초안을 만든다.
4. Metric 정의와 table 오류를 수정한다.
5. support-vs-TD-error 분석 가능성을 확인한다.
6. FPP mechanism controls 부족분을 파악한다.
7. 필요한 새 실험 목록을 최소화한다.
8. Related Work를 보강한다.
9. 논문 claim/abstract/introduction을 수정한다.
10. 영어 LaTeX 초안으로 변환한다.

## 11. 현재 8개 작업 체크리스트

상태: 2026-07-10 기준. Full review draft와 page-limited submission skeleton을 분리해 관리한다.

| # | 작업 | 상태 | 증거 / 남은 일 |
| -: | --- | --- | --- |
| 1 | Source-of-Truth 정리 | 완료 | 원고 원본은 `paper/saber_aaai27_draft_kr.tex`, numeric source는 `analysis/result_pipeline/output/*.csv`와 `report.md`, 리뷰 폴더는 snapshot으로 명시했다. |
| 2 | Claim Boundary 고정 | 완료 | Abstract/Conclusion/Limitations에서 support-conditioned critic objective, one-update-per-step matched-backbone regime, mixed mechanism evidence로 제한했다. |
| 3 | 최신 실험 결과 반영 | 완료 | FPP SAC exact 50k common-grid, REDQ-HER TQB, SAC diagnostic, high-RR, random controls, and Drawer stress rows를 result pipeline과 TeX에 반영했다. |
| 4 | 표/그림 배치 정리 | 완료 | method overview, FPP aggregate evidence, scaling table, and control-closure table을 submission main body에 배치했다. Dense learning curves는 full review draft/pipeline output에 유지한다. |
| 5 | Reproducibility Appendix 유지 | 완료 | Appendix A에 source scope, env/reward convention, HER/terminal handling, seeds, optimization, support settings, runtime caveat가 들어 있다. |
| 6 | Consistency Audit | 완료 | pipeline 재실행 결과 220 per-seed rows, 44 aggregate rows, 1315 paired seed-difference rows, and 127197 curve points를 생성했다. Packaged SAC paired CIs와 aggregate values를 TeX와 대조했다. |
| 7 | PDF 빌드 검증 | 완료 | Full review draft 25쪽과 submission skeleton 5쪽을 각각 빌드·렌더링했다. LaTeX error, undefined citation/reference, overfull box, clipping, and overlap은 검출되지 않았다. |
| 8 | 작업 완료 기준 관리 | 진행 중 | Common-grid/REDQ 반영은 완료했다. Canonical HGR와 진행 중 Drawer HGR/REDQ는 fidelity와 5-seed completion을 확인한 뒤 별도 rows로 추가한다. |

### 다음 우선순위

1. FPP DDPG canonical HGR 1M, seeds 1--5를 IS correction과 documented tuning으로 완료한다.
2. DrawerOpen/DrawerClose HGR와 REDQ-TQB가 끝나면 config/IS correction/evaluation grid를 먼저 audit하고, 통과한 rows만 pipeline에 추가한다.
3. Submission skeleton을 최종 영어 7-page main body로 확장하고, full review draft의 상세 tables/curves를 blind supplement로 이동한다.
4. 새 review round 전에는 두 PDF를 재빌드하고 anonymization과 source-of-truth manifest를 다시 점검한다.

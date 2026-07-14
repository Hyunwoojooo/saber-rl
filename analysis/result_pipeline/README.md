# SABER Result Statistics Pipeline

이 폴더는 AAAI-27 원고용 result table과 learning curve figure를 재생성하기 위한 파이프라인이다.

## 사용법

```bash
python3 analysis/result_pipeline/build_result_artifacts.py
```

기본 입력은 `analysis/result_pipeline/config.json`이다. 새 실험이 끝나면 `rows`에 `summary_globs`만 추가하거나 기존 row의 glob이 새 폴더를 잡도록 두면 된다. 같은 public seed끼리 자동으로 paired difference와 bootstrap CI를 계산한다.

FPP SAC rows는 legacy 25k evaluation schedule과 newer 10k schedule을 직접 섞지 않는다. `metric_overrides`가 blind artifact `fpp_sac_common_grid_50k`의 exact 50k--1M grid 재분석을 seed별로 적용하고, `paired_ci_overrides`가 같은 artifact에서 검증된 AUC interval을 연결한다. 원본 `summary.json`은 수정하지 않으며 `per_seed_metrics.csv`의 `native_*`, `metric_source`, `metric_override_path` 열에 native 값과 provenance를 함께 남긴다. 해당 SAC/REDQ learning curves도 동일한 50k grid로 필터링된다.

FPP TD3 confirmatory rows는 `experiment_FPP/paper_artifacts/fpp_td3_her_vs_saber_n10_1m/`의 HER/SABER seeds 1--10을 별도 plot group으로 읽는다. Cross-backbone n=5 rows를 대체하지 않는다. Paired AUC interval은 artifact의 frozen `analysis/analysis_results.json`에서 import하며, exact sign-flip과 leave-one-seed-out 통계는 raw paired seed differences로 pipeline이 독립 재생성한다.

DrawerOpen REDQ-HER TQB는 `metaworld_Drawer_Open_REDQ/paper_artifacts/drawer_open_redq_her_tqb_2m/`의 exact 10k-grid summaries를 직접 읽는다. 이 row는 `N=10`, target subset `M=2`, UTD 4의 architecture/compute-scaled reference이며 matched DDPG row로 해석하지 않는다. 원격 README의 AUC spread는 population SD였으므로 pipeline과 논문은 per-seed curve에서 재계산한 sample SD를 사용한다.

## 출력

기본 출력 폴더는 `analysis/result_pipeline/output/`이다.

- `per_seed_metrics.csv`: seed별 final/best/AUC/runtime metric
- `aggregate_metrics.csv`: method별 mean/std, Hit90 reach count
- `aggregate_bootstrap_ci.csv`: method별 bootstrap mean confidence interval
- `paired_seed_differences.csv`: paired seed difference
- `paired_bootstrap_ci.csv`: paired bootstrap confidence interval
- `paired_exact_inference.csv`: exact two-sided sign-flip test, sign counts, and leave-one-seed-out mean range
- `learning_curves_long.csv`: seed별 curve long table
- `learning_curves_aggregate.csv`: step별 mean/std learning curve
- `figures/*.png`: learning curve plots. matplotlib이 있으면 matplotlib을 쓰고, 없으면 Pillow fallback을 쓴다.
- `report.md`: 사람이 읽는 요약 보고서

## 주의

- 기본 AUC normalization은 각 run의 `summary.json`에 저장된 값을 사용한다. FPP SAC/REDQ rows만 config에 선언된 common-grid artifact로 override한다. task 간 AUC 절대값 비교에는 쓰지 않는다.
- Bootstrap CI는 seed 수가 작기 때문에 uncertainty diagnostic일 뿐, 강한 통계적 유의성 주장으로 쓰지 않는다.
- Exact sign-flip도 finite-seed robustness diagnostic이다. 다섯 nonzero seed pair에서는 가능한 최소 two-sided p-value가 `0.0625`이므로, bootstrap CI와 exact test를 같은 의미로 해석하지 않는다.
- Bootstrap resampling seed는 고정되어 있으므로 같은 입력 artifact에서는 같은 CI가 재생성된다.
- `steps/sec`는 기록된 wall-clock runtime 기준 training throughput이다.
- Interrupted/resumed runs may retain resume-only diagnostics for audit. When
  present, the pipeline uses `reported_steps_per_sec` and
  `reported_runtime_total_sec` before legacy top-level runtime fields; nested
  `resume_only_diagnostics` values must never be used in paper tables.

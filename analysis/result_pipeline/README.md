# SABER Result Statistics Pipeline

이 폴더는 AAAI-27 원고용 result table과 learning curve figure를 재생성하기 위한 파이프라인이다.

## 사용법

```bash
python3 analysis/result_pipeline/build_result_artifacts.py
```

기본 입력은 `analysis/result_pipeline/config.json`이다. 새 실험이 끝나면 `rows`에 `summary_globs`만 추가하거나 기존 row의 glob이 새 폴더를 잡도록 두면 된다. 같은 public seed끼리 자동으로 paired difference와 bootstrap CI를 계산한다.

FPP SAC rows는 legacy 25k evaluation schedule과 newer 10k schedule을 직접 섞지 않는다. `metric_overrides`가 blind artifact `fpp_sac_common_grid_50k`의 exact 50k--1M grid 재분석을 seed별로 적용하고, `paired_ci_overrides`가 같은 artifact에서 검증된 AUC interval을 연결한다. 원본 `summary.json`은 수정하지 않으며 `per_seed_metrics.csv`의 `native_*`, `metric_source`, `metric_override_path` 열에 native 값과 provenance를 함께 남긴다. 해당 SAC/REDQ learning curves도 동일한 50k grid로 필터링된다.

## 출력

기본 출력 폴더는 `analysis/result_pipeline/output/`이다.

- `per_seed_metrics.csv`: seed별 final/best/AUC/runtime metric
- `aggregate_metrics.csv`: method별 mean/std, Hit90 reach count
- `aggregate_bootstrap_ci.csv`: method별 bootstrap mean confidence interval
- `paired_seed_differences.csv`: paired seed difference
- `paired_bootstrap_ci.csv`: paired bootstrap confidence interval
- `learning_curves_long.csv`: seed별 curve long table
- `learning_curves_aggregate.csv`: step별 mean/std learning curve
- `figures/*.png`: learning curve plots. matplotlib이 있으면 matplotlib을 쓰고, 없으면 Pillow fallback을 쓴다.
- `report.md`: 사람이 읽는 요약 보고서

## 주의

- 기본 AUC normalization은 각 run의 `summary.json`에 저장된 값을 사용한다. FPP SAC/REDQ rows만 config에 선언된 common-grid artifact로 override한다. task 간 AUC 절대값 비교에는 쓰지 않는다.
- Bootstrap CI는 seed 수가 작기 때문에 uncertainty diagnostic일 뿐, 강한 통계적 유의성 주장으로 쓰지 않는다.
- Bootstrap resampling seed는 고정되어 있으므로 같은 입력 artifact에서는 같은 CI가 재생성된다.
- `steps/sec`는 기록된 wall-clock runtime 기준 training throughput이다.

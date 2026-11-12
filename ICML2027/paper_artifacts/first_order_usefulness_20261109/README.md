# 실험 4 결과 요약 사본

측정 원자료는 `ICML2027/experiments/draweropen_ddpg_first_order_usefulness_20261109/outputs/`에 있고 git에서 제외된다. 여기에는 잃어버리면 안 되는 작은 요약 파일만 복사해 둔다. 숫자를 고치려면 원자료에서 다시 집계한다.

| 폴더 | 내용 | 원본 |
|---|---|---|
| `early_phase_a1/` | 초반부(A1, SABER·HER 학습, seed 101–103, 10k–500k) checkpoint 48개의 보고서, checkpoint별 표, 가설 판정 | `outputs/main/analysis_primary_3seeds/` (2026-11-09) |
| `late_ext2m/` | 후반부 2M(SABER·Random·TD 학습, seed 11–20) checkpoint 30개의 보고서, 표, 사전 고정 판정, 실행 환경 기록 | `outputs/late_ext2m/analysis/` (2026-11-10) |
| `late_mature/` | seed 1의 500k–2M(SABER·HER 학습) checkpoint 8개. 서술용 | `outputs/late_mature/analysis/` (2026-11-10) |
| `fpp_incl/` | FetchPickAndPlace TD3, inclusive-future HER·SABER seed 6–10 × 9개 시점, checkpoint 90개의 보고서와 표 | `outputs/fpp_incl/analysis/` (2026-11-10) |
| `fpp_primary/` | FetchPickAndPlace TD3, 주 비교 규칙으로 2026-11-10에 재학습한 HER·SABER seed 1–5 × 9개 시점, checkpoint 90개. 재학습 자체의 성능 비교는 [fpp_td3_primary_rerun_20261110](../fpp_td3_primary_rerun_20261110/README.md) | `outputs/fpp_primary/analysis/` (2026-11-10) |
| `validation/` | 검증 기록. DrawerOpen 초반부·후반부, FPP TD3, 코드 일반화 뒤의 재검증(`_v2`) | `outputs/validation_*/validation.json` |

- 설계: [04_first_order_usefulness/README_ko.md](../../experment_design/04_first_order_usefulness/README_ko.md)
- 코드: [draweropen_ddpg_first_order_usefulness_20261109](../../experiments/draweropen_ddpg_first_order_usefulness_20261109/README.md)
- 해석과 다음 일: [논리 공백 로드맵](../../experment_design/2026-11-10_logic_gap_roadmap_ko.md)

`checkpoints.csv`의 부호 규칙: 모든 gain은 `R(uniform 갱신 후) − R(arm 갱신 후)`이고, 양수면 그 arm이 정책 조회 오차를 더 줄였다. `per_sample.npz`(샘플별 u, κ 등)는 크기 때문에 복사하지 않았다.


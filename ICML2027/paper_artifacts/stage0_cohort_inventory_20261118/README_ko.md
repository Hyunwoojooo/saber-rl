# 0단계: 과제 × 알고리즘별 기존 HER·SABER 짝 정리 (2026-11-18)

- 새 학습 없음. 코드: [inventory.py](../../experiments/stage0_cohort_inventory_20261118/inventory.py)
- 출처: 이 작업 트리 + origin/main(`5facb92`). seed 6–10 확장과 Drawer SAC 결과는 origin/main에만 있다.
- AUC는 모든 실행에서 평가 곡선으로 다시 계산했다. 규칙은 n20과 같다: `eval/success_mean`(또는 evaluation.csv의 `success`), [10k, 끝] 구간 사다리꼴 적분.
- 논문 표의 seed 1–5 값이 8개 조건 모두 그대로 재현된다. 참고로 일부 `summary.json`의 `eval_AUC_success`는 다른 성공 정의를 써서 값이 다르다(예: FPP SAC HER 0.463 대 0.471).
- 평가 곡선이 같은 사본은 하나로 셌다. 짝은 같은 seed의 HER·SABER다. 합산은 seed마다 재실행이 아닌 첫 실행을 쓴다. FPP TD3의 2026-11 cpu·cuda 재실행은 합산에서 제외하고 따로 보고한다.
- 표: `pairs_by_cohort.csv`(집단별, 출처 포함), `runs.csv`(실행별 AUC, 곡선 해시, 모든 사본 경로와 출처)

## 합산 결과 (SABER − HER, AUC)

| 조건 | 짝 | HER | SABER | 차이 [95% CI] | +/− | sign-flip p |
|---|---|---|---|---|---|---|
| FPP DDPG | 5 | 0.539 | 0.609 | +0.070 [−0.016, +0.193] | 3/2 | 0.38 |
| **FPP TD3** | **30** | 0.375 | 0.421 | **+0.046 [+0.023, +0.067]** | 22/8 | <0.001 |
| FPP SAC | 5 | 0.471 | 0.583 | +0.112 [+0.064, +0.161] | 5/0 | 0.06 |
| DrawerOpen DDPG | 10 | 0.401 | 0.652 | +0.251 [−0.065, +0.545] | 6/4 | 0.16 |
| DrawerOpen TD3 | 10 | 0.137 | 0.458 | +0.321 [+0.006, +0.605] | 7/2 | 0.11 |
| DrawerOpen SAC | 10 | 0.487 | 0.495 | +0.008 [−0.059, +0.076] | 4/5 | 0.81 |
| DrawerClose DDPG | 10 | 0.279 | 0.315 | +0.035 [−0.011, +0.087] | 6/3 | 0.23 |
| DrawerClose TD3 | 5 | 0.329 | 0.549 | +0.220 [−0.216, +0.656] | 2/3 | 0.50 |
| DrawerClose SAC | 5 | 0.261 | 0.103 | −0.157 [−0.427, +0.024] | 1/4 | 0.31 |

seed 6–10 확장만 보면 다음과 같다.

| 조건 | seed 6–10 차이 | +/− | 비고 |
|---|---|---|---|
| DrawerOpen DDPG | −0.045 | 1/4 | seed 1–5의 +0.546과 반대 |
| DrawerOpen TD3 | +0.406 | 4/1 | |
| DrawerOpen SAC | +0.015 | 2/2 | |
| DrawerClose DDPG | −0.003 | 2/3 | |
| FPP TD3 | +0.023 | 4/1 | |

## 읽는 법

- **seed 20개 이상으로 이득이 확립된 조건은 FPP TD3 하나다.**
- FPP SAC와 DrawerOpen TD3는 이득 쪽이지만 짝이 5–10개다.
  - FPP SAC: 5/0이지만 짝 5개의 최소 p가 0.0625다.
  - DrawerOpen TD3: CI 하한이 겨우 0을 넘는다. HER가 대부분 실패하는 조건이다.
- DrawerOpen DDPG는 seed 1–5(+0.546)와 seed 6–10(−0.045)이 갈린다. 두 arm 모두 seed마다 완전 실패(AUC 0)가 나온다(HER seed 10, SABER seed 7).
- DrawerOpen SAC와 DrawerClose DDPG는 이득이 거의 없다. DrawerClose SAC는 오히려 SABER가 낮은 쪽이다.

## 출처에 관한 확인

- **DrawerOpen DDPG의 SABER seed 6–10:** 2026-10-24 확장 실행(`<WORKSPACE>`)의 평가 곡선이 2026-07-09 gen4 기기 실행과 비트 단위로 같다. 이 코드는 같은 seed면 완전히 재현된다. 따라서 이 짝은 시기가 달라도 유효하다.
- **기기별 분포:** FPP 실행은 `/home/mimiclab`(5–7월), Drawer seed 1–5는 `/home/mimiclab_asus`, 확장은 `<WORKSPACE>`(10–11월), n20은 이 PC다.
- **아직 짝이 없는 실행:** DrawerOpen DDPG의 SABER·random·TD top-k 배정 비교 seed 11–20(`~/ICML2027`)은 HER 짝이 없어서 이 표에 없다.

# SABER AAAI-27 Metric and Table Definitions

작성일: 2026-06-24
목적: 현재 논문 초안의 metric/table 표기 혼동을 줄이고, reviewer가 결과 표를 추적할 수 있게 정의와 수정 결정을 고정한다.

## 1. 확인한 source artifact

이번 정리는 아래 파일과 코드 기준으로 했다.

- FPP main result: `experiment_FPP/reports/fpp_main_experiment_results_20260604.md`
- FPP DDPG/B3 selected report: `experiment_FPP/results/fpp_selected_1m_comprehensive_report.md`
- FPP trainer metric code: `experiment_FPP/td3_her_baseline_ls512/code/src/metacog_lite_v0/trainer.py`
- FetchSlide report: `experiment_FetchSlide/reports/FetchSlide/agent_family_results.md`
- FetchSlide metric code: `experiment_FetchSlide/src/saber_fetch_slide/experiment.py`
- DrawerOpen report: `metaworld_Drawer_Open_TD3/README.md`
- DrawerOpen raw summaries: `metaworld_Drawer_Open_TD3/results_2m/**/summary.json`

## 2. 논문에서 사용할 metric 정의

### Evaluation success

각 evaluation checkpoint에서 `eval_episodes=20` episodes의 environment success rate를 계산한다. 표의 success 값은 이 evaluation success curve에서 나온다. Seed `s`의 evaluation curve를 다음처럼 둔다.

\[
\{(t_k, q_k^{(s)})\}_{k=1}^{K_s}
\]

여기서 `t_k`는 environment step, `q_k^{(s)}`는 해당 checkpoint의 success rate다.

### Final success

Run의 마지막 evaluation checkpoint에서의 success rate다.

\[
\mathrm{Final}^{(s)} = q_{K_s}^{(s)}
\]

FPP/FetchSlide는 1M step, DrawerOpen stress-test는 2M step 기준이다.

### Best success

Run 전체 evaluation checkpoints 중 maximum success rate다.

\[
\mathrm{Best}^{(s)} = \max_k q_k^{(s)}
\]

### Success AUC

Evaluation success curve를 trapezoidal rule로 적분한 normalized area다.

\[
\widehat{\mathrm{AUC}}^{(s)}(I;Z)
=
\frac{1}{Z}
\sum_{\ell=1}^{m-1}
\frac{q_{j_\ell}^{(s)} + q_{j_{\ell+1}}^{(s)}}{2}
\left(t_{j_{\ell+1}}-t_{j_\ell}\right)
\]

여기서 \(I=\{j_1,\ldots,j_m\}\)는 AUC 계산에 포함하는 selected checkpoint index 집합이고, \(Z\)는 normalization denominator다. 구현상 discrete checkpoints에 대해 `np.trapezoid` 또는 `np.trapz`를 사용한다.

주의:

- FPP trainer의 `_auc_success`는 첫 evaluation step과 마지막 evaluation step 사이의 span으로 normalize한다.
- DrawerOpen summary도 selected evaluation step span 기준 AUC를 사용한다.
- FetchSlide summary에는 `auc_eval_mean`과 `auc_trapz_0_to_total`이 둘 다 있고, report의 `Success AUC`는 task analyzer가 선택한 normalized AUC를 사용한다.
- 따라서 현재 draft에서는 AUC를 task 내부 비교에만 사용하고, 서로 다른 task 간 AUC 절대값을 직접 비교하지 않는다.
- 최종 AAAI 원고 전에는 모든 task의 AUC normalization을 하나로 재계산하는 것이 가장 안전하다.

### Last-quarter AUC

FetchSlide에서 사용하는 late-stage metric이다. `step >= 0.75 * total_steps`인 evaluation history만 모아, 마지막 25% 구간 길이로 normalize한 trapezoidal AUC다.

\[
\widehat{\mathrm{AUC}}_{\mathrm{last25}}^{(s)}
=
\widehat{\mathrm{AUC}}^{(s)}(I_{\mathrm{last25}};T/4),
\quad
I_{\mathrm{last25}}=\{k:t_k\ge 0.75T\}
\]

### Drop best-final

\[
\mathrm{Drop} = \mathrm{BestSuccess} - \mathrm{FinalSuccess}
\]

현재 draft의 main tables에는 넣지 않고, appendix/reproducibility table 또는 stability discussion에 넣는 편이 낫다. 본문에서 "주요 metric"이라고만 쓰면 표에 없어서 혼동을 만든다.

### Steps/sec

Run 종료 시점의 total environment steps divided by recorded wall-clock seconds다. Support lookup overhead를 포함한다. Runner에 따라 evaluation/logging overhead가 포함될 수 있으므로 hardware-normalized wall-clock superiority claim에 사용하지 않는다. Inference-time cost가 아니라 training throughput이다. SABER의 inference-time policy architecture는 baseline과 동일하다.

### `+/-`

모든 표의 `+/-`는 seeds 1--5에 대한 sample standard deviation으로 통일한다. Standard error가 아니다.

\[
\bar m = \frac{1}{5}\sum_{s=1}^5 m^{(s)}, \qquad
\mathrm{std}(m)=\sqrt{\frac{1}{4}\sum_{s=1}^5(m^{(s)}-\bar m)^2}
\]

현재 draft의 `mean +/- std`만으로 통계적 유의성을 주장하지 않는다. 최종 제출 전 paired-seed 비교를 추가한다면 같은 seed index끼리

\[
\Delta^{(s)} = m_{\mathrm{SABER}}^{(s)} - m_{\mathrm{HER}}^{(s)}
\]

를 만들고, mean paired difference와 bootstrap confidence interval을 별도로 보고한다.

## 3. First Hit / Hit90 표기 결정

### FPP Table 1

기존 `First Hit >= 0.90` 열은 실제로 "seed별 first-hit step의 median among reached seeds"처럼 쓰이고 있었다. 예:

- DDPG+HER hit90 seed steps: 510k, 600k, 610k, 920k, 570k -> median 600k
- DDPG+HER+SABER: 480k, 540k, 630k, 480k, 610k -> median 540k
- TD3+HER: 1000k, 980k, 970k, two seeds unreached -> median among reached seeds 980k

따라서 `TD3 HER Best Success 0.880`과 `First Hit >=0.90 980k`는 논리 모순이 아니다. Mean best success가 0.880일 뿐, 3/5 seeds는 0.90에 도달했다. 문제는 표기다.

Seed별 first-hit step은 다음처럼 정의한다.

\[
H_{0.90}^{(s)} = \min\{t_k:q_k^{(s)}\ge 0.90\}
\]

Threshold에 도달하지 못한 seed는 undefined로 두고 median 계산에서 제외한다. 대신 table cell에 reach count를 반드시 같이 적는다.

수정:

- 열 이름을 `Median Hit90 Step`으로 바꾼다.
- 각 cell은 `step (n/5)`로 쓴다.
- unreached seed는 median 계산에서 제외하고, reach count를 괄호에 명시한다.

FPP Table 1 수정값:

| Row | Median Hit90 Step |
| --- | --- |
| DDPG+HER | 600k (5/5) |
| DDPG+HER+SABER | 540k (5/5) |
| TD3+HER | 980k (3/5) |
| TD3+HER+SABER | 840k (5/5) |
| SAC+HER | 800k (3/5) |
| SAC+HER+SABER | 750k (5/5) |

### DrawerOpen

DrawerOpen 표의 `Hit90`은 step이 아니라 0.90 success를 달성한 seed count다. 따라서 열 이름을 `Hit90 Seeds`로 바꾼다. Step 정보를 본문 table에 섞으면 FPP의 first-hit step과 혼동된다.

## 4. Table 숫자/표기 수정 결정

### DrawerOpen AUC 0.361 vs 0.362

Raw summary 기준 Critic-Only AUC values:

- 0.4865577889
- 0.0000000000
- 0.0005025126
- 0.6899497487
- 0.6321608040

Mean = 0.361834..., 3 decimal rounding = 0.362.

수정:

- Table 4와 Table 5 모두 `0.362 +/- 0.338`로 통일한다.

### Method naming

FPP:

- Backbone column: `DDPG`, `TD3`, `SAC`
- Method column: `HER`, `HER+SABER`
- PER table은 full method name을 유지한다: `DDPG+HER+PER-style`, `DDPG+HER+PER-full`

FetchSlide:

- Backbone column: `DDPG`, `TD3`, `SAC`
- Method column도 FPP와 같이 `HER`, `HER+SABER`로 통일한다.
- 기존 `TD3 & SABER`, `SAC & SABER`는 `TD3 & HER+SABER`, `SAC & HER+SABER`로 바꾼다.

DrawerOpen:

- Main table: `TD3+HER`, `TD3+HER+SABER`
- Control table: `Critic-only SABER`, `Actor-only weighting`, `Actor+critic weighting`, `Random-support`, `Random-fixed-count`, `Shuffled-weight`
- `B3` prefix는 code/run naming으로만 남기고 paper table에서는 제거한다.

## 5. 현재 LaTeX에 반영할 수정

- `Tasks and Metrics` 문단에 metric definitions를 짧게 추가한다.
- FPP Table 1 열 이름을 `First Hit >=0.90`에서 `Median Hit90 Step`으로 바꾼다.
- FPP Table 1 각 hit cell에 `(n/5)` reach count를 추가한다.
- FetchSlide method label을 `HER+SABER`로 통일한다.
- DrawerOpen table의 `Hit90` 열을 `Hit90 Seeds`로 바꾼다.
- DrawerOpen control table의 `B3-...` method 이름을 paper-facing 이름으로 바꾼다.
- DrawerOpen Critic-only control AUC를 `0.362`로 통일한다.

## 6. 남은 후속 작업

- 최종 원고 전에는 FPP/FetchSlide/DrawerOpen AUC를 동일 normalization으로 재계산하는 script를 따로 만들면 좋다.
- Main table에 `Drop best-final`을 넣지 않을 경우, metric 문단에서 main metric처럼 나열하지 않는다.
- First-hit step은 threshold에 도달하지 못한 seed를 어떻게 처리하는지 항상 reach count와 함께 보고해야 한다.

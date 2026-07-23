# DrawerOpen DDPG SABER 10-seed diagnostic artifact

Artifact ID: `DRAWEROPEN-DDPG-SABER-DIAGNOSTIC-10SEED-2M`

이 패키지는 sparse MetaWorld DrawerOpen-v3에서 DDPG+HER+SABER critic-only diagnostic run 10개를 compact하고 재생성 가능한 형태로 제공한다.

## Cohort

- Seeds 1–5: diagnostic logging 재실행 cohort
- Seeds 6–10: 추가 diagnostic seed-extension cohort
- 기존 seeds 1–5 비-diagnostic run: 성능 재현 reference 전용

Seeds 1–5와 6–10은 같은 한 번의 matched 10-seed launch가 아니므로 `unmatched extension`이다. 기존 비-diagnostic seeds 1–5는 diagnostic 통계에 포함하지 않는다.

## 구조

- `data/seeds/seed_XX/eval_10k.csv`: seed당 10k–2M exact 평가점 200개
- `data/seeds/seed_XX/diagnostic_snapshots.csv.gz`: seed당 step-deduplicated snapshot 9,976개
- `data/seeds/seed_XX/{config,env_snapshot,summary}.json`: allowlist 익명화 metadata
- `data/reproduction_reference/`: 기존 seeds 1–5 비-diagnostic 평가 reference
- `results/diagnostic_all_10_support_activation_summary.csv`: all-10 support-positive rate 집계
- `results/diagnostic_all_10_td_residual*.csv`: outcome과 무관하게 seed 1–10을 포함한 primary diagnostic 집계
- `results/successful_9seed_td_residual*.csv`: Hit90 도달 seed만 포함한 outcome-stratified 보조 집계
- `results/`: compact data에서 재생성되는 분석 CSV
- `source_input_manifest.json`: 원본 입력의 상대경로, byte size, SHA256
- `artifact_manifest.json`: artifact payload의 byte size와 SHA256

## 재생성

Artifact root에서 다음을 실행한다.

```bash
python code/rebuild_results.py
```

재생성기는 raw 결과 디렉터리를 읽지 않고 `data/`만 사용한다. Python 3와 SciPy가 필요하다.

## 중복 제거

동일 step의 분석 대상 필드가 수치적으로 완전히 같은지 먼저 확인한 뒤 raw 파일에서 가장 마지막에 나타난 행을 선택한다. 분석 필드가 하나라도 다르면 builder는 중단한다. 이번 패키지에서 제거된 중복행은 2,000개다.

## Diagnostic 집계 원칙

Primary diagnostic aggregate는 성능 결과와 무관하게 instrumented seed 1–10을 모두 포함한다. Hit90에 도달한 9개 seed만의 결과는 outcome-stratified secondary analysis로 별도 보존한다. Phase별 support rate는 해당 phase 관측이 있는 seed만 평균하며 각 결과 행에 유효 seed 수를 기록한다.

## 필수 해석 제한

1. Residual reduction은 critic·actor·target update 후 **같은 weighted minibatch**에서 측정한 in-sample 값이다. Held-out 일반화나 인과적 성능 향상을 증명하지 않는다.
2. Phase는 원래 환경 goal이 아니라 `desired_goal_train` 기준이며 HER relabeled goal을 포함한다.
3. Support lookup은 `local_search_max=512` 제한을 사용한다. 512/full-scan 동등성은 이 artifact로 증명되지 않는다.
4. Primary TD/residual 및 phase 집계는 all-10이지만, Hit90 전후와 exploratory correlation은 정의상 Hit90 도달 9개 seed만 사용한다. Correlation은 multiple-comparison uncorrected다.
5. Checkpoint, raw `metrics.csv`, JSONL, run log는 payload에서 제외했다.

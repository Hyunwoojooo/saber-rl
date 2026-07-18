# FPP SAC + AER without HER

이 실험은 FetchPickAndPlace(FPP)에서 HER를 완전히 끄고, AER의 replay 선택
효과를 분리해 검증한다. 결론적으로 비교 단위는 다음 세 arm이다.

| Arm | Replay | Actor/Critic LR | Temperature LR |
|---|---|---|---|
| `sac_er` | uniform ER | 고정 `3e-4` | 고정 `3e-4` |
| `sac_annealing_er` | uniform ER | AER와 동일한 `lambda_t` 배율 | 고정 `3e-4` |
| `sac_aer` | attentive ER | AER의 `lambda_t` 배율 | 고정 `3e-4` |

`sac_er`만 대조군으로 두면 attentive sampling 효과와 논문에서 함께 사용한
learning-rate schedule 효과가 섞인다. 따라서 주 비교는
`sac_aer - sac_annealing_er`이고, `sac_aer - sac_er`는 전체 AER recipe의
효과를 보여주는 보조 비교다.

## Frozen protocol

세 arm은 동일한 FPP sparse-reward SAC 설정과 seed mapping을 사용한다.
`her_ratio=0.0`, `lambda_support=0`, `weight_scope=none`이며 SABER의 support
lookup/cache/Numba 경로도 모두 꺼져 있다. 즉 HER relabeling, SABER weighting,
PER importance weights는 어느 arm에도 들어가지 않는다.

Agent는 Gaussian actor, twin critic과 twin target critic을 갖는 SAC다. 자동
temperature tuning을 사용하며 `alpha=0.2`, `target_entropy=-action_dim=-4`,
temperature optimizer LR은 `3e-4`다. 매 replay update에서 두 critic은 같은
entropy-regularized target을 사용하고 actor·temperature를 갱신한 뒤 두 target
critic을 soft-update한다. 완료 gate는 twin critic, `log_alpha`, 세 optimizer와
SAC checkpoint 상태를 검사한다.

학습 중 평가는 10k step마다 deterministic 20 episodes로 수행한다. 학습 종료
뒤에는 final policy를 공통 seed로 deterministic 100 episodes 평가하고 각 run의
`final_evaluation_100.json`에 별도로 기록한다.

AER 설정은 `lambda0=2.5`, `alpha=2/3`, cosine similarity다. 각 update에서
`ceil(lambda_t * 256)`개 transition을 uniform candidate로 뽑은 후 현재 online
state와 가장 유사한 256개를 사용한다. state는
`[observation, original desired_goal]` 전체를 사용하며 relabeled goal은 없다.
`sac_aer`와 `sac_annealing_er`는 actor와 critic 양쪽의 base learning rate에
동일한 `lambda_t`를 곱한다. 원 AER의 actor/critic LR 규칙에 없는 SAC
temperature optimizer는 세 arm 모두 `3e-4`로 고정한다. 따라서 temperature
schedule이라는 새로운 요인을 추가하지 않는다. 수치 안정성을 위한 cosine
epsilon은 `1e-8`이다.

세부 값과 phase/seed 제한은 `config.json`이 유일한 실행 명세이고, `run.py`는
명세가 바뀌면 실행 전에 거부한다.

이는 AER 원 논문의 buffer/evaluation 설정을 그대로 복제한 실험이 아니라,
현재 FPP SAC backbone에서 AER 요소만 옮긴 matched transfer 실험이다.

## Correctness checks

세 arm의 환경·agent·HER/LR/AER 경로를 학습 없이 확인한다.

```bash
for arm in sac_er sac_annealing_er sac_aer; do
  python3 experiment_FPP/sac_aer_noher_1m/run.py \
    --arm "$arm" --seed 101 --steps 20000 --device cpu --validate_only
done

.venv/bin/python -B \
  experiment_FPP/sac_aer_noher_1m/code/src/test_aer_sampling.py

.venv/bin/python -B \
  experiment_FPP/sac_aer_noher_1m/code/src/test_sac_aer_runner.py
```

`metrics.csv/jsonl`에는 lambda, 후보/선택 cosine 통계, similarity uplift,
duplicate rate, transition age, sparse-success 비율, approach/grasp/lift/place
선택 비율, actor/critic effective LR, 고정 temperature LR, SAC alpha/log-prob,
sampling time과 전체 throughput이 기록된다.

## Execution ladder

먼저 명령만 확인할 수 있다.

```bash
DRY_RUN=1 bash experiment_FPP/sac_aer_noher_1m/run_smoke_20k_seeds1_5_5parallel.sh
DRY_RUN=1 bash experiment_FPP/sac_aer_noher_1m/run_correctness_20k_seed101.sh
```

1. 20k parallel smoke, seeds 1-5, 세 arm:

```bash
DEVICE=cuda:0 bash experiment_FPP/sac_aer_noher_1m/run_smoke_20k_seeds1_5_5parallel.sh
```

seed worker 5개를 병렬 실행하고, 각 worker 안에서는
`sac_er -> sac_annealing_er -> sac_aer` 순서로 실행한다. 따라서 동시에 실행되는
trainer는 최대 5개다. 이 smoke 결과는 구현 확인 전용이며 1M confirmatory 분석이나
checkpoint resume에 사용하지 않는다. 단일 GPU 공유 실행의 처리량은 arm 간 overhead
비교에도 사용하지 않는다. 20k run에서는 AER lambda schedule이 20k 기준으로
압축되므로 1M run의 처음 20k prefix와 동일하지 않다.

2. 선택 사항: 20k single-seed correctness, seed 101, 세 arm:

```bash
DEVICE=cuda bash experiment_FPP/sac_aer_noher_1m/run_correctness_20k_seed101.sh
```

이 스크립트는 trainer의 레거시 `--smoke`를 쓰지 않는다. `total_steps=20000`만
정확히 override하므로 batch size, warmup, replay capacity 같은 frozen 설정이
축소되지 않는다. 결과는 구현·runtime 검증용이며 성능 결론에 사용하지 않는다.

3. 300k pilot, seeds 101-103, 세 arm:

```bash
DEVICE=cuda bash experiment_FPP/sac_aer_noher_1m/run_pilot_300k_seeds101_103.sh
```

pilot은 feasibility 확인용 독립 재시작이며 20k run의 checkpoint를 이어받지
않는다. pilot 결과로 hyperparameter나 confirmatory seed를 선택하지 않는다.

4. 1M confirmatory, seeds 1-10, 세 arm:

```bash
DEVICE=cuda bash experiment_FPP/sac_aer_noher_1m/run_full_1m_seeds1_10.sh
```

기본 launcher는 한 device에서 seed별 세 arm을 순차 실행한다. 여러 GPU에서는
서로 겹치지 않는 seed shard로 같은 launcher를 각각 실행할 수 있다.

```bash
DEVICE=cuda:0 SEED_LIST="1 2 3 4 5" \
  bash experiment_FPP/sac_aer_noher_1m/run_full_1m_seeds1_10.sh &
DEVICE=cuda:1 SEED_LIST="6 7 8 9 10" \
  bash experiment_FPP/sac_aer_noher_1m/run_full_1m_seeds1_10.sh &
wait
```

같은 output/log root에 같은 seed를 중복 배정하지 않는다. launcher는 기존
run directory나 log를 발견하면 timestamp가 붙은 중복 run을 만들지 않고
실행 전에 중단한다.

### SAC-AER-first main queue after the 20k smoke

seed 1-5 smoke의 15개 run이 모두 끝난 뒤 SAC-AER만 우선하여 같은 seed의
1M run 5개를 병렬로 시작하려면 다음 queue를 별도 tmux session에서 실행한다.

```bash
tmux new-session -d \
  -s fpp-sac-aer-noher-main1m-after-smoke \
  -c "$PWD" \
  'env CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 bash experiment_FPP/sac_aer_noher_1m/queue_main_1m_after_smoke20k_s1_5.sh'
```

queue는 smoke session 종료만으로 통과시키지 않는다. 각 run의 frozen config,
10k/20k evaluation grid, final checkpoint, 100-episode fixed evaluation,
finite final loss/evaluation, summary와 종료 marker를 검사하고 로그에 runtime/CUDA
오류가 없는지 확인한다. 15개 중 하나라도 실패하면 main을 시작하지 않고 종료한다.
모두 통과하면 sampler regression과 SAC-AER 1M `validate_only`를 다시 실행한 후
`run_main_1m_sac_aer_seeds1_5_5parallel.sh`를 시작한다. 이 단계에서는 ER와
AnnealingER main run을 시작하지 않는다.

### Queue SAC-ER after SAC-AER

SAC-AER 1M seed 1-5가 끝난 뒤 SAC-ER 1M seed 1-5를 자동으로 5병렬
실행하려면 `queue_sac_er_main_after_sac_aer_s1_5.sh`를 별도 tmux session에서
실행한다. queue는 AER의 100개 evaluation grid, summary, final checkpoint,
fixed 100-episode evaluation과 fatal-error-free 종료를 모두 확인한다.

queue를 건 시점의 `config.json`, `run.py`, frozen `code/src`, ER launcher와
completion validator SHA-256도 저장한다. AER가 실행되는 동안 이 소스 중 하나라도
바뀌면 ER를 시작하지 않는다.

```bash
tmux new-session -d \
  -s fpp-sac-er-main1m-after-aer-s1-5 \
  -c "$PWD" \
  'env CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 bash experiment_FPP/sac_aer_noher_1m/queue_sac_er_main_after_sac_aer_s1_5.sh'
```

### Queue SAC-AnnealingER after SAC-ER

세 번째 control arm도 자동으로 이어서 실행하려면 다음 queue를 미리 건다. 이 queue는
위 SAC-ER tmux session의 종료를 기다리고, ER seed 1-5의 전체 artifact와 frozen source
hash를 검증한 뒤 SAC-AnnealingER seed 1-5를 5병렬로 시작한다.

```bash
tmux new-session -d \
  -s fpp-sac-annealing-er-main1m-after-er-s1-5 \
  -c "$PWD" \
  'env CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 bash experiment_FPP/sac_aer_noher_1m/queue_sac_annealing_er_main_after_sac_er_s1_5.sh'
```

개별 command 검증은 다음처럼 할 수 있다.

```bash
python3 experiment_FPP/sac_aer_noher_1m/run.py \
  --arm sac_aer --seed 101 --steps 20000 --device cpu --dry-run
```

중단된 개별 run은 동일 arm/seed/steps/output root로 재개한다. 기본 checkpoint는
50k step 간격이며 FPP의 50-step episode 경계와 일치한다. sampler와 PyTorch RNG,
optimizer, replay, AER schedule clock을 모두 복원하며 설정이나 seed가 다르면
재개를 거부한다.

```bash
python3 experiment_FPP/sac_aer_noher_1m/run.py \
  --arm sac_aer --seed 1 --steps 1000000 --device cuda \
  --resume
```

## Confirmatory readout

primary endpoint는 정확한 `10k, 20k, ..., 1M` grid의 success AUC다. 주 contrast는
paired seed 기준 `sac_aer - sac_annealing_er`, 보조 contrast는
`sac_aer - sac_er`다. interpolation, smoothing, forward fill은 사용하지 않는다.
20k correctness와 300k pilot은 confirmatory 분석에 합치지 않는다.

```bash
python3 experiment_FPP/sac_aer_noher_1m/analysis/analyze_sac_aer_noher.py
```

분석기는 정확한 grid를 강제하고 10,000회 paired bootstrap CI와 two-sided exact
sign-flip test를 계산한다. 누락·중복 evaluation이나 공통 설정 불일치는 즉시
오류로 처리한다.

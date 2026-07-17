# MetaWorld DrawerOpen DDPG+HER+HGR 정식 baseline

이 디렉터리는 DrawerOpen DDPG+HER에 Hindsight Goal Ranking(HGR)을 적용한 독립 실험
snapshot이다. `metaworld_Drawer_Open_DDPG/`의 B0/SABER 코드는 변경하지 않는다.

여기서 정식 baseline은
`experiment_FPP/ddpg_hgr_tdpriority_1m/`과 같은 completed-episode
pair-priority, importance sampling(IS), beta annealing, replay overwrite, checkpoint
semantics를 사용한다.

## HGR 개입 정의

HGR는 DDPG 자체를 바꾸지 않고 HER relabel 샘플의 선택 분포를 바꾼다.

에피소드 길이가 \(T\)일 때 완료된 에피소드마다 모든 유효한
\((j, i)\), \(0 \le j < i \le T\) 쌍을 저장한다. 코드에서는 transition slot
\(j\)와 `next_achieved_goal[k] = g_(k+1)`를 연결하므로 \(k \ge j\)이며, 한
에피소드의 pair 수는 \(T(T+1)/2\)이다. DrawerOpen의 \(T=200\)에서는 에피소드당
20,100개이다.

샘플링은 다음 두 단계다.

1. 에피소드 pair priority 평균으로 에피소드를 선택한다.
2. 선택된 에피소드 안에서 pair priority로 transition/future-goal 쌍을 선택한다.

확률은 각각 `alpha_episode=0.6`, `alpha_goal=0.6`을 적용한다. 새 pair는 현재 global
maximum priority로 초기화되며, critic update 직전에 계산된 absolute TD error를
사용해 실제로 샘플된 relabel pair만 갱신한다.

critic의 HGR relabel 샘플에는 에피소드와 pair 선택확률의 곱에 대한 IS weight를
적용하고, relabel 샘플 내 최대값이 1이 되도록 정규화한다. actor weight와 원래
desired-goal branch의 critic weight는 항상 1이다.

## 고정된 DrawerOpen 설정

- 환경: MetaWorld `drawer-open-v3`, sparse reward, reward function `v2`
- horizon: 200
- backbone: single-critic DDPG
- batch / replay: 256 / 300,000
- HER ratio: 0.8
- total steps: 2,000,000
- start / warmup steps: 10,000 / 5,000
- updates per environment step: 1
- actor / critic learning rate: 각각 3e-4
- gamma / tau: 0.98 / 0.005
- exploration noise: 0.2
- terminal convention: `success_terminates=false`, `timeout_bootstrap=true`
- seeds: 1, 2, 3, 4, 5

HGR 설정:

- `hgr_priority_eps = 1e-6`
- `hgr_priority_init = 1.0`
- `hgr_priority_clip_max = 0.0` (clipping 비활성)
- `hgr_alpha_episode = hgr_alpha_goal = 0.6`
- `hgr_beta_episode_initial = hgr_beta_goal_initial = 0.4`
- `hgr_beta_episode_final = hgr_beta_goal_final = 1.0`
- `hgr_beta_anneal_updates = 0`: 첫 critic update부터 마지막 예정 update까지 자동 선형 anneal
- `hgr_uniform_mix = 0.0`
- checkpoint schema: `hgr_episode_pair_priority_v1`

SABER support lookup/loss weighting, PER, high replay ratio는 사용하지 않는다.

## 출력 경로

Full HGR 실험은 다음 경로에 기록한다.

```text
runs/ddpg_hgr_tdpriority_original_pair_is_2m/
  DDPG-HER-HGR-original-pair-is-2m/
    drawer_open_ddpg_hgr_original_pair_is_s{1..5}_2m_cuda/
```

로그도 `logs/ddpg_hgr_tdpriority_original_pair_is_2m/` 아래에 분리한다.

## 완료 결과: Full HGR 2M

2026-07-15 UTC에 정식 pair-priority + IS 설정으로 seeds 1-5, seed당 2M
environment steps를 완료했다. 아래 AUC는 정확한 `10k, 20k, ..., 2M` 평가 grid에서
계산했으며 smoothing, interpolation, carry-forward를 사용하지 않았다.

| Seed | Final success | Best success | AUC | Last-quarter AUC | Hit90 | Steps/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.50 | 1.00 | 0.800 | 0.995 | 410k | 13.24 |
| 2 | 1.00 | 1.00 | 0.705 | 0.990 | 530k | 13.24 |
| 3 | 1.00 | 1.00 | 0.692 | 0.994 | 610k | 13.24 |
| 4 | 1.00 | 1.00 | 0.795 | 0.998 | 430k | 13.24 |
| 5 | 1.00 | 1.00 | 0.449 | 0.938 | 1060k | 13.26 |
| Mean | 0.90 | 1.00 | 0.688 | 0.983 | 608k | 13.24 |

- AUC sample standard deviation: 0.142
- Hit90 coverage: 5/5, median 530k
- matched HER 대비 paired AUC 차이: `+0.481`, 95% bootstrap CI
  `[0.151, 0.738]`
- critic-only SABER 대비 paired AUC 차이: `-0.065`, 95% bootstrap CI
  `[-0.219, 0.047]`

Full HGR은 matched DDPG+HER보다 강한 결과를 보였다. SABER의 mean AUC가 더 높지만,
HGR-minus-SABER paired interval이 0을 포함하므로 이 다섯 seed만으로 어느 한 방법의
우위를 주장할 수 없다. HGR은 HER sampling과 IS correction을 바꾸고 SABER는 고정된
HER minibatch의 critic-loss allocation을 바꾸므로, 이는 동일 backbone/budget의 전체
방법 비교이지 동일한 개입 지점의 mechanism-isolated 비교는 아니다.

원자료, exact-grid curves, seed별 compact payload, aggregate/paired 통계 및 검증 코드는
[`paper_artifacts/drawer_open_ddpg_hgr_pair_priority_is_2m/`](paper_artifacts/drawer_open_ddpg_hgr_pair_priority_is_2m/README.md)에 있다.

## 구현과 기록

- sampler: `code/src/metacog_lite_v0/hgr.py`
- replay metadata: `code/src/metacog_lite_v0/replay.py`
- training integration: `code/src/metacog_lite_v0/trainer.py`
- entrypoint: `code/src/train_drawer_open_lite_v0.py`
- 회귀 테스트: `code/src/test_hgr_fake_batch.py`
- launcher: `DDPG-HER-HGR-2M.sh`

각 run은 성능 지표와 함께 pair table 수/메모리, episode/pair 확률, raw/normalized IS
weight, beta와 진행률, clipping/safety replacement, sampling runtime을 기록한다.
체크포인트에는 replay와 pair table, global max priority, critic update count, beta
schedule state가 함께 들어간다.

## 검증

```bash
PYTHONPATH=metaworld_Drawer_Open_DDPG_HGR/code/src \
  .venv/bin/python -m py_compile \
  metaworld_Drawer_Open_DDPG_HGR/code/src/metacog_lite_v0/config.py \
  metaworld_Drawer_Open_DDPG_HGR/code/src/metacog_lite_v0/hgr.py \
  metaworld_Drawer_Open_DDPG_HGR/code/src/metacog_lite_v0/replay.py \
  metaworld_Drawer_Open_DDPG_HGR/code/src/metacog_lite_v0/trainer.py \
  metaworld_Drawer_Open_DDPG_HGR/code/src/train_drawer_open_lite_v0.py \
  metaworld_Drawer_Open_DDPG_HGR/code/src/test_hgr_fake_batch.py

PYTHONPATH=metaworld_Drawer_Open_DDPG_HGR/code/src \
  .venv/bin/python metaworld_Drawer_Open_DDPG_HGR/code/src/test_hgr_fake_batch.py

PYTHONPATH=metaworld_Drawer_Open_DDPG_HGR/code/src \
  .venv/bin/python metaworld_Drawer_Open_DDPG_HGR/code/src/train_drawer_open_lite_v0.py \
  --method hgr --weight_scope none --validate_only --device cpu

bash -n metaworld_Drawer_Open_DDPG_HGR/DDPG-HER-HGR-2M.sh
DRY_RUN=1 metaworld_Drawer_Open_DDPG_HGR/DDPG-HER-HGR-2M.sh
```

## 실행

정식 2M, seeds 1–5:

```bash
CUDA_DEVICE=0 metaworld_Drawer_Open_DDPG_HGR/DDPG-HER-HGR-2M.sh
```

일부 seed만 실행할 때는 공백으로 구분한다.

```bash
SEED_LIST="1 2" STEPS=200000 STEP_TAG=pilot200k \
  CUDA_DEVICE=0 metaworld_Drawer_Open_DDPG_HGR/DDPG-HER-HGR-2M.sh
```

## 운영상 주의

- full pair table은 horizon 200, replay 300k가 모두 완료 에피소드로 채워지면 seed당
  priority array만 약 121 MB를 사용한다.
- replay가 slot을 덮어쓸 때 해당 과거 에피소드의 pair table 전체를 제거한다.
- `hgr_priority_clip_max=0`이 정식 설정이다. 비정상 TD 값은 safety replacement
  metric으로 감시한다.
- 다섯 seed를 한 GPU에서 병렬 실행하므로 GPU 메모리와 처리량을 시작 직후 확인한다.

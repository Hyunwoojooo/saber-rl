# MetaWorld DrawerOpen TD3 Experiments

This directory contains the prepared TD3 experiment code for MetaWorld
`drawer-open-v3`. The final comparison set is:

| Variant | Internal method | Weighting |
| --- | --- | --- |
| TD3+HER | `b0_ddpg_her` | unit critic/actor weights |
| TD3+HER+B3-Critic-Only | `b3_support_runtime_opt` | SABER/B3 support weights on critic loss only |
| TD3+HER+PER-Style | `per_style_loss_weighting` | batch-local TD-error/PER-style critic loss weights |

The internal B0 method name is kept for compatibility with the existing
MetaCog-Lite code path, but the actual agent backend is TD3.

## Run

From the repository root:

```bash
bash metaworld_Drawer_Open/run_all_td3_drawer_open_5parallel.sh
```

Defaults:

- seeds: 1-5
- parallelism: 5 seeds per variant
- sequence: B0 -> B3-Critic-Only -> PER-Style
- steps: 1,000,000 per seed
- device: CUDA, using `CUDA_VISIBLE_DEVICES=0` by default

Useful overrides:

```bash
CUDA_DEVICE=0 STEPS=1000000 bash metaworld_Drawer_Open/run_all_td3_drawer_open_5parallel.sh
```

## Smoke Test

```bash
bash metaworld_Drawer_Open/run_td3_drawer_open_1000_smoke.sh
```

The smoke script runs a validate-only gate first, then short 1000-step runs for
all three variants.

## Output Layout

Training outputs are written under:

```text
metaworld_Drawer_Open/results/<variant>/
```

Runner logs are written under:

```text
metaworld_Drawer_Open/orchestrator_logs/
metaworld_Drawer_Open/logs/<variant>/
```

## Dependency

MetaWorld must be installed in the Python environment selected by the runner.
The runner prefers `.venv/bin/python`; otherwise it falls back to `python3`.

## Preliminary Results

Status as of 2026-06-10 11:17 KST:

- `TD3+HER`, `TD3+HER+B3-Critic-Only`, and `TD3+HER+PER-Style` completed 5 seeds at 1M steps.
- A 2M follow-up for `TD3+HER` and `TD3+HER+B3-Critic-Only` completed under `results_2m/`; the completed 2M summary is reported below.

Aggregate metrics use 5 seeds. Success values are environment success rates
from 20 evaluation episodes per checkpoint.

| Variant | Final success | Best success | AUC success | Hit90 | Speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER | 0.13 +/- 0.29 | 0.21 +/- 0.44 | 0.053 +/- 0.118 | 740k, 1/5 seeds | 41.7 steps/s |
| TD3+HER+B3-Critic-Only | 0.29 +/- 0.39 | 0.55 +/- 0.51 | 0.141 +/- 0.173 | 670k +/- 156k, 2/5 seeds | 23.3 steps/s |
| TD3+HER+PER-Style | 0.20 +/- 0.45 | 0.23 +/- 0.43 | 0.161 +/- 0.360 | 220k, 1/5 seeds | 38.7 steps/s |

Seed-level final success:

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER | 0.65 | 0.00 | 0.00 | 0.00 | 0.00 |
| TD3+HER+B3-Critic-Only | 0.05 | 0.00 | 0.00 | 0.85 | 0.55 |
| TD3+HER+PER-Style | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |

Interpretation:

- B3-Critic-Only improves the mean final success, best success, and AUC over
  the TD3+HER baseline in this preliminary 1M-step run.
- PER-Style achieves the highest AUC among the 1M variants, but this comes from
  one strongly successful seed and high variance across the remaining seeds.
- DrawerOpen appears harder and less stable than the Fetch tasks under this
  setup, so the completed 2M follow-up below should be used for the stronger
  B0 vs B3-Critic-Only comparison.

## 2M Follow-Up Results

Status as of 2026-06-12 13:21 KST:

- `TD3+HER` and `TD3+HER+B3-Critic-Only` completed 5 seeds at 2M steps.
- The follow-up used the same sparse DrawerOpen goal wrapper, TD3+HER backbone,
  `drawer-open-v3`, `reward_function_version=v2`, and 20 evaluation episodes.
- Random/actor-scope ablations are running separately and are not included in
  this completed-result summary.

Aggregate metrics use 5 seeds:

| Variant | Final success | Best success | AUC success | Hit90 | Speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER 2M | 0.190 +/- 0.425 | 0.210 +/- 0.442 | 0.127 +/- 0.283 | 740k, 1/5 seeds | 41.55 steps/s |
| TD3+HER+B3-Critic-Only 2M | 0.590 +/- 0.539 | 0.620 +/- 0.522 | 0.362 +/- 0.338 | 803k +/- 256k, 3/5 seeds | 21.57 steps/s |

Seed-level final success:

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER 2M | 0.95 | 0.00 | 0.00 | 0.00 | 0.00 |
| TD3+HER+B3-Critic-Only 2M | 0.95 | 0.00 | 0.00 | 1.00 | 1.00 |

Seed-level best success:

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER 2M | 1.00 | 0.00 | 0.00 | 0.05 | 0.00 |
| TD3+HER+B3-Critic-Only 2M | 1.00 | 0.00 | 0.10 | 1.00 | 1.00 |

Interpretation:

- Extending `TD3+HER` from 1M to 2M did not materially reduce seed variance:
  only seed 1 reached stable success, while seeds 2, 3, and 5 stayed at zero
  and seed 4 only reached a best success of 0.05.
- `TD3+HER+B3-Critic-Only` improved the number of seeds reaching Hit90 from
  1/5 to 3/5 and improved final success, best success, and AUC over the 2M
  TD3+HER baseline.
- The main remaining limitation is high seed variance: B3-Critic-Only still
  failed on seed 2, and seed 3 showed only a weak nonzero signal.

## 2M Actor-Scope Ablation Results

Status as of 2026-06-14 19:37 KST:

- `TD3+HER+B3-Actor-Only` completed 5 seeds at 2M steps.
- `TD3+HER+B3-Critic-Only` is repeated from the completed 2M follow-up above
  for the direct critic-vs-actor scope comparison.
- `TD3+HER+B3-Actor+Critic` is still running and is intentionally excluded
  from this completed-result summary.

Aggregate metrics use 5 seeds:

| Variant | Final success | Best success | AUC success | Hit90 | Speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only 2M | 0.590 +/- 0.539 | 0.620 +/- 0.522 | 0.361 +/- 0.338 | 803k +/- 256k, 3/5 seeds | 21.57 steps/s |
| TD3+HER+B3-Actor-Only 2M | 0.400 +/- 0.548 | 0.410 +/- 0.539 | 0.249 +/- 0.341 | 780k, 2/5 seeds | 21.48 steps/s |

Seed-level final success:

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only 2M | 0.95 | 0.00 | 0.00 | 1.00 | 1.00 |
| TD3+HER+B3-Actor-Only 2M | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |

Seed-level best success:

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only 2M | 1.00 | 0.00 | 0.10 | 1.00 | 1.00 |
| TD3+HER+B3-Actor-Only 2M | 1.00 | 1.00 | 0.00 | 0.05 | 0.00 |

Interpretation:

- Actor-Only can solve some seeds, but it is weaker than Critic-Only on this
  DrawerOpen 2M run: fewer Hit90 seeds, lower final success, lower best
  success, and lower AUC.
- The current completed ablation evidence supports the view that applying B3
  support weights to critic learning is the more reliable scope than applying
  them only to actor updates.


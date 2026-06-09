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

Status as of 2026-06-09 13:03 KST:

- `TD3+HER` and `TD3+HER+B3-Critic-Only` completed 5 seeds at 1M steps.
- `TD3+HER+PER-Style` was still running and is excluded from the completed-result table.

Aggregate metrics use 5 seeds. Success values are environment success rates
from 20 evaluation episodes per checkpoint.

| Variant | Final success | Best success | AUC success | Hit90 | Speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER | 0.13 +/- 0.29 | 0.21 +/- 0.44 | 0.053 +/- 0.118 | 740k, 1/5 seeds | 41.7 steps/s |
| TD3+HER+B3-Critic-Only | 0.29 +/- 0.39 | 0.55 +/- 0.51 | 0.141 +/- 0.173 | 670k +/- 156k, 2/5 seeds | 23.3 steps/s |

Seed-level final success:

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER | 0.65 | 0.00 | 0.00 | 0.00 | 0.00 |
| TD3+HER+B3-Critic-Only | 0.05 | 0.00 | 0.00 | 0.85 | 0.55 |

Interpretation:

- B3-Critic-Only improves the mean final success, best success, and AUC over
  the TD3+HER baseline in this preliminary 1M-step run.
- The result is high variance: B3 succeeds strongly on seeds 4 and 5, but seeds
  2 and 3 remain at zero final success.
- DrawerOpen appears harder and less stable than the Fetch tasks under this
  setup, so a 2M-step follow-up is a reasonable next check before making a
  strong benchmark claim.

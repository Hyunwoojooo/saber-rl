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

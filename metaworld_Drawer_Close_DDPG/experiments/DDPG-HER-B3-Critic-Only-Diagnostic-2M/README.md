# DrawerClose DDPG SABER Diagnostic (2M)

This isolated experiment reruns the matched DrawerClose DDPG+HER+B3 Critic-Only configuration with logging needed for four diagnostics. It does not modify the shared DrawerClose code or existing result directories.

## Frozen training configuration

- Environment: `drawer-close-v3`, sparse reward, reward function `v2`
- Backbone: DDPG
- Method: HER 0.8 + Main SABER (`b3_support_runtime_opt`), critic-only
- Budget: 2,000,000 environment steps, UTD 1, horizon 200
- Seeds: 1-5
- Batch/buffer: 256 / 300,000
- Support: `M_B=16`, `M_Q_min=4`, `N_local_min=4`, `local_search_max=512`
- Weight: `lambda_support=1`, minibatch-mean normalization

Except for `trainer.py`, the training snapshot is byte-identical to the shared DrawerClose DDPG Main source for the files checked by `test_drawer_close_diagnostic_contract.py`. The trainer adds forward-only measurements after the normal parameter update and preserves metric history when resuming from an episode-boundary checkpoint. The uninterrupted parameter-update path adds no loss term, gradient, replay operation, random draw, or parameter update.

## Four diagnostics

1. **Support activation**: `diagnostic/support_positive_rate` and local support counts.
2. **TD-error association**: support-positive/negative pre-update absolute TD error, Pearson correlation, and support prevalence/enrichment within the batch's top-10% TD-error subset.
3. **Same-minibatch one-update correction**: absolute TD error before and immediately after one normal DDPG update on the same sampled batch, split by support. `diagnostic/same_minibatch_one_update=1` marks valid snapshots.
4. **Phase-conditioned allocation**: batch prevalence, support-positive rate, TD error, and residual reduction for `approach`, `contact`, `closing`, and `closed` under the HER training goal.

The residual diagnostic is an in-sample optimization response. As in the DrawerOpen diagnostic, the post-update measurement occurs after the normal actor and target-network updates, so it is not a fixed-target or held-out generalization estimate.

## Run

```bash
bash DDPG-HER-B3-Critic-Only-Diagnostic-2M.sh
```

Useful overrides:

```bash
CUDA_DEVICE=1 SEEDS="1 2 3 4 5" bash DDPG-HER-B3-Critic-Only-Diagnostic-2M.sh
RESUME=1 bash DDPG-HER-B3-Critic-Only-Diagnostic-2M.sh
```

`RESUME=1` accepts episode-boundary checkpoints. The default 50k checkpoint interval is divisible by the 200-step horizon and satisfies this contract.

Each run preserves `metrics.csv`, `metrics.jsonl`, `config.json`, `env_snapshot.json`, `summary.json`, and `checkpoint.pt`.

## Build the compact diagnostic artifact

After all five seeds finish:

```bash
python analysis/build_diagnostic_artifact.py
```

The builder validates Main SABER semantics and seed coverage, removes duplicate metric rows at the same step, and writes:

- `diagnostic_artifact/data/seeds/seed_XX/diagnostic_snapshots.csv.gz`
- `diagnostic_artifact/results/diagnostic_by_seed.csv`
- `diagnostic_artifact/results/diagnostic_aggregate.csv`
- `diagnostic_artifact/results/diagnostic_by_phase_window.csv`
- `diagnostic_artifact/validation/validation_summary.json`
- `diagnostic_artifact/artifact_manifest.json`

Snapshot values are first summarized within each seed; aggregate rows treat seeds, not snapshots, as the independent units. Phase support/TD/residual means are weighted by each phase's minibatch prevalence.

## Verification

```bash
PYTHONPATH=code/src python code/src/test_drawer_close_diagnostic_contract.py
bash -n DDPG-HER-B3-Critic-Only-Diagnostic-2M.sh
python -m py_compile analysis/build_diagnostic_artifact.py code/src/metacog_lite_v0/trainer.py
```

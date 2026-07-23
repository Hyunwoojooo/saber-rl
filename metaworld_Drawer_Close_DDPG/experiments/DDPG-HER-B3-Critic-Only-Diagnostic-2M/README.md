# DrawerClose DDPG SABER Diagnostic (2M)

This isolated experiment reruns the matched DrawerClose DDPG+HER+B3
Critic-Only configuration with the logging needed for four diagnostics. It
does not modify the shared DrawerClose source or existing result directories.

## Frozen training configuration

- Environment: `drawer-close-v3`, sparse reward, reward function `v2`
- Backbone: DDPG
- Method: HER 0.8 + Main SABER (`b3_support_runtime_opt`), critic-only
- Budget: 2,000,000 environment steps, UTD 1, horizon 200
- Seeds: 1-5
- Batch/buffer: 256 / 300,000
- Support: `M_B=16`, `M_Q_min=4`, `N_local_min=4`,
  `local_search_max=512`
- Weight: `lambda_support=1`, minibatch-mean normalization

The isolated `trainer.py` and `agent.py` add diagnostic-only target capture and
forward measurements. The remaining training files checked by
`test_drawer_close_diagnostic_contract.py` are byte-identical to the shared
DrawerClose DDPG Main source. No extra loss term, gradient, replay operation,
random draw, or parameter update is introduced.

## Four diagnostics

1. **Support activation**: support-positive rate and local support counts.
2. **TD-error association**: support-positive/negative pre-update absolute TD
   error, Pearson correlation, and support prevalence/enrichment among exactly
   `ceil(0.1 * batch_size)` samples. For batch size 256 this is 26 samples,
   selected by a stable descending sort.
3. **Same-minibatch one-update correction**: the pre-update Bellman target is
   captured, used for the normal critic update, and held fixed when computing
   `|Q_after - target_before|`. The older live-target interpretation is also
   recorded under separate `diagnostic/live_target_*` fields.
4. **Phase-conditioned allocation**: batch prevalence, support-positive rate,
   TD error, and fixed-target residual reduction for `approach`, `contact`,
   `closing`, and `closed` under the HER training goal.

The residual diagnostic remains an in-sample optimization response. It is not
a held-out critic-generalization estimate or proof that the diagnostic mediates
long-term policy performance.

## Run

```bash
bash DDPG-HER-B3-Critic-Only-Diagnostic-2M.sh
```

Useful overrides:

```bash
CUDA_DEVICE=1 SEEDS="1 2 3 4 5" bash DDPG-HER-B3-Critic-Only-Diagnostic-2M.sh
```

Resume is deliberately disabled. MetaWorld simulator state is not preserved in
the checkpoint, so a resumed process does not reproduce an uninterrupted
trajectory. Each official seed must run continuously from step 0 to 2M. The
checkpoint remains available for inspection, but not for continuation.

Each run preserves `metrics.csv`, `metrics.jsonl`, `config.json`,
`env_snapshot.json`, `summary.json`, and `checkpoint.pt`.

## Build the compact diagnostic artifact

After all five uninterrupted seeds finish:

```bash
python analysis/build_diagnostic_artifact.py
```

For a complete 2M artifact, the builder requires every seed to contain exactly
9,976 diagnostic snapshots on the 5,000-to-2,000,000 grid at a 200-step
cadence. It also verifies fixed-target semantics, exact Top-10% cardinality,
the four diagnostic contracts, and final training-step coverage.

Copied JSON is sanitized: local path keys are removed and remaining absolute
path strings are redacted. Gzip output is deterministic, and
`artifact_manifest.json` never hashes itself, so rebuilding from unchanged
inputs produces the same manifest.

The builder writes:

- `diagnostic_artifact/data/seeds/seed_XX/diagnostic_snapshots.csv.gz`
- `diagnostic_artifact/results/diagnostic_by_seed.csv`
- `diagnostic_artifact/results/diagnostic_aggregate.csv`
- `diagnostic_artifact/results/diagnostic_by_phase_window.csv`
- `diagnostic_artifact/validation/validation_summary.json`
- `diagnostic_artifact/artifact_manifest.json`

Snapshot values are first summarized within each seed; aggregate rows treat
seeds, not snapshots, as the independent units. Phase support, TD, and residual
means are weighted by each phase's minibatch prevalence.

## Verification

```bash
PYTHONPATH=code/src python code/src/test_drawer_close_diagnostic_contract.py
(cd analysis && python -m unittest -v test_build_diagnostic_artifact.py)
bash -n DDPG-HER-B3-Critic-Only-Diagnostic-2M.sh
python -m py_compile analysis/build_diagnostic_artifact.py code/src/metacog_lite_v0/agent.py code/src/metacog_lite_v0/trainer.py
```

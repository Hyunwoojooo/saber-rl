# FPP SAC Controls and REDQ-HER 1M Artifacts

This directory is a compact paper-analysis collection for FetchPickAndPlace.
It contains five SAC-based 1M conditions and one REDQ-HER 1M condition, each
with seeds 1 through 5.

## Included Conditions

| Folder | Method | Updates/step | Purpose |
|---|---|---:|---|
| `sac_her_baseline_ls512` | `sac` | 1 | SAC+HER reference |
| `sac_saber_critic_only_v4_ls512` | `sac_saber` | 1 | SAC+HER+SABER critic-only |
| `sac_support_random_critic_only_ls512` | `sac_support_random` | 1 | Random-Support control |
| `sac_random_fixed_count_k76_critic_only_ls512` | `sac_random_fixed_count` | 1 | Random-Fixed-Count control, K=76 |
| `sac_her_high_replay_ratio_x4` | `sac_her_high_rr_x4` | 4 | Compute-scaled SAC+HER control |
| `redq_her_utd4_n10_m2_tqbound_ls512_1m` | `redq_her` | 4 | REDQ-HER, N=10, M=2, target-Q bound |

Common settings are FetchPickAndPlace, sparse reward, 1,000,000 environment
steps, HER ratio 0.8, and seeds 1-5. Exact settings remain in each run's
`config.json`.

## Included Files

Every seed directory contains exactly:

- `metrics_selected_10k.csv`
- `summary.json`
- `config.json`

The collection intentionally excludes source code, checkpoints, model weights,
replay buffers, raw metrics, logs, environment snapshots, and diagnostic sample
files.

## Provenance

- SAC HER and SAC SABER: `saber-rl` commit `368a76686dc092e1fc676af2c6a6b79d9d7a6bdb`
- SAC random controls: `metacog-rl` commit `d5439b2e4512492850f746b322fc19a8727d479f`
- SAC high replay-ratio x4: `metacog-rl` commit `62de0f7c2f1c03a273fe7808d8092a3c9b598327`
- REDQ-HER: `metacog-rl` commit `8fd6f8e96a67cf4e5f3dd311c64b2999013d513c`

`artifact_manifest.json` provides the same information in a machine-readable
form.

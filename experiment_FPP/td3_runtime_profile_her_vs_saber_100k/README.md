# FPP TD3 HER vs HER+SABER Runtime Profiler

This experiment profiles where SABER runtime is spent under a matched one-update-per-step TD3+HER backbone. It is a 100k-step implementation profile, not a performance or wall-clock superiority experiment.

## Matched conditions

Both methods execute the same source snapshot in `code/src`. The environment, TD3 networks and optimizers, replay buffer, HER future relabeling, seed mapping, batch size, update ratio, evaluation schedule, and terminal handling are shared.

The only method-specific settings are:

| Setting | TD3+HER | TD3+HER+SABER |
|---|---|---|
| `method` | `td3_her` | `td3_saber_runtime_opt` |
| `weight_scope` | `none` | `critic_only` |
| `lambda_support` | `0.0` | `1.0` |
| support lookup/cache/Numba | off | on |
| `local_search_max` | unused | `512` |
| diagnostic audit | off | off |

HER therefore uses unit critic and actor weights. SABER changes only critic-side loss allocation.

## Profile windows

Steps 0-20k are cold-start/warm-up and are excluded. The primary window is `(20k, 100k]`, with `(20k, 40k]`, `(40k, 70k]`, and `(70k, 100k]` subwindows.

Paper-level groups are:

- Interaction: action selection, environment step, replay insertion.
- Replay + HER: replay sampling/materialization, future-goal relabeling, reward recomputation, terminal preparation.
- Batch preparation: NumPy state construction and CPU-to-device tensors.
- Support and weighting: policy action query, context hashing, local lookup, support predicate, candidate/Q work, weight construction and normalization.
- Critic update: target construction, twin critic forward/loss/backward/optimizer.
- Actor + target update: delayed actor work and target soft updates.
- Eval/logging amortized: evaluation, metric/log writes, and checkpoints.

`local_context` and `weight_computation` aliases are not added to their constituent timings. `Other/unaccounted` is the remaining measured loop time after mutually exclusive training groups.

## CUDA timing

Ordinary updates are not synchronized. Every 100th profiled update uses CUDA Events around critic and delayed actor/target stages, followed by one synchronization to materialize event durations. Aggregate CPU component statistics retain every profiled call, while `runtime_profile_events.jsonl` keeps only interval-sampled steps/updates. Event administration, JSONL writing, and synchronization diagnostics are recorded separately. `CUDA_LAUNCH_BLOCKING=1` is not used. Profiling off adds no CUDA synchronization and emits no profiler files.

## Commands

Dry-run both commands:

```bash
.venv/bin/python experiment_FPP/td3_runtime_profile_her_vs_saber_100k/run.py --method td3_her --dry-run
.venv/bin/python experiment_FPP/td3_runtime_profile_her_vs_saber_100k/run.py --method td3_saber_runtime_opt --dry-run
```

Run the sequential 100k pair on one GPU:

```bash
CUDA_DEVICE=0 \
OUTPUT_ROOT=experiment_FPP/runs/td3_runtime_profile_her_vs_saber_100k \
bash experiment_FPP/td3_runtime_profile_her_vs_saber_100k/run_fpp_td3_runtime_profile_pair_100k.sh
```

Run the short validation smoke:

```bash
CUDA_DEVICE=0 bash experiment_FPP/td3_runtime_profile_her_vs_saber_100k/run_fpp_td3_runtime_profile_smoke.sh
```

Analyze existing summaries:

```bash
.venv/bin/python experiment_FPP/td3_runtime_profile_her_vs_saber_100k/analyze_runtime_profile.py \
  --her-summary /path/to/her/runtime_profile_summary.json \
  --saber-summary /path/to/saber/runtime_profile_summary.json \
  --output-dir /path/to/comparison
```

## Outputs

Each run writes the existing `config.json`, `env_snapshot.json`, `run.log`, `metrics.csv`, `metrics.jsonl`, and `summary.json`, plus:

- `runtime_profile_summary.json`: aggregate components, counts, throughput, subwindows, cache rates, and hardware metadata.
- `runtime_profile_events.jsonl`: raw sampled/profile events when enabled.
- `runtime_profile_components.csv`: paper-level component table.

The analysis writes `runtime_profile_comparison.json`, `.csv`, and `.md`.

## Limits

Results are hardware-, driver-, PyTorch-, CUDA-, and implementation-specific. This profile does not establish wall-clock efficiency or superiority. A 100k result must not be linearly extrapolated into a 1M runtime claim because replay size, support-cache behavior, success frequency, and evaluation/checkpoint costs change over training.

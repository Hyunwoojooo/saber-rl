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

## Code Layout

Most TD3 DrawerOpen variants share the same implementation under
`B3_v4_512_s1_5/`; individual experiments are selected by runner arguments:

- `run_td3_drawer_open_b0_b3_2m_5parallel.sh`: B0, B3-Critic-Only
- `run_td3_drawer_open_actor_only_2m_5parallel.sh`: B3-Actor-Only
- `run_td3_drawer_open_actor_critic_2m_5parallel.sh`: B3-Actor+Critic
- `run_td3_drawer_open_random_support_2m_5parallel.sh`: Random-Support
- `run_td3_drawer_open_random_fixed_count_2m_5parallel.sh`: Random-Fixed-Count
- `run_td3_drawer_open_shuffled_weight_2m_5parallel.sh`: Shuffled-Weight
- `run_td3_drawer_open_ablation_2m_5parallel.sh`: Random-Support -> Actor-Only -> Actor+Critic

The support-rate cap variant is intentionally isolated in
`TD3-HER-B3-Critic-Only-SupportRateCap-04/` so that its cap-specific code does
not overwrite the completed baseline B3 implementation.

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
- Random/actor-scope ablations are reported separately below.

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

Status as of 2026-06-17 17:15 KST:

- `TD3+HER+B3-Actor-Only` completed 5 seeds at 2M steps.
- `TD3+HER+B3-Actor+Critic` completed 5 seeds at 2M steps.
- `TD3+HER+B3-Critic-Only` is repeated from the completed 2M follow-up above
  for the direct critic-vs-actor scope comparison.

Aggregate metrics use 5 seeds:

| Variant | Final success | Best success | AUC success | Hit90 | Speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only 2M | 0.590 +/- 0.539 | 0.620 +/- 0.522 | 0.361 +/- 0.338 | 803k +/- 256k, 3/5 seeds | 21.57 steps/s |
| TD3+HER+B3-Actor-Only 2M | 0.400 +/- 0.548 | 0.410 +/- 0.539 | 0.249 +/- 0.341 | 780k, 2/5 seeds | 21.48 steps/s |
| TD3+HER+B3-Actor+Critic 2M | 0.600 +/- 0.548 | 0.610 +/- 0.534 | 0.432 +/- 0.395 | 603k +/- 55k, 3/5 seeds | 21.75 steps/s |

Seed-level final success:

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only 2M | 0.95 | 0.00 | 0.00 | 1.00 | 1.00 |
| TD3+HER+B3-Actor-Only 2M | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| TD3+HER+B3-Actor+Critic 2M | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 |

Seed-level best success:

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only 2M | 1.00 | 0.00 | 0.10 | 1.00 | 1.00 |
| TD3+HER+B3-Actor-Only 2M | 1.00 | 1.00 | 0.00 | 0.05 | 0.00 |
| TD3+HER+B3-Actor+Critic 2M | 1.00 | 1.00 | 0.05 | 1.00 | 0.00 |

Interpretation:

- Actor-Only can solve some seeds, but it is weaker than Critic-Only on this
  DrawerOpen 2M run: fewer Hit90 seeds, lower final success, lower best
  success, and lower AUC.
- Actor+Critic is competitive with Critic-Only in aggregate final success and
  AUC, but it does not cleanly dominate: both reach Hit90 on 3/5 seeds and the
  best-success difference is negligible under high seed variance.
- The completed ablation evidence supports using Critic-Only as the cleaner
  primary SABER/B3 scope, with Actor+Critic reported as a competitive but less
  isolated variant.


## 2M Random-Support Negative Control Results

Status as of 2026-06-17 17:15 KST:

- `TD3+HER+B3-Random-Support` completed 5 seeds at 2M steps.
- The variant keeps the B3-style weighted critic update path but replaces the
  structured support assignment with random support weights.
- `TD3+HER+B3-Critic-Only` is repeated from the completed 2M follow-up above
  for the direct structured-vs-random support comparison.

Aggregate metrics use 5 seeds:

| Variant | Final success | Best success | AUC success | Hit90 | Speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only 2M | 0.590 +/- 0.539 | 0.620 +/- 0.522 | 0.361 +/- 0.338 | 803k +/- 256k, 3/5 seeds | 21.57 steps/s |
| TD3+HER+B3-Random-Support 2M | 0.600 +/- 0.548 | 0.600 +/- 0.548 | 0.426 +/- 0.392 | 570k +/- 141k, 3/5 seeds | 20.84 steps/s |

Seed-level final success:

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only 2M | 0.95 | 0.00 | 0.00 | 1.00 | 1.00 |
| TD3+HER+B3-Random-Support 2M | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |

Seed-level best success:

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only 2M | 1.00 | 0.00 | 0.10 | 1.00 | 1.00 |
| TD3+HER+B3-Random-Support 2M | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |

Interpretation:

- Random-Support performs unexpectedly strongly on this DrawerOpen 2M run:
  it matches Critic-Only on Hit90 count and mean final success, and has a
  higher AUC in this seed set.
- The result suggests that part of the gain may come from support-conditioned
  critic reweighting and exploration stabilization rather than only from the
  exact structured support rule.
- Because the successful seeds differ from Critic-Only, this should be reported
  as an important negative-control result rather than folded into the main
  structured-support claim.

## 2M Fully Random Fixed-Count Negative Control Results

Status as of 2026-06-18 08:09 KST:

- `TD3+HER+B3-Random-Fixed-Count` completed 5 seeds at 2M steps.
- This control keeps critic-only loss reweighting, `lambda_support=1.0`,
  batch-normalized weights, and a fixed high-weight count of `k=122` per
  256-sample batch.
- It does not call the SABER support scanner, does not perform local support
  lookup, and does not use support-positive counts. High-weight samples are
  chosen uniformly at random each batch.

Aggregate metrics use 5 seeds:

| Variant | Final success | Best success | AUC success | Hit90 | Speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only 2M | 0.590 +/- 0.539 | 0.620 +/- 0.522 | 0.361 +/- 0.338 | 803k +/- 256k, 3/5 seeds | 21.57 steps/s |
| TD3+HER+B3-Random-Support 2M | 0.600 +/- 0.548 | 0.600 +/- 0.548 | 0.426 +/- 0.392 | 570k +/- 141k, 3/5 seeds | 20.84 steps/s |
| TD3+HER+B3-Random-Fixed-Count 2M | 0.200 +/- 0.447 | 0.200 +/- 0.447 | 0.144 +/- 0.322 | 520k, 1/5 seeds | 41.60 steps/s |

Seed-level final success:

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only 2M | 0.95 | 0.00 | 0.00 | 1.00 | 1.00 |
| TD3+HER+B3-Random-Support 2M | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| TD3+HER+B3-Random-Fixed-Count 2M | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |

Seed-level best success:

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only 2M | 1.00 | 0.00 | 0.10 | 1.00 | 1.00 |
| TD3+HER+B3-Random-Support 2M | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| TD3+HER+B3-Random-Fixed-Count 2M | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |

Implementation diagnostics:

| Metric | Value |
| --- | ---: |
| `random_fixed_count_k` | 122 |
| `assigned_high_weight_rate` | 0.4766 |
| `mean_w_P` | 1.4766 |
| `psi_q_rank_rate` | 0.0000 |
| support lookup used | no |

Interpretation:

- Fully Random Fixed-Count succeeds on only 1/5 seeds, while Random-Support and
  Critic-Only each solve 3/5 seeds in this DrawerOpen 2M setting.
- This weakens the explanation that the gains come only from random critic loss
  reweighting. A fixed amount of random high weighting is not sufficient to
  reproduce the support-aware controls.
- The result supports the importance of support-conditioned weighting. The
  completed Shuffled-Weight control below further separates support-density
  effects from exact sample-level support alignment.

## 2M Shuffled-Weight Negative Control Results

Status as of 2026-06-19 13:39 KST:

- `TD3+HER+B3-Shuffled-Weight` completed 5 seeds at 2M steps.
- This control computes the same SABER/B3 raw support weight vector as
  Critic-Only, then shuffles the raw weights across the batch before applying
  critic-only loss weighting.
- It preserves support lookup, support-density information, batch-normalized
  weight scale, cache/JIT runtime optimization, and the critic-only update
  path, but breaks exact sample-to-weight alignment.

Aggregate metrics use 5 seeds:

| Variant | Final success | Best success | AUC success | Hit90 | Speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only 2M | 0.590 +/- 0.539 | 0.620 +/- 0.522 | 0.361 +/- 0.338 | 803k +/- 256k, 3/5 seeds | 21.57 steps/s |
| TD3+HER+B3-Random-Support 2M | 0.600 +/- 0.548 | 0.600 +/- 0.548 | 0.426 +/- 0.392 | 570k +/- 141k, 3/5 seeds | 20.84 steps/s |
| TD3+HER+B3-Random-Fixed-Count 2M | 0.200 +/- 0.447 | 0.200 +/- 0.447 | 0.144 +/- 0.322 | 520k, 1/5 seeds | 41.60 steps/s |
| TD3+HER+B3-Shuffled-Weight 2M | 0.000 +/- 0.000 | 0.010 +/- 0.022 | 0.00005 +/- 0.00011 | none, 0/5 seeds | 20.03 steps/s |

Seed-level final success:

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only 2M | 0.95 | 0.00 | 0.00 | 1.00 | 1.00 |
| TD3+HER+B3-Random-Support 2M | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| TD3+HER+B3-Random-Fixed-Count 2M | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| TD3+HER+B3-Shuffled-Weight 2M | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Seed-level best success:

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD3+HER+B3-Critic-Only 2M | 1.00 | 0.00 | 0.10 | 1.00 | 1.00 |
| TD3+HER+B3-Random-Support 2M | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| TD3+HER+B3-Random-Fixed-Count 2M | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| TD3+HER+B3-Shuffled-Weight 2M | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 |

Implementation diagnostics:

| Metric | Value |
| --- | ---: |
| `mean_w_P_reference` | 1.8789 +/- 0.2343 |
| `reference_high_weight_rate` | 0.8789 +/- 0.2343 |
| `assigned_high_weight_rate` | 0.8789 +/- 0.2343 |
| `support_weight_alignment_rate` | 0.8846 +/- 0.2204 |
| `support_weight_overlap_rate` | 0.8188 +/- 0.3328 |
| `support_cache_hit_rate` | 0.5595 +/- 0.0529 |
| support lookup used | yes |

Interpretation:

- Shuffled-Weight fails on all 5 seeds despite preserving the SABER-generated
  weight distribution and support lookup path.
- This is a strong negative control against the explanation that support
  density alone is sufficient. In this DrawerOpen run, assigning the
  SABER-generated weights to the correct samples appears important.
- Together with Fully Random Fixed-Count, this supports the claim that the
  performance gain is not merely random critic reweighting; the support signal
  and its sample-level alignment both matter.


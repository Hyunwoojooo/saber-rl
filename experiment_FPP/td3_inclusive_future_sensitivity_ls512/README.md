# FPP TD3 Inclusive-Future HER Sensitivity

This paired sensitivity experiment tests whether the observed TD3 SABER improvement
depends on excluding the sampled transition's own next achieved goal from future-goal
HER.

The source is byte-identical to `td3_saber_critic_only_v4_ls512` except for one
replay predicate. For a sampled transition at episode step `t`, candidate transition
steps satisfy `u >= t` and the relabeled goal is the candidate's
`next_achieved_goal`. The candidate set therefore includes the current transition
and admits `m(s_{t+1})`. Every stored transition, including the final transition, is
eligible for relabeling.

## Fixed comparison

- Task: FetchPickAndPlace
- Backbone: TD3
- Arms: HER and critic-only SABER
- Paired seeds: 1--5
- Budget: 1,000,000 environment steps
- Nominal HER probability: 0.8
- Replay: uniform
- UTD: 1
- SABER support settings: V4 LS512, unchanged
- No sensitivity-specific tuning

The only intended difference from the corresponding primary experiments is the
inclusive future-goal predicate shared by both arms.

## Validation

```bash
python validate_experiment.py
python test_inclusive_future_her.py
python run.py --arm her --seed 1 --validate_only --device cuda
python run.py --arm saber --seed 1 --validate_only --device cuda
```

## Launch

```bash
MAX_PARALLEL=7 ./run_sensitivity_1m_seeds1_5_parallel.sh
```

Run outputs and launcher logs are written under
`experiment_FPP/runs/td3_inclusive_future_sensitivity_ls512/`.

## Results

All ten runs completed. Mean normalized success AUC was `0.3261` for HER and
`0.3935` for SABER. The paired SABER-minus-HER difference was `+0.0674`
(95% paired-bootstrap CI `[-0.0068, 0.1416]`; 3/5 positive pairs). Thus, the
observed mean SABER-over-HER direction was preserved under inclusive future-goal
indexing, while the five-seed interval included zero.

The compact result artifact, including exact evaluation curves, seed-level
metrics, paired statistics, and sanitized provenance, is available at
[`../paper_artifacts/fpp_td3_inclusive_future_her_sensitivity_1m/`](../paper_artifacts/fpp_td3_inclusive_future_her_sensitivity_1m/).

## Seeds 6--10 extension

The paired extension used the same training source and configuration with only
the seed cohort and non-overwriting output root changed. It is launched with:

```bash
MAX_PARALLEL=7 ./run_sensitivity_1m_seeds6_10_parallel.sh
```

All ten extension runs completed. Mean normalized success AUC was `0.3274`
for HER and `0.3388` for SABER. The paired SABER-minus-HER difference was
`+0.0114` (95% paired-bootstrap CI `[-0.0507, 0.0950]`; 2/5 positive
pairs). The interval includes zero and the seed directions are mixed.

The compact extension artifact is available at
[`../paper_artifacts/fpp_td3_inclusive_future_her_sensitivity_seeds6_10_1m/`](../paper_artifacts/fpp_td3_inclusive_future_her_sensitivity_seeds6_10_1m/).

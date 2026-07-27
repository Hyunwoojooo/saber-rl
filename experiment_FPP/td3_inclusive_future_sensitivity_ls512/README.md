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

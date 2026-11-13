# SABER

**Support-Conditioned Adaptive Bellman Error Reweighting for Sparse-Reward Hindsight Experience Replay**

SABER reweights critic losses using local replay support for the current-policy
action. The main preset retains HER relabeling, replay sampling, Bellman targets,
actor updates, network architectures and the update budget.

[Submitted paper](paper/SABER_Support_Conditioned_Adaptive_Bellman_Error_Reweighting.pdf)
| [Technical supplement](paper/technical_supplement.pdf)
| [Post-AAAI experiment archive](ICML_27/README.md)
| [Review-response handoff](ICML_27/postAAAI_Review.md)

## DrawerOpen Results

Snapshot: **2026-11-13 UTC**. This is a results overview, not a live dashboard.
It covers the current main-comparison cohorts and post-AAAI follow-ups; historical
supplementary-only studies are linked below rather than pooled into this table.

All numerical training rows use DrawerOpen-v3, 2M environment steps per run and
UTD=1. AUC is normalized success AUC on the exact 10k-2M evaluation grid; final
success is the evaluation success rate at 2M. AUC is mean +/- sample SD across
seeds; final success is a mean on the 0-1 scale. Every completed seed, including
zero-success runs, is retained.

**Paired difference = SABER AUC minus the row's method AUC within the same
cohort.** Positive values favor SABER's sample mean; the brackets contain the
reported pointwise 95% bootstrap CI, not a simultaneous guarantee. A CI crossing
zero establishes neither superiority nor equivalence. Reference rows have no
self-comparison interval. Multiplicity-adjusted intervals, where specified, are
in the linked reports.

| Cohort / backbone | Method | Seeds | AUC mean +/- SD | Final success | Paired difference [95% CI] | Results |
| --- | --- | --- | ---: | ---: | --- | --- |
| Main extension / DDPG | HER | 1-10 | 0.4011 +/- 0.3577 | 0.595 | +0.2508 [-0.0676, +0.5439] | [Data][ddpg] |
| Main extension / DDPG | SABER (Ours), critic-only | 1-10 | 0.6519 +/- 0.2459 | 0.895 | Reference | [Data][ddpg] |
| Main extension / DDPG | HGR | 1-10 | 0.5649 +/- 0.3160 | 0.750 | +0.0870 [-0.1814, +0.3445] | [Data][ddpg] |
| Main extension / DDPG | PER-full | 1-10 | 0.6504 +/- 0.3450 | 0.795 | +0.0014 [-0.1050, +0.1487] | [Data][ddpg] |
| Main extension / DDPG | Replay-Action Query | 1-10 | 0.4329 +/- 0.3765 | 0.600 | +0.2190 [-0.0915, +0.5156] | [Data][ddpg] |
| Main extension / DDPG | Shuffled-Policy Query | 1-10 | 0.6228 +/- 0.2779 | 0.895 | +0.0291 [-0.1997, +0.2373] | [Data][ddpg] |
| Main extension / DDPG | Random-Support | 1-10 | 0.5661 +/- 0.3099 | 0.800 | +0.0858 [-0.0240, +0.2198] | [Data][ddpg] |
| Main extension / DDPG | Context-Eligible Count-Matched | 1-10 | 0.5157 +/- 0.3594 | 0.700 | +0.1362 [-0.1187, +0.3739] | [Data][ddpg] |
| Follow-up / DDPG | SABER (Ours), shared Tail-512 reference | 1-10 | 0.6526 +/- 0.2484 | 0.890 | Reference | [Assignment][assignment] |
| Follow-up / DDPG | Context-eligible random assignment | 1-10 | 0.5157 +/- 0.3594 | 0.700 | +0.1369 [-0.1275, +0.3843] | [Assignment][assignment] |
| Follow-up / DDPG | Context-eligible TD-error Top-k | 1-10 | 0.5346 +/- 0.3802 | 0.700 | +0.1180 [-0.0549, +0.3249] | [Assignment][assignment] |
| Follow-up / DDPG | HER + PAL | 1-10 | 0.5454 +/- 0.3028 | 0.800 | +0.1073 [-0.1531, +0.3577] | [PAL][pal] |
| Follow-up / DDPG | SABER (Ours), Random-512 retrieval | 1-10 | 0.4302 +/- 0.3772 | 0.590 | +0.2224 [-0.0809, +0.5184] | [Retrieval][retrieval] |
| Post-AAAI / SAC | HER | 1-5 | 0.4470 +/- 0.4108 | 0.600 | +0.0015 [-0.0466, +0.0413] | [Data][sac] |
| Post-AAAI / SAC | SABER (Ours), critic-only | 1-5 | 0.4485 +/- 0.4108 | 0.600 | Reference | [Data][sac] |
| Post-AAAI / SAC | HGR | 1-5 | 0.7066 +/- 0.0659 | 1.000 | -0.2581 [-0.5965, +0.0651] | [Data][sac] |
| Post-AAAI / SAC | PER-full | 1-5 | 0.7976 +/- 0.0722 | 1.000 | -0.3491 [-0.6747, -0.0235] | [Data][sac] |
| Main extension / SAC | HER, new seeds only | 6-10 | 0.5263 +/- 0.3837 | 0.800 | +0.0153 [-0.1089, +0.1459] | [Data][sac-extension] |
| Main extension / SAC | SABER (Ours), new seeds only | 6-10 | 0.5416 +/- 0.3346 | 0.800 | Reference | [Data][sac-extension] |
| Main extension / SAC | HGR, new seeds only | 6-10 | 0.6758 +/- 0.0996 | 1.000 | -0.1342 [-0.3661, +0.0280] | [Data][sac-extension] |
| Main extension / SAC | HER, pooled | 1-10 | 0.4867 +/- 0.3771 | 0.700 | +0.0084 [-0.0598, +0.0769] | [Data][sac-extension] |
| Main extension / SAC | SABER (Ours), pooled | 1-10 | 0.4951 +/- 0.3566 | 0.700 | Reference | [Data][sac-extension] |
| Main extension / SAC | HGR, pooled | 1-10 | 0.6912 +/- 0.0812 | 1.000 | -0.1962 [-0.4134, -0.0118] | [Data][sac-extension] |
| Main extension / TD3 | HER | 1-10 | 0.1374 +/- 0.2907 | 0.195 | +0.3208 [+0.0167, +0.6030] | [Data][td3-extension] |
| Main extension / TD3 | SABER (Ours), critic-only | 1-10 | 0.4582 +/- 0.3235 | 0.695 | Reference | [Data][td3-extension] |
| Main extension / TD3 | HGR | 1-10 | 0.2438 +/- 0.3171 | 0.390 | +0.2144 [-0.0976, +0.5015] | [Data][td3-extension] |
| Main extension / TD3 | PER-full | 1-10 | 0.3950 +/- 0.3446 | 0.600 | +0.0632 [-0.1500, +0.3001] | [Data][td3-extension] |
| Offline diagnostics / DDPG | Tail / random / exhaustive retrieval; failure and saturation | 10 checkpoints; 30 training logs | Not a training comparison | N/A | Descriptive analysis only | [Report][offline] |

### Reading the Cohorts

- **Main extension / DDPG:** the submitted seeds 1-5 plus five new seeds per
  method. The eight methods' 40 new runs are complete. The artifact retains
  historical-only, added-seed-only and pooled estimates separately. All seven
  pooled AUC intervals above include zero. Historical execution-environment
  identity is not asserted where original records were unavailable.
- **Follow-up / DDPG:** the assignment experiment's SABER reference is reused
  by PAL and Random-512, so it appears only once here. It is not interchangeable
  with the main-extension SABER cohort (AUC 0.6526 vs. 0.6519). Some results or
  references overlap across artifacts; never treat these rows as extra
  independent seeds or pool different cohorts into a larger sample. The
  Random-512 contrast is sign-reversed from the source's Random-minus-Tail
  convention so every table contrast uses the same direction.
- **Random assignment is not random retrieval:** assignment controls choose
  high-weight minibatch positions; Random-512 chooses replay candidates used
  to estimate support. The count-matched assignment controls derive eligibility
  and the high-weight count from each run's own SABER rule.
- **Post-AAAI / SAC:** these are FPP-reference SAC runs with automatic
  temperature, not the older fixed-temperature prototype. HGR/PER were added
  after observing the HER/SABER results. The inherited observation includes
  original-goal coordinates alongside the relabeled goal input; all four
  methods share that representation, and its causal effect is unresolved.
- **Main extension / SAC:** HER, SABER and HGR each have five completed new
  runs (seeds 6-10). HGR completed on November 13 at 02:35 UTC. The pooled
  SABER-minus-HER AUC difference is small and both its pointwise 95% and nominal three-contrast
  adjusted interval [-0.0750, +0.0942] include zero. This establishes neither
  superiority nor equivalence. Final success is 0.700 for both methods;
  final-zero seeds 2, 3 and 8 are retained. Historical, new and pooled rows
  overlap and are not extra independent observations. HGR reaches final success
  1.0 in all ten runs and has a higher mean AUC with lower seed variability.
  The SABER-minus-HGR pointwise 95% interval is negative, but the predeclared
  three-contrast-adjusted interval [-0.4597, +0.0271] includes zero. This is
  cohort-specific evidence, not an adjusted superiority or causal claim.
  PER seeds 6-10 are still running at this snapshot and are not published;
  the PER five-seed row above is not a ten-seed comparison.
- **Main extension / TD3:** all 20 new runs (HER, SABER, HGR and PER, each
  seeds 6-10) completed on November 7 at 21:12 UTC. Historical 1-5, extension
  6-10 and pooled 1-10 results remain separate in the artifact. The pooled
  SABER-minus-HER AUC interval excludes zero; the HGR and PER intervals do not.
  This is cohort-specific evidence, not proof of the support mechanism or a
  multiplicity-adjusted superiority claim. SABER seed 8 has zero AUC and is
  retained. The original submitted [five-seed metrics][td3-metrics] and
  [paired statistics][td3-ci] are unchanged. Retained source/preset parity does
  not establish byte-for-byte historical execution-environment identity.
- **Scope:** these are cohort-specific results, not a cross-backbone ranking
  or a claim of universal superiority. Equal UTD is not equal wall-clock cost;
  runtime includes different instrumentation and concurrent workloads. See
  the artifacts for FQ-AUC, Hit90, learning curves, seed-level outcomes and costs.

### Additional DrawerOpen Studies

Historical supplementary-only experiments remain separate from the overview:
[DDPG actor scope](paper/supplementary_artifacts/actor_scope_ddpg/),
[DDPG AER without HER](metaworld_Drawer_Open_DDPG_AER/paper_artifacts/drawer_open_ddpg_aer_noher_2m/),
[TD3 AER without HER](metaworld_Drawer_Open_TD3_AER/paper_artifacts/drawer_open_td3_aer_noher_2m/),
[REDQ](metaworld_Drawer_Open_REDQ/),
[fixed-target diagnostics](metaworld_Drawer_Open_DDPG/paper_artifacts/drawer_open_ddpg_saber_fixed_target_diagnostic_all_10_2m/),
and the [technical artifact index](paper/supplementary_artifacts/technical_supplement/artifact_index.csv).
The [historical TD3 task README](metaworld_Drawer_Open_TD3/README.md) also records
older ablations and an explicitly excluded PER-style cohort; excluded results
are not included in the overview above.

## Verification

Numerical rows are transcribed from the linked published CSV/JSON artifacts,
not recomputed from rounded README values. With Python 3.12 and NumPy installed,
the following checks are read-only and do not start training:

```bash
python3 ICML_27/tools/verify_archive.py
python3 ICML_27/tools/ddpg_completed_extension_results.py verify
python3 ICML_27/tools/sac_results.py verify --task drawer_open
python3 ICML_27/tools/sac_seed_extension_results.py verify
python3 ICML_27/tools/td3_seed_extension_results.py verify
```

See the [archive guide](ICML_27/README.md) for source snapshots, provenance,
environment requirements and reuse limits. Compact results do not include all
raw logs, model checkpoints or replay buffers. Original code licensing and
preserved third-party notices are described in the [archive license section](ICML_27/README.md#license).

[ddpg]: metaworld_Drawer_Open_DDPG/paper_artifacts/drawer_open_ddpg_main_seed_extension_2m/README.md
[assignment]: ICML_27/experiments/draweropen_ddpg_saber_vs_random_vs_td_topk/README.md
[pal]: ICML_27/experiments/draweropen_ddpg_saber_vs_pal/README.md
[retrieval]: ICML_27/experiments/draweropen_ddpg_saber_tail_vs_random512/README.md
[sac]: metaworld_Drawer_Open_SAC/paper_artifacts/drawer_open_sac_her_saber_hgr_per_2m/README.md
[sac-extension]: metaworld_Drawer_Open_SAC/paper_artifacts/drawer_open_sac_main_seed_extension_2m/README.md
[td3-metrics]: paper/supplementary_artifacts/primary_seed_results/aggregate_metrics.csv
[td3-ci]: paper/supplementary_artifacts/primary_seed_results/paired_statistics.csv
[td3-extension]: metaworld_Drawer_Open_TD3/paper_artifacts/drawer_open_td3_main_seed_extension_2m/README.md
[offline]: ICML_27/experiments/draweropen_ddpg_retrieval_and_failure_diagnostics/results/outputs/final/REPORT.md

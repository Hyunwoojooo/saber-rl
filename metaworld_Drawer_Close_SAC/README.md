# MetaWorld DrawerClose SAC Experiments

The post-AAAI FPP-reference SAC comparison is complete: **HER, SABER (Ours),
HGR and full PER, seeds 1-5, 2M steps each (20 unique runs)**.

See [the compact results and protocol](paper_artifacts/drawer_close_sac_her_saber_hgr_per_2m/README.md)
for seed-level metrics, full evaluation curves, paired CIs, execution environments,
source provenance and SHA256 verification. All seeds, including collapsed and
zero-final-success runs, are retained. An early Hit90 does not imply stable success.

Exact executed auto-temperature SAC code is preserved in the
[HER/SABER source](../ICML_27/experiments/draweropen_drawerclose_sac_her_vs_saber/source/)
and [HGR/PER source](../ICML_27/experiments/draweropen_drawerclose_sac_hgr_per/source/).
No generic or fixed-temperature prototype has been substituted. The inherited
observation/goal representation caveat is documented with the results.

```bash
python3 ICML_27/tools/sac_results.py verify --task drawer_close
```

Run verification from the repository root with Python 3.12 and NumPy. It does not train a model.

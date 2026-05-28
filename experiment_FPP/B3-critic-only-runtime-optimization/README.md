# B3-critic-only-runtime-optimization

Purpose: Exact-semantics runtime-optimization path for critic-only B3 support weighting.

Source spec: `SABER/SABER.md`.

Dry-run one seed:

```bash
python3 SABER/FetchPickAndPlace/B3-critic-only-runtime-optimization/run.py --seed 1 --dry-run
```

Smoke run:

```bash
python3 SABER/FetchPickAndPlace/B3-critic-only-runtime-optimization/run.py --seed 1 --smoke --device auto
```

Full first-pass run for all document seeds:

```bash
bash SABER/FetchPickAndPlace/B3-critic-only-runtime-optimization/run_all_seeds.sh --device auto
```

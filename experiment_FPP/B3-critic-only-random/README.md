# B3-critic-only-random

Purpose: Positive-count control: random K high-weight critic samples with B3 diagnostics retained.

Source spec: `SABER/SABER.md`.

Dry-run one seed:

```bash
python3 SABER/FetchPickAndPlace/B3-critic-only-random/run.py --seed 1 --dry-run
```

Smoke run:

```bash
python3 SABER/FetchPickAndPlace/B3-critic-only-random/run.py --seed 1 --smoke --device auto
```

Full first-pass run for all document seeds:

```bash
bash SABER/FetchPickAndPlace/B3-critic-only-random/run_all_seeds.sh --device auto
```

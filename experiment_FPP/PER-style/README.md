# PER-style

Purpose: PER-style TD-error priority-derived critic loss weighting with uniform replay sampling.

Source spec: `SABER/SABER.md`.

Dry-run one seed:

```bash
python3 SABER/FetchPickAndPlace/PER-style/run.py --seed 1 --dry-run
```

Smoke run:

```bash
python3 SABER/FetchPickAndPlace/PER-style/run.py --seed 1 --smoke --device auto
```

Full first-pass run for all document seeds:

```bash
bash SABER/FetchPickAndPlace/PER-style/run_all_seeds.sh --device auto
```

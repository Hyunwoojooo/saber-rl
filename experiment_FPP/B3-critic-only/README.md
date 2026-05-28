# B3-critic-only

Purpose: SABER main candidate: support reliability weighting on critic loss only.

Source spec: `SABER/SABER.md`.

Dry-run one seed:

```bash
python3 SABER/FetchPickAndPlace/B3-critic-only/run.py --seed 1 --dry-run
```

Smoke run:

```bash
python3 SABER/FetchPickAndPlace/B3-critic-only/run.py --seed 1 --smoke --device auto
```

Full first-pass run for all document seeds:

```bash
bash SABER/FetchPickAndPlace/B3-critic-only/run_all_seeds.sh --device auto
```

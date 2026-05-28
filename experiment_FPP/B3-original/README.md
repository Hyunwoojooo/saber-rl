# B3-original

Purpose: Required actor-critic B3 support-only reference.

Source spec: `SABER/SABER.md`.

Dry-run one seed:

```bash
python3 SABER/FetchPickAndPlace/B3-original/run.py --seed 1 --dry-run
```

Smoke run:

```bash
python3 SABER/FetchPickAndPlace/B3-original/run.py --seed 1 --smoke --device auto
```

Full first-pass run for all document seeds:

```bash
bash SABER/FetchPickAndPlace/B3-original/run_all_seeds.sh --device auto
```

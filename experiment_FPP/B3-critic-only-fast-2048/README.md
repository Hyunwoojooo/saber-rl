# B3-critic-only-fast-2048

Purpose: FetchPickAndPlace local_search_max sensitivity run at 2048.

Source spec: `SABER/SABER.md`.

Dry-run one seed:

```bash
python3 SABER/FetchPickAndPlace/B3-critic-only-fast-2048/run.py --seed 1 --dry-run
```

Smoke run:

```bash
python3 SABER/FetchPickAndPlace/B3-critic-only-fast-2048/run.py --seed 1 --smoke --device auto
```

Full first-pass run for all document seeds:

```bash
bash SABER/FetchPickAndPlace/B3-critic-only-fast-2048/run_all_seeds.sh --device auto
```

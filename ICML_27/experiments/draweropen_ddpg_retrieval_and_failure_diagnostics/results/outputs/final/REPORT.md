# Post-PAL retrieval and failure diagnostics

This report contains no new training runs. All analyses are descriptive.

## Retrieval

Ten SABER final checkpoints; 1024 fixed HER queries per seed; five random retrieval repeats.
Tail parity and direct full-replay spot checks passed for every checkpoint.

| Method | Support rate (seed mean) | Miss rate given full positive (seed mean) |
| --- | ---: | ---: |
| tail | 0.537109 | 0.410458 |
| random | 0.625352 | 0.311727 |
| full | 0.902734 | 0.000000 |

See retrieval/seed_summary.csv and random_repeat_summary.csv for denominators and variation.
Timing is for a shared CPU reference implementation, not production Numba throughput.

## Failure and saturation

Zero evaluation success: 7/30 runs; nonzero evaluation success: 23/30.
Any saturation among zero/nonzero runs: 7/7 and 0/23 respectively.
These outcome-defined groups do not establish that saturation caused failure.
First saturation is known only to a logged update window; training and evaluation success are reported separately.
Saturation selects all eligible batch positions, not necessarily the entire minibatch.

## Limits

- Final-checkpoint search agreement does not establish online learning equivalence.
- Exhaustive matching uses the same discretized context, not ground truth continuous support.
- Neither query counts nor random repetitions increase the number of independent training seeds.
- The original paired AUC intervals are retained unchanged; no new confirmatory tests were selected.
- Latest/best retention cannot reconstruct all earlier policies and replay buffers.

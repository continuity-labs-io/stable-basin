# Prompt: Gate 1 day census (reconnaissance)

## Goal

Find out whether the fs369 recording contains a pair of adjacent days that both run the
32-block stimulation sweep and neither of which is preceded by uncontrolled stimulation.
We have only ever examined 2025-02-20 and 2025-02-21. There are about 21 days in the file.

This is reconnaissance, not a measurement. It answers one yes/no question and stops.

## Rules

- **Line budget: 90 lines of Python, hard.** If you are over, you are doing something
  that was not asked for.
- **No scaffolding.** No tests, no config file, no Makefile target, no JSON sidecar,
  no plots, no argparse, no logging, no classes, no docstrings beyond one line at the top.
  `analysis_standards.md` governs measurements. This is not a measurement.
- Write it as `scripts/day_census.py`. It is throwaway and will not be committed as
  library code.
- Print to stdout only. Write no files.

## Input

Read **only** the `fs369_wholelife_stimulations` table. Do not open the events table and
do not open the raw HDF5. The stimulations table is small, so a single full read is fine.

Take the store path and the read pattern from the constants already in
`scripts/gate1_run.py`. Do not invent a path.

Columns of interest: `time_of_stim`, `electrode`, `a1`, `a2`, `d1`, `d2`,
`nb_stim_pulse`, `pulse_train_period`, `stim_polarity`, `stim_shape`.

## What to compute

1. **Collapse to unique timestamps.** Stimulation is whole-array, so roughly 32 electrode
   rows share each timestamp. Reduce to one row per unique `time_of_stim`, carrying the
   parameter tuple `(a1, a2, d1, d2, nb_stim_pulse, pulse_train_period, stim_polarity,
   stim_shape)`. If the tuple is not constant across electrodes at a given timestamp,
   print a line saying so and carry the first tuple.

2. **Segment into blocks.** Sort unique timestamps ascending. A gap of more than 15 s
   starts a new block. Record each block's start time, stimulus count, and parameter tuple.

3. **Define the sweep tuple set empirically.** Take the set of distinct parameter tuples
   that appear on 2025-02-21. That day is known to be the clean sweep. Call that set
   `SWEEP_TUPLES`. Do not hardcode amplitudes or durations.

4. **Per calendar day**, in local file time, report:
   - total unique stimulation timestamps
   - total blocks
   - number of blocks whose tuple is in `SWEEP_TUPLES`
   - timestamp of the first sweep block on that day
   - number of **non-sweep** stimulation timestamps in the 24 hours immediately preceding
     that first sweep block. Compute this in absolute time, not by calendar day, because
     pre-conditioning can cross midnight.

5. **Mark a day CLEAN** if it has at least 30 sweep blocks and zero non-sweep stimulation
   timestamps in the preceding 24 hours.

6. **Find adjacent clean pairs.** Two clean days exactly one calendar day apart.
   Disqualify any pair that straddles the medium change at `2025-02-28T14:01:34`.

## Output

One line per day, then the verdict block. Exact format, so it stays greppable:

```
DAY 2025-02-19  stims 0  blocks 0  sweep 0  pre24h_nonsweep 0  CLEAN no
DAY 2025-02-20  stims 2360  blocks 33  sweep 32  pre24h_nonsweep 2040  CLEAN no
DAY 2025-02-21  stims 320  blocks 32  sweep 32  pre24h_nonsweep 0  CLEAN yes
...
SWEEP TUPLES: 32
CLEAN DAYS: 2025-02-21, 2025-02-23
ADJACENT CLEAN PAIRS: 2025-02-22/2025-02-23
MEDIUM CHANGE: 2025-02-28T14:01:34
VERDICT: MATCHED PAIR FOUND 2025-02-22/2025-02-23
```

If there is no adjacent clean pair, the last line is exactly:

```
VERDICT: NO MATCHED PAIR
```

Still print `CLEAN DAYS` in that case, so we can see whether non-adjacent clean days exist.

## Stop condition

Print the verdict and exit. Do not rerun any analysis, do not open `gate1_run.py`'s
pipeline, do not compute recovery times, do not search for near-miss pairs, do not
propose next steps in the output. The verdict line is the deliverable.

## Sanity checks to report inline

Two numbers we already know. If either disagrees, say so in one line rather than
adjusting the code to make them match:

- 2025-02-20 should have about 2,360 unique stimulation timestamps.
- 2025-02-21 should have about 320, in 32 blocks of 10.

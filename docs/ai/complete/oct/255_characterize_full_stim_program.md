# Prompt — Characterize the full stimulation program

Reconnaissance script. Throwaway. No config, no sidecar, no tests, no Makefile
target. Print to stdout.

**Budget: 70 lines excluding imports. If you exceed it, stop and say what is
making it long.**

## Background

We assumed a day contained 10 stimulations. It contains 2,360 unique
stimulation timestamps on 2025-02-20. Our CANDIDATE set is a small island
inside an all-day program we have never looked at. Before choosing what to
analyze, describe the program.

## File

`scripts/stim_program.py`.

## Input

`data/ephys/finalspark/fs369data/fs369_package.hdf5`, table
`fs369_wholelife_stimulations` only.

**Do not open the events table. Do not open `fs369_raw.hdf5`.** The stimulation
table is about 130k rows and loads whole.

Days: 2025-02-20 and 2025-02-21.

## Definitions

A **stimulation event** is one unique `time_of_stim`. Several electrode rows
share it.

The **parameter tuple** is `(a1, a2, d1, d2, nb_stim_pulse,
pulse_train_period, stim_polarity, stim_shape)`.

A **block** is a run of consecutive stimulation events, in time order, whose
parameter tuple does not change. A new block starts whenever the tuple changes.

## Report, per day

**Overview:** number of stimulation events, first and last timestamp, number of
distinct parameter tuples, number of blocks, and the distribution of electrodes
per event (min, median, max).

**Parameter tuples:** one row each, with the number of events, number of
blocks, first and last timestamp. Sort by event count descending.

**Block structure:** number of blocks, and the distribution of block sizes
(min, median, max). Then the median within-block gap and the median
between-block gap, in seconds.

**The block sequence:** one row per block with index, parameter tuple, number
of events, start, end, within-block median gap, and gap to the next block. If
there are more than 40 blocks, print the first 20 and the last 20 and say how
many were omitted.

**Quiet gaps:** every gap between consecutive stimulation events longer than 60
seconds, sorted descending, top 10, with start, end and duration. Also the
total number of such gaps and the total quiet seconds they account for.

**CANDIDATE in context:** locate the CANDIDATE block
(`a1=0.8, a2=0.8, d1=300, d2=300, nb_stim_pulse=2, pulse_train_period=10000,
stim_polarity=0, stim_shape=0`) and print the 3 blocks before it and the 3
after, so we can see what the tissue experienced on either side.

## Do the two days match

Compare the two days' block sequences as lists of parameter tuples. Report
whether the sequences are identical, and if not, where they first diverge and
how many blocks each day has.

## Verdict lines

End with exactly five lines, nothing after:

```
EVENTS PER DAY: <n on 02-20>, <n on 02-21>
PARAMETER TUPLES: <n on 02-20>, <n on 02-21>
BLOCKS: <n on 02-20>, <n on 02-21>
LONGEST QUIET GAP: <seconds> on <day> starting <time>
DAYS MATCH / DAYS DIFFER at block <i>
```

## Constraints

- Describe, do not decide. No recommendation about which stimulations to use.
- No events, no rates, no plots.
- Never substitute a default for something read from the file.

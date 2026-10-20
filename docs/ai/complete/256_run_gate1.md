# Prompt — Run Gate 1

Implements `docs/ai/design/3_recovery_profiling_gate1.md`. This one produces a
result, so it writes a tidy table and a JSON sidecar. It still gets no config
file, no Makefile target and no unit tests; that debt is deliberate and is paid
only if Gate 1 passes.

**Budget: 250 lines excluding imports. If you exceed it, stop and say what is
making it long.**

## File

`scripts/gate1_run.py`. Outputs to `output/gate1/`.

## Input

`data/ephys/finalspark/fs369data/fs369_package.hdf5`. Never open
`fs369_raw.hdf5`.

Days: 2025-02-20 and 2025-02-21.

## Structure, from the stimulation table

Each day runs a 32-block sweep. A block is a run of consecutive unique
`time_of_stim` values sharing the parameter tuple `(a1, a2, d1, d2,
nb_stim_pulse, pulse_train_period, stim_polarity, stim_shape)`, 10 stimuli at
10 s spacing.

02-20 has one extra leading block (amplitude 2.5, 2,040 events, 1 electrode
each) that is **not** part of the sweep. Exclude any block whose events have
fewer than 32 electrodes.

**Match blocks across days by parameter tuple.** Expect 32 matched pairs. If
the count differs, report it and stop.

## Events, four queries total

Per day, two bounded `HDFStore.select` calls on `fs369_wholelife_events`:

1. **Baseline window:** the 30 minutes ending at the day's first sweep stimulus.
2. **Sweep window:** first sweep stimulus to last sweep stimulus plus 10 s.

Do not load the events table whole. Do not query per stimulus.

## Baseline and electrode selection

Per electrode per day, baseline rate = events in the baseline window divided by
1800 s.

**Active set** = electrodes above 0.1 Hz baseline on *both* days. Report the
count; the design expects about 19. All results below use only these.

## The measurement

For each (electrode, block, day):

Pool the 10 trials. Collect events from 3 ms to 10 s after each stimulus.
Histogram on about 40 log-spaced bins over that range. Rate per bin = count /
(10 trials x bin width).

Compute four numbers:

- `t_return` — the centre of the **last** bin whose rate exceeds
  `1.5 x baseline`. If no bin exceeds it, 0. If the final bin exceeds it,
  record 10.0 and flag censored.
- `t_half_peak` — time from `t_peak` to the first later bin at or below half of
  `peak_rate`. Censored at 10.0 if never reached. This measure does not depend
  on the baseline.
- `peak_rate` and `t_peak`.
- `residual` — mean rate over the last 1 s, as a multiple of baseline.

**Report the censoring rate for both `t_return` and `t_half_peak`.** If one is
censored in more than 30% of cells it has no usable variance; say so and use
the other as the primary measure. Record which was used and why.

Everything below runs on the chosen primary measure, and is also reported for
the other.

## Step 1 — reliability ceiling

Within each (electrode, block, day), split the 10 trials odd and even, compute
the measure on each half, and correlate across all electrode-block cells.
Apply the Spearman-Brown correction. Report per day.

**If the corrected ceiling is below 0.7, print `CEILING TOO LOW` and continue
anyway**, reporting Gate 1 as uninterpretable rather than failed.

## Gate 1 — three correlations

1. **Per block.** For each of the 32 matched blocks, Pearson correlation of the
   electrode vector across days. Report all 32, plus median and range.
2. **Condition-centered pooled.** Flatten to one value per electrode-block per
   day, subtract the block mean within each day, correlate the residuals across
   days. **This is the headline.**
3. **Dose slope.** Per electrode, regress the measure on `a1` across blocks
   where `a1 == a2`, then correlate the slopes across days.

**PASS: correlation 2 is >= 0.7.**

## Null and positive control

**Null:** permute electrode labels on day 2, recompute correlation 2, 1,000
times. Report the 95th percentile and where the observed value sits.

**Positive control:** take the day 1 values, build a synthetic day 2 by adding
Gaussian noise scaled so that the synthetic split-half matches the observed
ceiling, and confirm correlation 2 recovers a value near the ceiling. This is
the check that makes a null interpretable. Roughly 15 lines.

## Output

- `output/gate1/cells.csv` — one row per electrode, block, day, with all four
  measures, the censoring flags, baseline, and trial count.
- `output/gate1/gate1.json` — every parameter, the seed, the chosen primary
  measure and the reason, the ceiling, all three correlations, the null
  percentile, the positive control result.
- stdout: a short report ending with exactly these lines, nothing after:

```
PRIMARY MEASURE: <t_return|t_half_peak> (<reason>)
ACTIVE ELECTRODES: <n>
MATCHED BLOCKS: <n>
CEILING: <r> (day1 <r>, day2 <r>)
PER BLOCK r: median <r> range <lo>-<hi>
CONDITION-CENTERED r: <r>  null p95 <r>  positive control <r>
GATE 1: PASS | FAIL | UNINTERPRETABLE
```

## Constraints

- Never substitute a default for something read from the file.
- No plots.
- Do not compute anything across non-adjacent days, and do not touch Gate 2 or
  Gate 3.
- Report `n = 1 culture, <n> electrodes, 32 blocks, 2 sessions, 10 trials` in
  the stdout report.

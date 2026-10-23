# Prompt: Amplitude vs timing reliability on the Sharf dataset (reconnaissance)

## Goal

On fs369 we found that amplitude measures are reliable within a session regardless of
timescale, while timing measures mostly are not, with only the slow-band `t_return`
surviving. Find out whether that holds on a second substrate and a second instrument.

One question: **does amplitude beat timing on within-session split-half reliability, and
does fast-band timing fail here too?**

This is reconnaissance. It answers that question and stops.

## Rules

- **Line budget: 200 lines of Python, hard.**
- **No scaffolding.** No tests, no config, no Makefile target, no sidecar, no plots, no
  argparse, no classes. `analysis_standards.md` governs measurements; this is a look.
- Write it as `scripts/sharf_amp_vs_timing.py`. Throwaway.
- Print to stdout only. Write no files.
- Do not modify `scripts/gate1_run.py` or `scripts/timescale_spike.py`.

## The comparison only works if the measures are identical

This exists to be compared against the fs369 numbers. Any difference in measure definition
makes the comparison meaningless. Import from `gate1_run.py` and `timescale_spike.py`:
`EDGES`, `WIDTHS`, `POST_LO_S`, `POST_HI_S`, `measures`, `centered_r`, `pearson`,
`spearman_brown`, `WINDOWS`, `BANDS`. Write new code only for loading Sharf data and
finding its blocks.

If a constant or function cannot be imported, copy it verbatim and print a NOTE saying
which and why. Do not re-derive or improve any of them.

## Data

Sharf dataset, already on disk. Find it, do not download anything.

**Inventory first, and print what you find before computing anything:**

1. List the Sharf files and identify which contain stimulation events.
2. For each candidate recording session, print: session identifier, number of stimulation
   events, number of distinct stimulus parameter conditions, trials per condition, number
   of recording electrodes, and whether a quiet pre-stimulus period exists.
3. Pick the single session with the most trials per condition, requiring **at least 8
   trials per condition and at least 8 active electrodes.**

**Hard stop.** If no session has 8 trials per condition and 8 active electrodes, print
`STOP: no session supports split-half` with the best candidate's numbers, and exit. Do not
relax the thresholds. Do not pool across sessions to reach them.

Prefer the `Drug_*` recordings if they qualify, since those carry repeated stimulation. The
`Interval_*` files are a cross-session comparison and are not what this asks for.

## Baseline

Define baseline as the event rate in a quiet window immediately before the first stimulus
of each block. Use the same duration `gate1_run.py` uses if that much quiet exists;
otherwise use the longest quiet window available and **print the duration you used**. If
stimulation falls inside the chosen baseline window, print `STOP: no clean baseline` and
exit.

## What to compute

The same eight measures, per electrode-block cell, from pooled per-trial latencies:

`t_return`, `t_half_peak`, `peak_rate`, `residual`, `t_peak`, `early_amp`, `hump_amp`,
`late_amp`

Split-half: odd-indexed against even-indexed trials inside each cell. Correlate across
cells with condition-centering. Spearman-Brown correct with `2r / (1 + r)`. Same procedure
as `timescale_spike.py`.

## Output

Same MEASURE line format as `timescale_spike.py` so the two runs sit side by side, with one
addition: a `kind` field, `amp` or `timing`.

```
SESSION <id>  electrodes N  conditions N  trials/condition N  baseline_s X
MEASURE early_amp     kind amp     band fast  ceiling 0.xxx  undefined_frac 0.xx
MEASURE t_return      kind timing  band slow  ceiling 0.xxx  undefined_frac 0.xx
...
KIND MEDIAN CEILING: amp 0.xx  timing 0.xx
BAND MEDIAN CEILING (timing only): fast 0.xx  mid n/a  slow 0.xx
FS369 REFERENCE:     amp 0.84  timing 0.34   fast-timing 0.25  slow-timing 0.76
VERDICT: <holds|does not hold> — amplitude <beats|does not beat> timing
VERDICT: fast-band timing <fails|works> here
```

Kinds: `amp` for `peak_rate`, `early_amp`, `hump_amp`, `late_amp`. `timing` for `t_return`,
`t_half_peak`, `t_peak`. `residual` is a ratio, print it with kind `ratio` and exclude it
from both medians.

Band median over timing measures only. That is the lesson from the fs369 run: mixing
amplitude and timing measures inside a band median produced a misleading verdict.

Flag any measure undefined for more than 30 percent of cells with ` UNUSABLE` and exclude
it from the medians.

## Stop condition

Print the two verdict lines and exit. Do not compute cross-session correlations, do not
run Gate 1, do not examine the `Interval_*` drift files, do not test additional measures,
do not propose next steps in the output.

## What would make this untrustworthy

If the Sharf recording has no quiet pre-stimulus baseline, or fewer than 8 trials per
condition, the comparison does not hold and the honest output is the STOP line. A result
produced by relaxing either threshold is worse than no result.

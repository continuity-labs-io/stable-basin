# Prompt: Timescale assessment spike (reconnaissance)

## Goal

Decide whether the recovery measurement carries its reproducible signal in the
millisecond band, the sub-second band, or the seconds band. Answer by computing
within-session split-half reliability for every candidate measure and comparing ceilings.

This is reconnaissance. It answers one question and stops.

## Rules

- **Line budget: 200 lines of Python, hard.**
- **No scaffolding.** No tests, no config, no Makefile target, no JSON sidecar, no plots,
  no argparse, no classes. `analysis_standards.md` governs measurements; this is a look at
  data we already have.
- Write it as `scripts/timescale_spike.py`. Throwaway.
- Print to stdout only. Write no files.
- **Do not modify `scripts/gate1_run.py`.** Import from it.

## Why this cannot read cells.csv alone

Split-half reliability requires splitting the trials inside each electrode-block cell.
`cells.csv` is already aggregated across the 10 trials, so the trial-level structure is
gone. Recompute latencies from source.

## Reuse, do not rewrite

Import from `scripts/gate1_run.py` (it has a main guard): the store path and table keys,
the block/sweep detection (`sweep_blocks` or equivalent), the per-stimulus latency
extraction (`latencies` or equivalent), the binning edges, the active-electrode selection,
and the two matched days. Only the measure definitions and the split-half loop are new
code.

If a needed function is not importable as written, copy the minimum necessary and say so
in one printed line. Do not refactor `gate1_run.py`.

## Measures

Compute each per electrode-block cell from that cell's pooled latencies. Carry the two
that already exist so they act as a control.

**Existing (control):**
- `t_return` — last log-spaced bin exceeding 1.5x baseline
- `t_half_peak` — time to fall to half of peak rate
- `peak_rate` — maximum binned rate
- `residual` — rate in the final bin, baseline-subtracted

**New:**
- `t_peak` — center time of the highest-rate bin
- `early_amp` — mean rate over 0.003 to 0.060 s, baseline-subtracted
- `hump_amp` — mean rate over 0.3 to 2.0 s, baseline-subtracted
- `late_amp` — mean rate over 2.0 to 10.0 s, baseline-subtracted

## Band assignment

Print this alongside each measure. Hardcode it, it is a label not a computation.

- **fast** (3 to 60 ms): `t_peak`, `early_amp`, `peak_rate`, `t_half_peak`
- **mid** (0.3 to 2 s): `hump_amp`
- **slow** (2 to 10 s): `t_return`, `late_amp`, `residual`

## Split-half procedure

For each day independently:

1. Split the 10 trials in each block into odd-indexed and even-indexed halves.
2. Compute every measure on each half separately, giving two values per cell.
3. Correlate the two halves across all cells, using the same condition-centering that
   `gate1_run.py` applies before its pooled correlation.
4. Apply the Spearman-Brown correction: `r_full = 2r / (1 + r)`.
5. That corrected value is the measure's ceiling for that day.

Use the same seed and the same active-electrode set as the Gate 1 run.

## Output

One line per measure, then a verdict block. Exact format, so it stays greppable:

```
MEASURE t_return      band slow  ceiling_d1 0.795  ceiling_d2 0.762  undefined_frac 0.21
MEASURE t_half_peak   band fast  ceiling_d1 0.161  ceiling_d2 0.151  undefined_frac 0.00
MEASURE peak_rate     band fast  ceiling_d1 ...    ceiling_d2 ...    undefined_frac ...
...
CELLS: 24 electrodes x 32 blocks x 2 days
BAND MEDIAN CEILING: fast 0.xx  mid 0.xx  slow 0.xx
VERDICT: <fast|mid|slow> carries the reproducible signal
```

Flag any measure undefined for more than 30 percent of cells by appending ` UNUSABLE` to
its line, and exclude it from the band median.

## Sanity check, report inline

`t_return` should come back near 0.779 and `t_half_peak` near 0.156, matching the Gate 1
run. Those two are the control for the new split-half loop. If either disagrees by more
than 0.05, print a line saying so and do not adjust the code to make them match. A
mismatch means the reimplementation is wrong and the new measures cannot be trusted.

## Stop condition

Print the verdict and exit. Do not rerun Gate 1, do not compute cross-session
correlations, do not propose next steps in the output, do not explore measures not listed
above. The verdict line is the deliverable.

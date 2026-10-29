# Prompt — Move Gate 1 analysis logic out of the script

Step 1 of the Gate 1 follow-up plan. This step only moves code. It adds no new
analysis, no config file and no Makefile target. The double-exponential fit, the
polarity comparison and the config file are separate later prompts.

## Why

The analysis functions in `scripts/gate1_run.py` (prompt 256) can't be unit
tested: `scripts/` is not a package, and `sys.path` hacks are forbidden. Moving
them into `src/` makes them importable. They also become the place where later
Gate 1 work will be added.

## Changes

### [NEW] `src/benchmarks/recovery/gate1.py`

Moved from `scripts/gate1_run.py`: `sweep_blocks`, `latencies`, `measures`,
`pearson`, `centered_r`, `analyse`, and `spearman_brown`, which was a nested
function inside `analyse` and is now a top-level function.

- Every parameter is passed as an argument. The module has no constants of its own.
  - `sweep_blocks(rows, tuple_cols, min_electrodes)`
  - `latencies(..., post_lo_s, post_hi_s)`
  - `measures(trials, baseline, edges, return_x_baseline, half_peak_frac, residual_s)`
  - `analyse(mat, half, days, sym_a1, sym, rng, n_perm, null_pctl)`
- `measures` derives bin centres, widths and the window end from `edges`.
- `sweep_blocks` raises `ValueError` instead of calling `sys.exit`, and returns the
  total block count so the caller can print the census.
- `sweep_blocks` returns empty results instead of crashing when no block survives.
- `centered_r` raises `ValueError` if the two arrays have different shapes.
- Otherwise the code is unchanged.

### [MODIFY] `scripts/gate1_run.py`

- Keeps the parameter values, the HDF5 queries, the output writing and the
  stdout report.
- `sweep_blocks(day, rows)`, `latencies(ev, electrodes, stim_times)` and
  `measures(trials, baseline)` stay in the script, with their **original
  signatures**, as thin wrappers that pass this run's parameters to the library.
  `pearson`, `centered_r`, `CENTERS` and `WIDTHS` are still exported. Reason:
  `scripts/timescale_spike.py` (prompt 258) and `scripts/sharf_amp_vs_timing.py`
  (prompt 259) import these names from `gate1_run`, and must not change.
- Docstring updated: unit tests exist now; the config file and Makefile target
  are still missing.

### [NEW] `tests/benchmarks/recovery/test_gate1.py`

16 tests, all on synthetic data, each with ARRANGE / ACT / ASSERT blocks:

- `measures`
  - `t_return` is the centre of the last bin above threshold.
  - `t_return` is 0 when no bin exceeds the threshold.
  - `t_return` is censored at 10 s, and `residual` is correct.
  - `t_half_peak` is computed correctly.
  - `t_half_peak` is censored when the peak is in the final bin.
- `spearman_brown`: known values.
- `pearson`: matches NumPy; returns NaN on constant input.
- `centered_r`: unchanged when a constant is added to each block; raises on shape mismatch.
- `sweep_blocks`: drops blocks below the electrode minimum; raises on a stimulus with two tuples; raises on a tuple repeated across kept blocks.
- `latencies`: window includes its start and excludes its end.
- `analyse`
  - Finds a shared electrode effect above the null.
  - Finds nothing beyond the null when electrode effects are independent.

## Verification (2026-10-29)

- `python scripts/gate1_run.py` on FS369. Compared with outputs saved before the move:
  - `output/gate1/stdout.txt` and `output/gate1/cells.csv` are byte-identical.
  - `output/gate1/gate1.json` is identical except `code_version`.
  - Still `GATE 1: FAIL`, condition-centered r 0.128.
- `python scripts/timescale_spike.py`: stdout is identical to a run against the
  pre-move `gate1_run.py`.
- `scripts/sharf_amp_vs_timing.py` and `scripts/day_census.py` import without error.
- `pytest tests/benchmarks/recovery/`: 77 passed.

## Known, not changed here

- The censored flags returned by `measures` are NumPy booleans, not Python `bool`.
  The tests allow for this.
- The JSON key `dose_slope_r` uses a word restricted by naming rule 2. Not renamed.
- `scripts/gate1_1a.py` is marked THROWAWAY and is still present.

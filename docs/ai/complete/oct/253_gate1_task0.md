# Prompt — Gate 1 Step 0: resolve the four unknowns

Write one script that answers the four open questions in
`docs/ai/design/3_recovery_profiling_gate1.md` Step 0, plus one more that was
implied but not written down. It reports evidence and proposes values. It does
not decide anything and it does not implement any part of the measurement.

Read `docs/ai/design/analysis_standards.md` first. The output contract there
applies: a tidy table plus a JSON sidecar with code version, every parameter,
and the seed.

## Input

`data/ephys/finalspark/fs369data/fs369_package.hdf5` only. The 60 GB
`fs369_raw.hdf5` is not used and must not be opened.

Tables:
- `fs369_wholelife_stimulations` — 130,113 rows, one per trigger. Columns
  include `electrode`, `time_of_stim`, `a1`, `a2`, `d1`, `d2`,
  `nb_stim_pulse`, `pulse_train_period`, `stim_polarity`, `stim_shape`.
- `fs369_wholelife_events` — one row per detected spike. Columns `electrode`,
  `time_of_event`, `max_voltage_uv`. Large; never load it whole.
- `fs369_wholelife_metadata` — one row.
- `fs369_wholelife_incubator_door_opening` — `time`, `incubator_id`,
  `incubator_door_opening` (True = opening).

The candidate parameter set, referred to below as CANDIDATE:

```
a1=0.8, a2=0.8, d1=300, d2=300, nb_stim_pulse=2,
pulse_train_period=10000, stim_polarity=0, stim_shape=0
```

Expected to be 1,294 stimulations across 32 electrodes and 5 distinct days.
Verify that count; if it differs, report the difference and continue.

## Layout

```
src/benchmarks/recovery/step0.py
configs/recovery_gate1.yaml
output/recovery_gate1/step0_report.md
output/recovery_gate1/step0.json
output/recovery_gate1/stim_windows.csv
output/recovery_gate1/artifact_psth.csv
output/recovery_gate1/artifact_psth.png
tests/benchmarks/recovery/test_step0.py
```

Makefile target:

```make
gate1-step0:
	python -m src.benchmarks.recovery.step0 --config configs/recovery_gate1.yaml
```

## Data access rule

The events table is tens of millions of rows. Do not load it whole and do not
issue one query per stimulation.

For each calendar day that CANDIDATE touches, issue **one** `HDFStore.select`
on `fs369_wholelife_events` with a `where` clause bounding `time_of_event` to
that day's span plus a margin, process every stimulation on that day in memory,
then release it. Five days means five queries.

Record the per-day row counts in the sidecar.

## Task 0a — which days, and is there an adjacent pair

Filter stimulations to CANDIDATE. Group by UTC calendar day.

Report, one row per day:
- calendar date
- days since the first recording in the metadata `Date` field
- n stimulations
- n distinct electrodes
- first and last `time_of_stim` on that day

Then:
- the list of days
- every pair of days exactly one calendar day apart
- whether the medium change at `2025-02-28T14:01:34` falls between any pair
- whether any door opening falls inside any day's stimulation span

**Verdict line:** `ADJACENT PAIR: <date>, <date>` naming the best pair, or
`NO ADJACENT PAIR` with the two closest days and their separation in days.

A pair is disqualified if the medium change falls between them.

## Task 0b — where the stimulation artifact ends

This is the highest-risk unknown. Events are threshold crossings, so each
stimulus produces an artifact that is recorded as spikes. Charge recovery is
enabled, so the artifact may outlast the pulse.

For every CANDIDATE stimulation, collect events in `[-50 ms, +500 ms]` relative
to `time_of_stim`, and split them three ways:

1. events on the **stimulated electrode itself**
2. events on **other electrodes in the same slot** (see note below)
3. events on **all other electrodes**

Build a peri-stimulus time histogram for each, pooled over all stimulations,
with variable bins: 0.1 ms from -5 to +20 ms, 1 ms from +20 to +100 ms, 10 ms
from +100 to +500 ms. Use the same bin edges for the pre-stimulus side so the
baseline rate is comparable.

Note on slots: the setup is `4x8` and electrode ids run 64 to 95. Assume slot
membership is contiguous in id (64-71, 72-79, 80-87, 88-95) and **state that
assumption in the sidecar as unverified**. The metadata says 12 neurospheres
across 4 slots, so slot is not the same as neurosphere, and nothing downstream
should treat slot as a tissue unit.

Also compute, in the same bins, the median and 95th percentile of
`max_voltage_uv`. The artifact should sit at or near the detection rail;
biological events should not.

**Proposed blanking window:** the first bin after t=0 at which the pooled event
rate on the stimulated electrode falls below `pre_stimulus_median_rate * 3` and
stays below it for 5 consecutive bins. Report that value. Report the same
number computed on all other electrodes. If the two differ by more than a
factor of two, say so prominently, because it means the artifact is local and a
global blanking window is wrong.

Write `artifact_psth.csv` with one row per bin and all three curves plus the
voltage percentiles. Write `artifact_psth.png` with a log-scaled x axis and a
vertical line at the proposed blanking window. The CSV is authoritative; the
plot is for reading.

**Verdict line:** `BLANKING <N> ms` plus `LOCAL` or `GLOBAL`.

## Task 0c — usable response window

For each CANDIDATE stimulation compute the time until the next stimulation,
twice: on the same electrode, and on any electrode.

Report the distribution of both: min, 1st, 5th, 10th, 25th, 50th, 75th
percentiles, and max. Then a coverage table: for each candidate window length
in `[0.5, 1, 2, 5, 10, 20, 30, 60]` seconds, how many CANDIDATE stimulations
have at least that much quiet on both definitions, and how many distinct
electrodes and days survive.

**Verdict line:** `RESPONSE WINDOW <N> s` for the largest value that retains at
least 60% of stimulations and all 32 electrodes on at least two days. If no
value satisfies that, say `NO WINDOW SATISFIES CONSTRAINTS` and print the table.

## Task 0d — pre-stimulus baseline

Same as 0c, running backwards: time since the previous stimulation, same
electrode and any electrode.

**Verdict line:** `BASELINE WINDOW <N> s` using the same 60% rule.

Then the joint count: how many CANDIDATE stimulations satisfy both the 0c and
0d windows simultaneously, broken down by day and by electrode. Write this to
`stim_windows.csv`, one row per CANDIDATE stimulation, with columns:
`stim_id, electrode, time_of_stim, day, gap_before_same, gap_before_any,
gap_after_same, gap_after_any, usable`.

## Task 0e — is there enough activity to measure

For each of the five days, and for each of the 32 electrodes, compute the event
rate during the quiet periods on that day, meaning time outside any
stimulation's [0d window, 0c window] span.

Report a day-by-electrode table of rates, plus per day: how many electrodes
exceed 0.1 Hz, 0.5 Hz and 1 Hz.

**Verdict line:** `ACTIVE ELECTRODES <n> of 32` at the 0.1 Hz threshold on the
adjacent pair identified in 0a, or on the two best days if there is no pair.

## Output

`step0_report.md` ends with exactly five verdict lines in order, one per task,
nothing after them. Everything above those lines is tables and the evidence
behind each verdict.

`step0.json` carries every parameter, the per-day event row counts, the slot
assumption flagged as unverified, the code version and the seed.

## Tests

Synthetic fixtures in `tmp_path`, built with h5py and pandas to match the real
table schema. Minimum coverage:

- `test_candidate_filter_exact` — a stim table with CANDIDATE rows plus
  near-miss rows differing in one parameter; only the exact matches are
  selected.
- `test_adjacent_pair_detection` — days 1, 2 and 9 give the pair (1, 2); days
  1, 5 and 9 give NO ADJACENT PAIR.
- `test_medium_change_disqualifies_pair` — a medium change between two adjacent
  days removes that pair.
- `test_blanking_recovers_known_artifact` — synthesize events with a 4 ms
  saturating burst after each stimulus on top of a Poisson baseline; the
  proposed blanking window must land within 1 ms of 4 ms.
- `test_blanking_returns_nothing_on_clean_data` — same fixture with no
  artifact; the proposed window must be the first bin.
- `test_gap_computation_same_vs_any_electrode` — stimulations interleaved
  across two electrodes give different same-electrode and any-electrode gaps.
- `test_events_never_loaded_whole` — monkeypatch `HDFStore.select` and assert
  the call count equals the number of distinct days and that every call passes
  a `where` clause.

The last one is the one that matters for runtime. Write it first.

## Constraints

- Report and propose. Do not choose a blanking window or a response window and
  bake it into a config. Every verdict is a recommendation for a human to
  accept.
- Never substitute a default for a value that should be read from the file. If
  something is absent, say so and stop that task.
- Do not implement the recovery fit, the split-half test, the day-to-day
  correlation, or anything touching Gate 2.
- Do not open `fs369_raw.hdf5`.
- No plots other than `artifact_psth.png`.

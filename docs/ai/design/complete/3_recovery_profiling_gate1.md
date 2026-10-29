# Gate 1 Design

Scope: FinalSpark fs369, days 2025-02-20 and 2025-02-21. No lab, no new data.
`analysis_standards.md` applies and is not repeated here.

## Status

# Gate 1 — Status, 2026-10-20

Gate 1 asks whether a measurement of how fast tissue recovers after a
perturbation repeats when you make it twice. We ran it on FinalSpark fs369,
one neural culture recorded continuously for 21 days until it died. The
dataset turned out to be richer than expected and nothing like what the
original design assumed: stimulation is whole-array and simultaneous, and each
day runs an identical 32-block dose-response sweep of 16 parameter conditions
at two polarities, 10 trials each at 10 second spacing. Two consecutive days,
2025-02-20 and 2025-02-21, run the same sweep and are the only adjacent pair
not separated by a medium change. A long quiet period before each sweep gives
a clean baseline. 24 of 32 electrodes are active on both days.

**Gate 1 failed.** The pre-registered measure, the condition-centered
correlation of per-electrode recovery time across the two days, came back at
0.128 against a threshold of 0.7. The failure is interpretable rather than
ambiguous: within-session reliability was 0.779, so ten trials are enough; a
positive control recovered 0.878, so the pipeline finds an electrode-identity
effect when one is present; and the observed value sits far below the ceiling
rather than near it. So the measurement is reproducible inside a session and
does not survive overnight. Post-hoc checks, not part of the pre-registered
gate, show a gradient: peak response amplitude correlates across days at 0.80,
electrode-averaged recovery time at 0.61, the per-cell value at 0.13, and the
residual at -0.14. Something about each electrode persists overnight, strongest
in response size and weaker in recovery time, but none of it reaches the gate.

One confound is unresolved and it changes the meaning of the result. Day 1 had
three hours of maximum-amplitude stimulation ending 7.4 hours before its sweep;
day 2 had nothing comparable. "The tissue changed overnight" and "day 1 was
pre-conditioned" both explain the data and imply opposite things. Baseline
firing rates also moved by about 1.5x between the two days, consistent with a
culture 21 days from death. Until that is separated, the honest statement is
that this measurement did not repeat across these two sessions in this culture,
and we do not know whether that is a property of the measurement or of the
tissue. Code: `scripts/gate1_run.py`, outputs in `output/gate1/`.

## Closeout, 2026-10-29

Archived to `docs/ai/design/complete/`. The gate ran and failed (Status above).
The items below were not done and are not planned under this design.

**Code.** `scripts/gate1_run.py` (parameters, HDF5 queries, outputs),
`src/benchmarks/recovery/gate1.py` (analysis functions, moved there in prompt 260),
`tests/benchmarks/recovery/test_gate1.py`. Outputs: `output/gate1/`.

**Named in this design, not built:**

- Double-exponential fit (Secondary). As written, it also cannot represent the
  second rise at 0.3-2 s.
- Polarity comparison.
- Amplitude-response: only the cross-day correlation of per-electrode slopes was
  computed (0.323). Whether `t_return` lengthens with amplitude was never reported.

**`analysis_standards.md` requirements not met for this gate:**

- No synthetic spike-train positive or negative control. The positive control
  exists only at the correlation step.
- The null permutes electrode labels at the last step, not a whole-pipeline
  shuffle of stimulus times.
- Confound register not run. `t_return` is defined against 1.5 x baseline and
  baseline moved about 1.5x between days, so part of the cross-day change in
  `t_return` may follow from the definition. Not tested.
- Headline not re-run under two alternative definitions (threshold, bin count).
- No config file; parameters are constants in `scripts/gate1_run.py`.

**Unresolved:** the block-0 pre-conditioning confound. Prompt 257
(`scripts/day_census.py`) found no other clean adjacent day pair in fs369, so it
cannot be separated without new data.

**Provenance gap:** in the Status section, 0.13 is the condition-centered r
(0.128, `output/gate1/stdout.txt`). The post-hoc 0.80, 0.61 and -0.14 are not
written by any committed script or output file.

**Body text vs the run:** the sections below say 19 active electrodes and 16
conditions. The run used 24 active electrodes and 32 blocks (16 conditions x 2
polarities); see `output/gate1/stdout.txt`. The text below is left as designed.


## What the data is

Each day runs an **identical 32-block dose-response sweep**: 16 parameter
conditions, each at polarity 0 then polarity 1, 10 stimulations per block at
10.0 s spacing, 21.0 s between blocks. Amplitudes 0.8, 1.0, 1.2, 1.5 with
durations 100, 200, 300, plus four asymmetric pairs. Whole-array: 32 electrodes
fire at each timestamp.

02-20 blocks 1-32 and 02-21 blocks 0-31 are the same sequence in the same
order.

So the available data is **16 conditions x 10 trials x 32 electrodes x 2 days**,
not 10 trials at one setting.

**Baseline:** a long quiet period sits immediately before each day's sweep.
02-20 has 7.4 hours with no stimulation (08:42 to 16:07). 02-21 is preceded by
roughly 20 hours since the previous day's last stimulation at 17:06. Use the
**last 30 minutes before each sweep begins**, so the baseline is temporally
adjacent to the measurement.

**Blanking: 3 ms after each stimulus.** Settled, not re-derived.

**Active electrodes:** 19 of 32 exceed 0.1 Hz on both days. Everything below is
computed on those 19.

## The response shape

Pooled recon shows a peak at 10-20 ms, decay by 60 ms, a second rise around
0.3-2 s, then a tail flattening from 3-10 s. **A single exponential is the wrong
model** and must not be used as the primary measure.

**Primary measure, non-parametric.** Per electrode, per condition, per day:
bin the pooled post-stimulus rate from 3 ms to 10 s on log-spaced bins. Define
`t_return` as the **last** bin whose rate exceeds `1.5 x baseline`. This is
well-defined on a multi-humped curve and assumes nothing about shape.

Also record `peak_rate`, `t_peak`, and `residual`, the mean rate in the final
second as a multiple of baseline.

**Secondary, exploratory.** A double exponential over 0.1 to 10 s, reported but
not gating anything.

## Step 1 — Reliability ceiling

Per electrode and condition, split the 10 trials into odds and evens, compute
`t_return` for each half, correlate across all 19 x 16 = 304 electrode-condition
pairs, and apply the Spearman-Brown correction.

This is now a well-estimated ceiling rather than the thin 5-against-5 the
previous design relied on.

**If the corrected ceiling is below 0.7, Gate 1 is not answerable at this trial
count** and that is the output, not a failed gate.

## Gate 1 — Repeatability

Three correlations, reported together. The decomposition matters.

**1. Per condition.** For each of the 16 conditions, correlate the 19-element
`t_return` vector across the two days. Report all 16, plus their median and
range.

**2. Pooled, condition-centered.** Flatten to 304 electrode-condition values per
day, subtract the condition mean within each day, then correlate the residuals
across days. This is electrode-specific repeatability with the condition effect
removed.

**Pooling without centering is a trap** and must not be the headline. If
`t_return` varies strongly by condition and that pattern repeats, raw pooled
correlation will look high while saying nothing about per-electrode
reliability.

**3. Dose-response slope.** Per electrode, regress `t_return` on amplitude
across the symmetric conditions. Correlate the 19 slopes across days.

**Pass: the condition-centered correlation (2) is >= 0.7.** Report the other two
alongside, with the Step 1 ceiling and the shuffle null.

If (2) falls below 0.7 but near the ceiling, the limit is trial count, not the
method. Say which.

**n = 1 culture, 19 electrodes, 16 conditions, 2 sessions, 10 trials each.**

## What is now possible that was not

**Dose-response.** Does `t_return` lengthen with stimulation amplitude? There
are four amplitudes and three durations, 10 trials each, on two days. If
recovery time tracks the size of the perturbation and does so reproducibly,
that is a real relationship measured on public data.

This is not Gate 3. There is still no vehicle. But it is the nearest thing this
dataset supports and it is worth reporting whatever Gate 1 says.

**Polarity.** Every condition runs at both polarities 21 s apart. Whether
`t_return` differs between them is a biological question, not a replicate
check, since negative-first and positive-first stimulation are not equivalent.
Report it; do not use it as a reliability estimate.

## Confounds

- **Block 0 on 02-20.** Three hours of sequential per-electrode stimulation at
  amplitude 2.5, the highest in the file, 2,040 events ending 7.4 hours before
  the sweep. 02-21 has nothing like it. If the two days disagree, look here
  first.
- One culture. 19 electrodes nested inside it. Not 19 independent units.
- Twelve neurospheres across four slots of eight electrodes, mapping unknown.
  Slot is not neurosphere and must not be treated as a tissue unit.
- Whole-array stimulation. A per-electrode `t_return` is one electrode's view
  of a global perturbation, not a local measurement.
- `pulse_train_period` units unresolved. Constant across the sweep, so it does
  not affect the gate, but stimulation frequency cannot be stated in a writeup.

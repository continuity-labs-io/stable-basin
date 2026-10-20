# Gate 1 Design

Scope: FinalSpark fs369 only. No lab, no new data. `analysis_standards.md`
applies and is not repeated here.

## What Step 0 established

**Stimulation is whole-array and simultaneous.** A median of 32 electrodes fire
at the identical timestamp. Each session is **10 stimulation events spaced
exactly 10 seconds apart across about 90 seconds**, once per day. The 1,294
rows are 10 events times 32 electrodes, not 1,294 independent perturbations.

Consequences: there is no stimulated-versus-unstimulated electrode contrast
anywhere in this dataset, so no sham exists at any level. And per electrode per
session there are **10 trials**, not hundreds.

**One adjacent pair exists: 2025-02-20 and 2025-02-21.** 320 stimulations and
all 32 electrodes on each, no medium change between them, no door openings
inside either span. The medium change on 2025-02-28 disqualifies every other
pair involving a later day.

**19 of 32 electrodes are active above 0.1 Hz on both days of that pair.** The
Gate 1 correlation is over 19 points, not 32.

**There is a large, clean evoked response.** Pooled across all stimulations:
baseline 1.9 Hz, peak 118.7 Hz in the 10 to 20 ms bin, then 41.3, 8.7, 6.6 and
7.4 Hz through 500 ms. A peak at 10 to 20 ms rather than in the first bin means
this is a network response, not an electrical artifact.

**The response has not returned to baseline by 500 ms.** It sits at roughly
four times the pre-stimulus rate at the end of the measured window.

**Activity declines across the five sessions**: 22, 22, 19, 13, 0 electrodes
above 0.1 Hz, with death two days after the last session. The adjacent pair is
the least confounded pair available, since both days sit at 22.

## Settled by fiat: blanking at 3 ms

Step 0 could not determine a blanking window because the rule used a median of a
sparse histogram, which is zero. Rather than rerun it: the pooled rate in the
first millisecond is 27.8 Hz against a 118.7 Hz peak at 10 to 20 ms, and there
is a zero-event run from 2 to 2.4 ms.

**Blank the first 3 ms after each stimulus.** It is conservative, it costs 3 ms
out of a 10,000 ms window, and the cost of being wrong is negligible. Recorded
as a parameter, not re-derived.

## Step 1 — The reliability ceiling, before either gate

Two checks. Both are cheap and either can stop the work.

**1a. Does the response return within 10 seconds?** Build the pooled PSTH out to
the full 10 second inter-stimulus interval, not 500 ms. Compare the rate in the
last second before a stimulus against the rate in the quiet period before the
session's first stimulus.

If they differ, the 10 second spacing is shorter than the recovery and the 10
trials in a session are not independent. That is not fatal but it changes
everything downstream: the baseline becomes the pre-session quiet period rather
than the inter-stimulus interval, and the trials must be modeled as a sequence
rather than averaged.

**Report:** the full 10 s PSTH, the pre-session rate, the last-second-before
rate, and whether they agree.

**1b. What is the within-session reliability?** Per electrode, split the 10
trials into odds and evens, fit each half, correlate the two 5-trial estimates
across the 19 active electrodes, and apply the Spearman-Brown correction.

This is the ceiling. A between-day correlation cannot exceed it. **If the
corrected within-session reliability is below 0.7, Gate 1 is not answerable on
this dataset at this trial count**, and the honest output is that finding rather
than a failed gate.

Report a bootstrap confidence interval over trials, because five trials per half
is thin and a point estimate alone will mislead.

## The measurement

Per electrode, per session:

1. Take the 10 stimulus times.
2. Discard 0 to 3 ms after each.
3. Build the peri-stimulus rate over 3 ms to 9 s, pooled across the 10 trials,
   with log-spaced bins.
4. Baseline is the pre-session quiet period, or the inter-stimulus tail if 1a
   shows they agree.
5. Fit `rate(t) = baseline + A * exp(-t / tau)`. Record `A`, `tau`, and the
   residual at 9 s.
6. Also record the non-parametric version: time to fall below 1.5 times
   baseline.

A session gives a 19-element vector of `tau`, one per active electrode, plus the
same for `A` and the residual.

## Gate 1 — Repeatability

Compute the `tau` vector for 2025-02-20 and for 2025-02-21. Correlate across the
19 active electrodes.

**Pass: r >= 0.7.**

Report alongside, always: the within-session ceiling from 1b, the same
correlation for `A` and the residual, the shuffle null, and the disattenuated
correlation. If r is below 0.7 but close to the ceiling, the measurement is as
repeatable as 10 trials allow and the limitation is trial count, not the method.
Say which.

**n = 1 culture, 19 electrodes, 2 sessions, 10 trials each.** Every verdict
states this.

**Do not** compute Gate 1 across non-adjacent days. Activity falls from 22 to 13
to 0 electrodes across the later sessions, so a weak correlation there is the
culture dying, not a reliability failure.

## Confounds

- One culture. 19 electrodes nested inside it. Not 19 independent units.
- Twelve neurospheres across four slots of eight electrodes. The electrode to
  neurosphere mapping is unknown, and slot is not neurosphere. Unresolved, and
  nothing may treat slot as a tissue unit.
- Whole-array stimulation means every response is a network response. A
  per-electrode `tau` is that electrode's view of a global perturbation, not a
  local measurement.
- `pulse_train_period` units remain unresolved. Same value throughout the
  candidate set, so it does not affect either gate, but the stimulation
  frequency cannot be stated in a writeup yet.

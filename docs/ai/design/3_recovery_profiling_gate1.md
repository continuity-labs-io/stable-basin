# Gates 1 and 2 — Design

Scope: FinalSpark fs369 only. No lab, no new data, no hardware. Read
`analysis_standards.md` first; its rules apply throughout and are not repeated
here.

## What the gates ask

**Gate 1 — Repeatability.** The same measurement, made on the same tissue on two
consecutive days, agrees. Threshold: Pearson r >= 0.7 across electrodes.

**Gate 2 — Agreement.** The recovery rate measured by perturbing agrees with the
rate inferred from spontaneous activity in the same tissue. Threshold: r >= 0.5
across electrodes.

Gate 1 failing means the measurement is not a measurement. Gate 2 failing means
perturbing and passive observation are reading different things, which is
informative either way but stops the current plan.

## The data

One culture, fs369. First recording 2025-02-19, death 2025-03-12, lifespan 21
days 16:33, cause of death recorded as "almost no activity."

- Events table: `electrode`, `time_of_event`, `max_voltage_uv`. Spikes already
  detected at 6x the median standard deviation. No detection needed.
- 32 electrodes in a 4x8 air-liquid-interface setup, 12 neurospheres total.
- Stimulations: 130,113 triggers from 2025-02-20 to 2025-03-10, on seven day
  bins: 0, 1, 7, 8, 12, 14, 18.
- One row is a trigger, not a pulse. Confirmed empirically: zero percent of
  per-electrode groups match `nb_stim_pulse`.
- The candidate parameter set for Gate 1: a1=0.8, a2=0.8, d1=300, d2=300,
  nb_stim_pulse=2, pulse_train_period=10000, stim_polarity=0, stim_shape=0.
  1,294 stimulations, 32 electrodes, 5 distinct days.
- 485.6 hours with no stimulation anywhere: 19.1 h before the first stim, 416 h
  across seven gaps longer than 6 h, and a 50.5 h tail after the last stim.

## Step 0 — Four things to determine before defining the measurement

Each is cheap, each changes the design, and none should be assumed.

**0a. Which five days does the candidate parameter set actually cover?** Gate 1
as specified needs two consecutive days. The stimulation day bins include two
adjacent pairs (0/1 and 7/8), but it is not yet known whether the candidate set
appears on either. If it covers no adjacent pair, either pick a different
parameter set or restate Gate 1 for the closest available pair and say so.

**0b. Where does the stimulation artifact end?** This is the largest technical
risk in the whole design. Events are threshold crossings, and a stimulus
produces a large artifact that will be detected as spikes. Determine the
blanking window empirically: plot event count against time since stimulus at
sub-millisecond resolution, pooled across all stimulations, and find where the
rate stops being saturated. Charge recovery is enabled, so the artifact may
outlast the pulse itself. Everything inside the blanking window is discarded and
the window length goes in the sidecar.

**0c. What is the usable response window?** Set by the gap to the next
stimulation. Compute the distribution of quiet time following each stimulation in
the candidate set. The response window is the largest value that most
stimulations can support. If that is under a second, the measurement is a short
one and the design says so rather than pretending otherwise.

**0d. What is the pre-stimulus baseline?** Same logic, running backwards.
Stimulations without enough quiet time before them are excluded.

## The measurement

For one stimulation on one electrode:

1. Baseline rate, from the pre-stimulus window determined in 0d.
2. Discard everything inside the blanking window from 0b.
3. Binned event rate across the response window from 0c.
4. Fit `rate(t) = baseline * (1 - A * exp(-t / tau))`. Record `A` (suppression
   depth), `tau` (recovery time), and the residual at the end of the window.
5. Separately, record the non-parametric version: time to return to 90% of
   baseline. Per the standards doc, both are computed from the start.

A session produces a 32-element vector of `tau`, one per electrode, plus the
same for `A` and the residual.

Per-electrode is the primary unit rather than population rate. Both the PULSE
drift result and the structure of this dataset point the same way: population
summaries drift while electrode-specific responses persist.

## Gate 1 procedure

**Pre-check, runs first.** Within a single session, split stimulations into odd
and even, compute a `tau` vector for each half, correlate. If split-half
reliability is below 0.6, stop. The across-day test cannot succeed when the
within-session one fails, and finding that out costs one session's analysis
rather than five.

**The gate.** Take the adjacent day pair identified in 0a. Compute the `tau`
vector for each day. Correlate across the 32 electrodes.

**Pass:** r >= 0.7.

**Report alongside, always:** the same correlation for `A` and for the residual,
the shuffle null from the standards doc, and the number of electrodes
contributing (electrodes with too few usable stimulations are excluded and
counted).

**Do not** compute Gate 1 across distant days. The five days span a culture's
decline toward death, so a low correlation between day 1 and day 18 is a
biological finding rather than a reliability failure. Those two questions must
not be collapsed.

## Gate 2 procedure

**Perturbed rate:** the `tau` vector from a stimulation session, as above.

**Passive rate:** from spontaneous activity in the quiet period immediately
adjacent to that session, using the unstimulated hours. Per electrode, compute
the autocorrelation decay time of the binned event rate. Bin size is chosen to
match the timescale of the perturbed recovery measured above, and recorded.

**The gate:** correlate the two 32-element vectors. Pass at r >= 0.5.

**The adjacency requirement is load-bearing.** Use the nearest quiet window in
time, not a pooled average over all 485 hours. Drift over days is exactly the
confound this comparison is vulnerable to.

## Confounds specific to these gates

Beyond the standing register:

- **All five days are one culture.** The verdict states n = 1 biological unit,
  32 electrodes, 5 sessions. This is the independence accounting rule and it is
  not optional.
- **Twelve neurospheres across four slots of eight electrodes.** The mapping
  from electrode to neurosphere is unknown, so electrodes cannot be treated as
  independent tissue units. Report it as unresolved.
- **Medium was changed once, on 2025-02-28T14:01:34.** Check whether it falls
  between the two days used for Gate 1. If it does, pick a different pair.
- **Door openings are logged.** Check whether any fall inside a session.
- **`pulse_train_period` units are unresolved.** It does not block either gate,
  because the same value is used throughout the candidate set, but it means the
  stimulation frequency cannot be stated in the writeup yet.

## What failure means

Gate 1 fails: the measurement is not reproducible on the best available data, and
the protocol needs rebuilding before anything else proceeds. That is a real
result and it is cheap to reach.

Gate 2 fails: perturbation and passive observation disagree. Worth knowing
precisely because the passive route is what everyone else is already doing.
Report the disagreement rather than dropping it.

Either way the question survives the answer.

## Out of scope

- Gate 3. No vehicle or sham exists in fs369; all 32 electrodes are stimulated.
- Gate 4. Needs a second site.
- The lifespan trend, meaning whether `tau` lengthens toward death. It is the
  most interesting question in this dataset and it depends on Gate 1 passing
  first. Separate doc.
- Raw voltage. Everything here runs on the events table in
  `fs369_package.hdf5`. The 60 GB raw file is not needed.

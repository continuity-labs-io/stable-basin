# Analysis Standards

Standing rules for every gate in this project. Short on purpose. If a rule here
conflicts with a gate design doc, this file wins.

## Why this exists

Two previous negative results in this project turned out to be broken pipelines
rather than absent biology, and both cost time and resources. The governing rule follows from that:

**A null is uninterpretable unless the pipeline has been shown to find a signal
that is known to be there.**

Every measurement below gets its positive control before it gets its first real
number.

## Validation comes before measurement

Four controls. None of the real analysis is written until all four exist.

**Synthetic positive control.** Generate spike trains with a known recovery time
constant. The pipeline must recover it within a stated tolerance. This runs in CI
on every commit. If it ever fails, no real-data result is trusted until it passes
again.

**Synthetic negative control.** Generate spike trains with matched firing rates
and matched event counts but no recovery structure. The pipeline must return
nothing. A pipeline that finds structure here is finding it in the method.

**Shuffle control on real data.** Permute which perturbation each response window
follows, preserving every rate and every per-electrode statistic. This destroys
only the link between a perturbation and its own response. The null distribution
is built by running the **entire pipeline** on shuffled data, not by shuffling at
the last step.

**Split-half reliability.** Within a single session, compute the measure on
odd-numbered events and even-numbered events separately, then correlate. A
measure that is not reproducible within one session cannot separate anything.
This is the cheapest and most informative test available and it runs first.

## Method decisions, pre-registered

These are settled. Changing one is a deliberate act that gets recorded in the
gate doc, not a quiet edit.

- **Multi-unit per electrode. No spike sorting.** Every measure here is a
  network-level quantity. Sorting introduces errors that are not independent
  across conditions.
- **One detector, chosen in advance.** Pick it, write it down, then show the
  headline result survives two alternatives. Reporting only the method that
  worked is the failure mode this project exists to avoid.
- **Compute the fit and the threshold crossing, both.** An exponential fit gives
  one interpretable number but assumes a shape. A time-to-return-to-fraction
  assumes nothing and is noisier. They should agree. Where they disagree, the
  disagreement is the finding.
- **Crude detectors are quarantined.** Any detector written to answer "is
  anything happening here" lives in its own module with a header saying so, and
  is never imported by an analysis module.

## Output contract

Every analysis writes a tidy table, one row per event or per session, plus a JSON
sidecar recording the code version, every parameter value, and the random seed.

No figure is ever produced from anything but one of those tables.

## Confound register

Before any comparison, each candidate confound is tested against the outcome on
its own, and every confound that predicts the measure is reported alongside the
result whether or not it changes the conclusion.

Standing list: session duration, number of active electrodes, clock time of day,
days since first recording, electrode identity, and anything that changed in the
protocol between sessions.

## Independence accounting

Report the breakdown, never just the headline count. Five sessions from one
culture is not five independent measurements, and any verdict that hides that is
wrong even when the number is right.

Every gate verdict states: n observations, n independent biological units, and
how they are nested.

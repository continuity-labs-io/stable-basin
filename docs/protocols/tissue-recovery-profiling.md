# Tissue Recovery Profiling

Protocol brief, revised 2026-10-21. Open protocol: design, gates and analysis code are
published whether they pass or fail.

Ryan J. McCall, PhD, ContinuityLabs.io

## What this is

Apply a controlled load to living neural tissue on a microelectrode array, then measure how
long it takes to return to baseline. The readout is recovery time, not activity level.

The claim under test is that recovery rate carries information that no resting measurement
carries. Two near-term uses: a functional acceptance test for engineered or grafted tissue,
where histology tells you tissue is alive and nothing tells you it works; and an aging
readout, where the hypothesis is that recovery slows with age.

This is not a firing-rate or burst-rate assay. Those describe what tissue is doing. This
describes what it can still recover from.

## What the instrument has to do

The requirements, stated as a spec rather than as questions. Items marked OPEN are
currently with MaxWell's engineering team.

1. **Non-uniform stimulation intervals.** Probe pulses are delivered at user-specified,
   unevenly spaced delays after a conditioning event. A fixed interburst interval repeated
   N times does not satisfy this. OPEN: whether the hardware sequencer supports it or only
   the API does.
2. **Stable electrode addressing across sessions.** The same electrodes must be recorded
   weeks apart. On a switch-matrix array this means a routing configuration saved once and
   reloaded verbatim, never re-derived from a per-session activity scan. OPEN: whether
   configurations persist and reload exactly.
3. **Multi-site stimulation with independent per-site amplitude.** Four to six sites per
   session, each calibrated separately. MaxWell's 32 stimulation buffers are more than
   sufficient in count. OPEN: whether each buffer takes an independent amplitude.
4. **Short post-stimulus dead time.** The response window opens as soon as the stimulation
   artifact clears. That duration sets the floor of the measurement. OPEN: artifact duration
   on channels near a stimulation site, and whether the software blanks it.
5. **Repeated non-destructive sessions** on the same preparation over weeks.
6. **Sampling above 10 kHz.** Adequate for threshold-crossing and multi-unit analysis, which
   is what this protocol uses. Spike sorting is deliberately not used, so the single-unit
   tracking HD-MEA platforms are usually sold for is not part of the design.
7. **Controlled perfusion**, if the perturbation is pharmacological. In static culture a drug
   arrives by diffusion with onset smeared over minutes, which destroys the temporal
   resolution the whole measurement depends on. Perfusion exchanges in seconds.

Scale of the first study: twenty wells, two weeks, twenty-minute sessions.

Not vendor-locked. Any array with scriptable multi-site stimulation, stable electrode
addressing and a sampling rate above about 10 kHz can run this.

## The measurement

**Readout: paired-pulse electrical probes.** Settled. They cost nothing, repeat indefinitely,
carry microsecond timing, and have two standardized human analogs (cardiac S1-S2 pacing,
paired-pulse TMS).

**Perturbation: under decision.** Two configurations are live.

| | Clinical bridge | Age evidence in humans | Cost | Open issue |
| --- | --- | --- | --- | --- |
| Electrical conditioning burst | none | none | free | no translational path |
| Anesthetic, GABA-A | surgery, universal | strong: delayed emergence, POCD | one well per dose | needs perfusion; tissue maturity gate |

The anesthetic configuration is favored because emergence from anesthesia is the only
recovery measure in this space already standardized and already known to worsen with age.
Its one real cost, smeared onset, is removed by perfusion. The electrical version is cheaper
and is the right way to establish the readout before any compound is introduced.

One open design decision gates this and is mine to make: paired-pulse operates in
milliseconds to hundreds of milliseconds, while the pilot's probe delays ran to 32 seconds.
Whether the assay is a millisecond-scale measure, a seconds-scale measure, or one measure
spanning both is not yet settled.

**Protocol requirements that are settled**, whichever perturbation wins:

- At least ten repeats per condition. The pilot measured within-session split-half
  reliability at ten trials and got 0.78.
- No uncontrolled stimulation before a session. Conditioning history is logged, and sessions
  are only compared across matched histories. Matched means matched program, not matched
  parameters: in the pilot recording one session carried the same parameter set as another
  while delivering 54 stimuli per block instead of 10.
- Fixed routing across sessions, and electrode attrition reported rather than excluded.
- Stimulus amplitude calibrated per site to half-maximal response on the rising limb, swept
  across the full range rather than ramped. The dose-response curve is not monotonic; in
  published MaxWell slice data it peaks near 900 mV and falls above it, so a ramp-until-half
  procedure can land on either side of the peak.

## What comes out

Four numbers per session: resting response, suppression depth, recovery time, residual.
Depth and rate are kept apart deliberately, since a system knocked down further but
recovering at the same rate is a different finding from one knocked down equally that takes
longer to return.

**Recovery time is a threshold crossing, not a fit parameter.** An earlier version of this
protocol specified a single exponential return. The pilot showed that is wrong: the response
to a single pulse is at least two-component, with a sharp peak at 10 to 20 ms decaying within
60 ms, a second rise between 0.3 and 2 seconds, and a tail still elevated at 10 seconds. A
single exponential forced onto that returns a number with no stable meaning. Recovery time is
the last time bin exceeding 1.5x baseline on log-spaced bins. The exponential fit is computed
and reported alongside; disagreement between them is a finding, not an error.

**Two measures, one baseline-free.** Baseline is not stable across days, so a baseline-free
measure, time to half of peak, is computed in parallel. Which one is primary is decided per
substrate by its measured within-session ceiling and declared before any cross-session
comparison. The pilot found half-peak unusable on that substrate, ceiling 0.16 against 0.78.

**Censoring is reported.** Responses that have not returned by the last delay are censored,
not missing. Any analysis states its censored fraction and fails above 30 percent.

**Response window is measured, not assumed.** The lower edge is the blanking period, measured
once on the instrument by recording the artifact near a stimulation site, then fixed. The
pilot's 3 ms came from threshold-crossing data and does not transfer to continuous voltage.

## Gates

Pre-declared pass/fail tests, each able to kill the program.

**Gate 1, repeatability.** Two sessions on consecutive days, untreated wells, matched
conditioning history, identical routing. Pass if recovery times correlate at 0.7 or better.
The gate is an acceptance test with five parts, and a bare correlation without them is not a
result:

1. Report the within-session ceiling, split-half with Spearman-Brown correction. A
   cross-session correlation is uninterpretable without it. Pass requires a ceiling of at
   least 0.7 before the cross-session number is read.
2. Run a synthetic positive control carrying a known electrode-identity effect through the
   same pipeline. If it does not return near 1.0, the pipeline is broken.
3. Run a label-permutation null. The observed value must clear its 95th percentile.
4. Center within condition before pooling, and report the raw pooled value separately.
5. Count independent units honestly: cultures first, then wells, electrodes, conditions,
   sessions, trials. On an HD-MEA, neighboring electrodes partly read the same neuron, so
   non-independence is worse than on a low-density array.

**Gate 2, is it a real rate.** The rate measured by pushing should agree with the rate read
from spontaneous fluctuations. Pass at 0.5 across wells.

**Gate 3, does it beat what exists.** Pass if recovery time separates a drug dose from vehicle
where firing rate and burst rate do not. This is also the commercial gate.

**Gate 4, route diversity.** With multi-site stimulation, pass if recovery trajectories from
different sites stay reliably distinguishable in healthy tissue. Exploratory; the first study
sets the threshold. This is the only gate that genuinely requires high electrode density.

## What has been run

Nothing on purpose-collected data. One pilot ran on public data, to exercise the method and
find out what the protocol was failing to specify.

**Data.** FinalSpark Neuroplatform export fs369: 12 human iPSC-derived neurospheres on a
32-electrode array, recorded until the culture died at 21 days. Two consecutive days run an
identical 32-block sweep, 10 trials per block.

**Result.** Gate 1 failed. Condition-centered cross-session correlation 0.128 against 0.7.
Within-session ceiling 0.779. Positive control 0.878. Permutation null cleared.

**Reading.** The measure is reproducible inside a session and did not survive overnight in
this preparation. The failure is interpretable rather than ambiguous: the ceiling rules out
too few trials, the positive control rules out a broken pipeline, and the observed value sits
far below the ceiling rather than near it. Post-hoc there is a gradient, with peak response
amplitude correlating across days at 0.80, electrode-averaged recovery time at 0.61, and the
per-cell value at 0.13.

**What it does not mean.** This is not Gate 1 failing. Gate 1 requires matched conditioning
history and more than one culture, and this had neither: one day followed three hours of
maximum-amplitude stimulation that the other did not. A census of the full recording settles
whether a cleaner pair exists. It does not. Stimulation occurred on 7 of roughly 21 days,
never two adjacent, under a materially different program nearly every time. The recording was
never designed as a repeated-measures experiment.

**What it changed here.** Minimum trial count, the matched-history requirement, the
response-window definition, replacement of the exponential fit with a threshold crossing, the
parallel baseline-free measure, explicit censoring, fixed routing, and the five-part
structure of the Gate 1 acceptance test.

Code and outputs: `scripts/gate1_run.py`, `scripts/day_census.py`, `output/gate1/`. Standing
analysis rules: `docs/ai/design/analysis_standards.md`.

## Where this sits

The measurement is not new. Cardiac electrophysiology has run it for decades as restitution:
pace the tissue, deliver one premature beat at varying coupling interval, see how the response
recovers. It is an established arrhythmia risk measure. Paired-pulse TMS is the same
experiment in awake human cortex. Neither has been pointed at tissue quality or at aging.

Recovery-from-perturbation is also already the primary functional endpoint at the better-funded
reprogramming companies. NewLimit, heading into human trials next year, reports regenerative
rescue in aged hepatocytes and resilience to dietary damage as its headline results, and uses
its transcriptomic age predictors as a screening tool rather than as the result. What is
unclaimed is not recovery as a readout, which is in use; it is recovery measured as a *rate*
rather than an amount, in electrically active tissue.

The standardization template is CiPA, the FDA-initiated cardiac safety initiative that took a
mechanism-based in vitro assay to regulatory recognition between 2013 and 2022. Worth noting
honestly that its iPSC-cardiomyocyte MEA component ended up optional in the final guidance,
and that CiPA could be validated only because 28 drugs with known torsades risk already
existed. Aging has no equivalent answer key.

## Substrate

No preparation offers donor age, multi-week survival and intact circuit architecture at once.
Dissociated rodent cortical culture has the lowest variance, which is what you want when the
instrument rather than the biology is under test, and is the Gate 1 substrate. Acute slices
from mice at 3, 12 and 24 months carry real donor age and real architecture but live 6 to 12
hours, so they compare animals rather than tracking a preparation. Directly converted human
neurons are the only route to human cells retaining donor age, since iPSC reprogramming erases
it. Organoids are set aside: no donor age, high batch variance, and only the contacting surface
is recorded.

## Open questions

**For hardware and software people.** Items 1 through 4 in the requirements section, plus:
electrode impedance and what current a given voltage command actually delivers, since
stimulation is voltage-commanded in millivolts; whether the roll-off above 900 mV is
depolarization block in the tissue or saturation in the instrument; whether a 2-second 20 Hz
burst at half-maximal amplitude degrades PEDOT-coated electrodes over repeated weekly
sessions; realistic throughput per day including setup, calibration and media changes.

**For electrophysiologists.** Which controls are missing beyond the interleaved no-burst
cycles and the Gate 1 acceptance test. Whether comparing a pushed rate against one inferred
from spontaneous fluctuation is a fair test. Whether 0.7 and 0.5 are defensible thresholds.
How long a preparation needs between sessions before conditioning history stops mattering,
since 7.4 hours was not enough. Whether an existing measure of trajectory distinguishability
should be used for Gate 4 rather than inventing one.

**Practical.** Whether this can be run as a service and on what instrument. Cost for twenty
wells over two weeks, with and without the multi-site extension. Raw data format.

## Sources

**Prior art, measurement:** [Tabak, Mascagni & Bertram 2007](https://pubmed.ncbi.nlm.nih.gov/17414972/) ·
cardiac restitution (textbook electrophysiology) · paired-pulse TMS

**Prior art, standardization:** [Blinova et al., Cell Reports 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6226030) ·
[FDA ICH E14/S7B Q&A, Aug 2022](https://www.fda.gov/media/161198/download) ·
[HESI NeuTox MEA subteam](https://hesiglobal.org/wp-content/uploads/2020/09/HESI-NeuTox-2020-MidYear-Update.pdf)

**Platform:** [MaxOne specifications](https://www.mxwbio.com/products/maxone) ·
[MaxWell switch-matrix architecture](https://www.mxwbio.com/our-technology) ·
[Kagan et al., Neuron 2022](https://www.cell.com/neuron/fulltext/S0896-6273(22)00806-6)

**Substrate:** [Directly reprogrammed neurons retain aging signatures](https://www.salk.edu/news-release/researchers-learn-how-to-grow-old-brain-cells/) ·
[Stimulation and artifact suppression on HD-MEAs](https://pmc.ncbi.nlm.nih.gov/articles/PMC6711758/)

**Pilot data:** FinalSpark Neuroplatform whole-life export fs369, [finalspark.com](https://finalspark.com/)

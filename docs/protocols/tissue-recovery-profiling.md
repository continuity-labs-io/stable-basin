# Tissue Recovery Profiling — Protocol Brief for Review

Last revised 2026-10-20. Open protocol: the design, the gates and the analysis code are published whether they pass or fail.

## What this measures

How long living neural tissue takes to return to baseline after a controlled electrical load, and whether that return time lengthens with the age of the tissue.

The claim being tested is that recovery rate carries information about aging that no resting measurement carries. In human cohorts, variables that return slowly to baseline already predict mortality — but those rates are inferred from years of passive observation, never from a load applied at a known time by a known amount. This protocol applies the load.

It is deliberately not a firing-rate or burst-rate assay. Those describe what the tissue is doing. This describes what it can still come back from, which is a different quantity and the one the aging literature keeps pointing at without measuring directly.

## What already exists

Almost none of this is new on its own. The measurement is decades old, the standardization pattern is established in two adjacent fields, and only the last step has never been taken.

| Layer | State of the art | Where |
| --- | --- | --- |
| The measurement — condition the tissue, then probe the return at increasing delays | Standard synaptic physiology since the 1960s: paired-pulse depression, post-tetanic depression, and synaptic depression-and-recovery driving network activity on MEAs | [Tabak et al. 2007](https://pubmed.ncbi.nlm.nih.gov/17414972/) |
| The same logic in cardiac tissue, under the name **restitution** — a conditioning train, then a premature beat at variable coupling interval | Established arrhythmia-risk measure, run as S1–S2 pacing | textbook cardiac electrophysiology |
| The standardized assay — fixed protocol, blinded compound panel, several labs, a regulator in the room | **CiPA** in cardiac: FDA-initiated with HESI and the Cardiac Safety Research Consortium; the iPSC-cardiomyocyte MEA component ran a 28-drug multi-site validation | [Blinova et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6226030), [Millard et al. pilot](https://files.axionbiosystems.com/sites/default/files/millard_cipa_pilot_study_toxsci2108.pdf) |
| The same, in neural tissue | HESI NeuTox MEA subteam on seizure liability; the EPA network formation assay inside ToxCast for developmental neurotoxicity | [HESI NeuTox](https://hesiglobal.org/wp-content/uploads/2020/09/HESI-NeuTox-2020-MidYear-Update.pdf) |
| A functional in vitro readout validated against **when the organism dies** | Does not exist, in any tissue or species | — |

So the honest framing of this protocol is: the stimulus and the readout are borrowed from synaptic physiology; the validation structure is borrowed from CiPA; the extension is running it on the same preparation across weeks rather than once; and the new claim is the last row.

Two things in the protocol are unusual rather than borrowed. The probe delays sit between 0.25 and 32 seconds, which is the gap between paired-pulse work in milliseconds and plasticity work in minutes to hours — that band is thinly studied. The pilot described below found real response structure across that whole band, including a component still running at 10 seconds, which supports putting the measurement window there. And recovery time is treated as the primary readout rather than as a mechanistic probe of synaptic function, which is a reframing, not a discovery.

## Substrate

No preparation gives donor age, survival over weeks, and intact circuit architecture at the same time. The plan uses three, each for a different job.

| Preparation | What it is for | Donor age | Lives for | Architecture |
| --- | --- | --- | --- | --- |
| Dissociated rodent cortical culture, E18, plated on the array | Gate 1 — is the assay repeatable at all | none | months | none |
| Acute hippocampal or cortical slices, C57BL/6 at 3, 12 and 24 months | the aging comparison | real | 6–12 hours | yes |
| Directly converted induced neurons, young vs old human donors | the human arm | real | weeks | none |

The first is chosen because it has the lowest variance available, which is what you want when the thing under test is the instrument rather than the biology. The second is cross-sectional by necessity — one session per slice, so this compares animals rather than tracking a preparation. The third is the only route to human cells that carry the donor's actual age: iPSC reprogramming erases age signatures, direct conversion from fibroblasts retains them.

Organoids were considered and set aside: a three-dimensional body only contacts the electrodes it happens to sit on, batch variability is large relative to the effect, and there is no donor age, so there is no aging phenotype to measure. The pilot below ran on organoid data and is consistent with that read. In a 32-electrode recording of 12 neurospheres, 24 electrodes carried usable activity and baseline firing rate moved by roughly 1.5x overnight in a culture approaching the end of its life. Organoids remain useful for method development and unsuitable as the aging substrate.

Organotypic hippocampal slice culture sits between the first two — weeks of viability with laminar circuitry intact, neonatal donor — and may be the right substrate for the blockade work described below.

## Platform

The protocol is written against a CMOS switch-matrix HD-MEA, specifically the MaxWell MaxOne and MaxTwo. Three properties of that architecture constrain the design and are stated here because they change the protocol rather than merely implementing it.

**Electrodes are not channels.** A well holds 26,400 electrodes at 17.5 µm pitch over a 3.85 × 2.10 mm area, which is 3,265 electrodes/mm². Only 1,020 can be read at once, and any electrode can be routed to any readout channel. So there is a choice between full 17.5 µm resolution over roughly six percent of the well, or about 89 µm effective pitch across the whole well. Gate 1 can use either. Gate 4 needs the dense patch, because route diversity is a spatial measure.

**Stimulation is not the bottleneck.** Up to 32 stimulation buffers can each be assigned to any of the 26,400 electrodes, with patterns, sequences and timing driven through the API. The multi-site extension below asks for four to six sites, so it sits well inside the hardware limit and the real cost is session length.

**Sampling rate differs by instrument.** MaxOne runs at 20 kHz, MaxTwo at 10 kHz. The first study needs twenty wells, which means MaxTwo and therefore 10 kHz. That is adequate for the threshold-crossing and multi-unit analysis this protocol specifies, and marginal for spike sorting. This protocol does not sort, so the constraint is accepted rather than worked around, and the single-unit longitudinal tracking that HD-MEA platforms are usually sold for is not part of the design.

**Closed-loop timing is available and not required.** Vendor guidance puts spike-to-stimulation latency at 12 to 15 ms, and [Kagan et al. 2022](https://www.cell.com/neuron/fulltext/S0896-6273(22)00806-6) demonstrated about 5 ms with custom processing on MaxOne. The shortest probe delay in this protocol is 250 ms, so latency is irrelevant to the core measurement. It does make one optional variant available: deliver the conditioning burst on a detected quiet network state rather than on a clock. Pre-stimulus baseline instability is the main thing that made the pilot's two sessions incomparable, so state-triggered delivery is worth testing as a variance-reduction measure once the fixed-clock version is working.

Nothing in the protocol depends on this particular vendor. Any array that offers scriptable multi-site stimulation, stable electrode addressing across sessions, and a sampling rate above about 10 kHz can run it.

## The protocol

One 20-minute session per preparation, non-destructive, repeatable on the same tissue.

1. Record three minutes of spontaneous activity at rest.
2. Deliver a conditioning burst: 20 Hz for 2 seconds, 40 biphasic cathodic-first pulses, 100 µs per phase.
3. Deliver single probe pulses at log-spaced delays from 0.25 s out to 32 s, and measure how much of the response has returned at each delay.
4. Record three minutes of spontaneous activity after.

Twelve cycles per session: eight with the conditioning burst, four interleaved control cycles that carry the same probe-delay sequence with no burst. The controls exist because the probe train has its own suppressive effect, and without them that effect is indistinguishable from the thing being measured.

Stimulus amplitude is calibrated per site to half-maximal response, and recalibrated every session rather than carried forward.

**Minimum trial count.** At least ten repeats of each condition per session. The pilot measured within-session split-half reliability at ten trials and got 0.78, so ten is adequate rather than a guess. Fewer than ten and a cross-session comparison cannot be interpreted, because the ceiling drops below the gate threshold and a failure becomes indistinguishable from noise.

**Fixed electrode routing across sessions.** On a switch-matrix array the standard practice is to run an activity scan each session and route the available channels to whatever looks most active. That must not be done here. The routing configuration is chosen once, from a single recording-only activity scan at the start of the study, saved, and reloaded verbatim for every subsequent session. If routing is re-derived per session, the electrode set differs between sessions and the cross-session comparison silently becomes a test of selection stability rather than recovery stability. This is the same failure as the pilot's conditioning-history confound: two things moving together with no way to separate them afterward.

**Electrode attrition is reported, not excluded.** With fixed routing, some routed electrodes will fall silent between sessions. The fraction that does is reported as its own number and those electrodes are not quietly dropped, because attrition is a fact about the tissue and excluding it biases the surviving set toward whatever was stable.

**No uncontrolled prior load.** A session must not be preceded by stimulation outside the protocol. The pilot's two sessions could not be cleanly compared because one followed three hours of maximum-amplitude stimulation that the other did not. Any conditioning history on the preparation is logged, and sessions are only compared across matched histories. Matched means matched program, not matched parameters: in the pilot recording one session carried the same parameter set as the sweep while delivering 54 stimuli per block instead of 10, which is a different experiment wearing the same label.

**Multi-site extension.** For the route-diversity measure, the burst is delivered at four to six distinct sites per session rather than one, and the return trajectories are compared. This multiplies session length rather than plate count and needs no change to the hardware.

## What comes out

Four numbers per session, from the probe responses against delay.

| Output | What it is | Why it is separate |
| --- | --- | --- |
| Resting response | how strongly the tissue answers a probe pulse before the burst | the baseline everything else is read against |
| Suppression depth | how far the burst knocks the response down | a drug can soften the blow without touching the rate |
| Recovery time | how long the response takes to climb back | the quantity of interest |
| Residual | how much has not returned by the last delay | incomplete recovery is a different failure from slow recovery |

Depth and rate are kept apart deliberately. A system that is knocked down further but recovers at the same rate is a different finding from one that is knocked down the same and takes longer to come back, and collapsing them into one score loses that.

**The response window is defined, not assumed.** A probe response is counted over a fixed window whose lower edge is the blanking period. The blanking period is measured once on the instrument, by recording the artifact on channels adjacent to a stimulation site, and then fixed for the study. The pilot used 3 ms, but that was on threshold-crossing data rather than continuous voltage and does not transfer to an HD-MEA at 10 or 20 kHz. The upper edge is fixed in advance and reported. A data-driven threshold is not used for the lower edge, because on sparse event data the median of the pre-stimulus histogram collapses to zero and the threshold goes with it.

**Recovery time is a threshold crossing, not a fit parameter.** The original version of this protocol specified a single exponential return, response at delay d equals one minus A times exp of minus d over tau, with tau as recovery time. The pilot showed that is the wrong model. The response to a single pulse is at least two-component: a sharp peak at 10 to 20 ms that decays within 60 ms, a second rise between 0.3 and 2 seconds, and a tail still elevated at 10 seconds. A single exponential forced onto that returns a number with no stable meaning.

So recovery time is the last time bin whose rate exceeds 1.5 times baseline, on log-spaced bins. The exponential fit is still computed and reported alongside, but it is not the primary measure and a disagreement between the two is a finding rather than an error.

**Two measures, one baseline-free.** A baseline-relative measure such as the threshold crossing moves when baseline moves, and baseline is not stable across days. So a baseline-free measure, time to half of peak response, is computed in parallel. The pilot found half-peak time unusable on that substrate, with a within-session ceiling of 0.16 against 0.78 for the threshold crossing, so which one is primary is decided per substrate by its measured ceiling and declared before the cross-session comparison.

**Censoring is reported, not silently dropped.** Responses that have not returned by the last delay are censored, not missing. The pilot censored about 21 percent of cells at a 10 second cutoff, which is the main argument for carrying the window out to 32 seconds. Any analysis states its censored fraction and fails if that fraction exceeds 30 percent.

## Pass / fail gates

Stated before any data exists. Each one can kill the program and is meant to.

**Gate 1 — is it repeatable?** Two sessions on consecutive days, untreated wells, matched conditioning history, identical electrode routing. Pass if recovery times correlate at 0.7 or better across sessions. Fail and nothing downstream is worth building.

Gate 1 is an acceptance test with five parts, and a bare correlation without them is not a result:

1. **Report the within-session ceiling.** Split-half reliability inside each session, Spearman-Brown corrected. A cross-session correlation is uninterpretable without it. 0.5 against a ceiling of 0.55 means the measure is noisy; 0.5 against a ceiling of 0.95 means the tissue changed. Pass requires a ceiling of at least 0.7 before the cross-session number is even read.
2. **Run a positive control.** A synthetic dataset carrying a known electrode-identity effect, pushed through the same pipeline. If it does not come back near 1.0, the pipeline is broken and the real number means nothing.
3. **Run a label-permutation null.** Shuffle electrode identity on the real data and recompute. The observed value has to clear the 95th percentile of that null.
4. **Center within condition before pooling.** If sessions contain several stimulus conditions, pooling cells across conditions without centering produces a high correlation that reflects the condition structure rather than electrode identity. Report the centered value as the headline and the raw pooled value separately.
5. **Count independent units honestly.** Electrodes in one culture are not independent samples. On an HD-MEA this is worse than on a low-density array, because at 17.5 µm pitch one neuron's field reaches several electrodes, so neighboring channels are partly the same signal. A result is reported as n cultures first, then wells, electrodes, conditions, sessions and trials. Two sessions on one culture cannot pass or fail this gate on its own, whatever the correlation comes out at.

**Gate 2 — is it a real rate?** The rate measured by pushing should agree with the rate read from the tissue's own spontaneous fluctuations. Pass if they correlate at 0.5 or better across wells. This is the gate that matters conceptually: it tests whether the active and passive routes to the same quantity actually agree, which the whole two-track program assumes and nobody has checked.

**Gate 3 — does it beat what already exists?** Pass if recovery time separates a drug dose from vehicle where firing rate and burst rate do not. Fail and this is a slower version of standard MEA software. This is also the commercial gate: a recovery readout that catches a liability signal the standard metrics miss has buyers in seizure and neurotoxicity screening today, independent of whether the aging claim holds.

**Gate 4 — are there distinct routes back?** With multi-site stimulation, pass if recovery trajectories from different stimulation sites stay reliably distinguishable in a preparation believed healthy. Collapse onto a common trajectory is the predicted signature of lost route diversity. This gate is exploratory: there is no established threshold, and the first study sets one.

Twenty wells and two weeks answer the first three. Gate 4 rides along inside the same study at the cost of session length.

## Pilot: what has been run

Nothing in the protocol above has been run on purpose-collected data. One pilot has been run on public data, and its only job was to exercise the method and find out what the protocol was failing to specify.

**The data.** FinalSpark Neuroplatform export fs369: 12 human iPSC-derived neurospheres on a 32-electrode MEA in a 4x8 grid, not a high-density array. Recorded from 2025-02-19 until the culture died on 2025-03-12, 21 days, cause of death logged as almost no activity. Events are threshold crossings at 6 times median SD; the raw file holds 3 ms snippets around each crossing rather than continuous voltage. Each day runs an identical 32-block dose-response sweep, 16 parameter conditions at two polarities, 10 trials per block, whole-array simultaneous stimulation. Two consecutive days, 2025-02-20 and 2025-02-21, run the same sweep and are the only adjacent pair not separated by a medium change.

**What was run.** The Gate 1 acceptance test above, on those two days, 24 active electrodes, 32 matched blocks.

**Result.** Condition-centered cross-session correlation of recovery time, 0.128 against a threshold of 0.7. Within-session ceiling 0.779. Positive control 0.878. Permutation null cleared, but far below the ceiling rather than near it.

**What it means.** The measure is reproducible inside a session and does not survive overnight in this preparation. The failure is interpretable rather than ambiguous, because the ceiling rules out too few trials and the positive control rules out a broken pipeline. Post-hoc, there is a gradient: peak response amplitude correlates across days at 0.80, electrode-averaged recovery time at 0.61, the per-cell value at 0.13. Something about each electrode persists overnight, strongest in response size, weaker in recovery time.

**What it does not mean.** This is not Gate 1 failing. Gate 1 requires matched conditioning history and more than one culture, and this had neither. Day 1 followed three hours of maximum-amplitude stimulation ending 7.4 hours before its sweep; day 2 did not. "The tissue changed overnight" and "day 1 was pre-conditioned" both explain the data and imply opposite things.

**The confound cannot be resolved in this dataset.** A census of the full recording settles that. Stimulation occurred on 7 of the roughly 21 recording days, no two of them adjacent, under a materially different program nearly every time: 2,040 pre-conditioning stimuli before the 02-20 sweep, a 2,127-stimulus session on 02-28 that also carries the medium change, 7,072 stimuli in seven large blocks on 03-04, and by 03-10 a degenerate pattern of 289 isolated single stimuli, two days before the culture died. Exactly two days run the sweep with nothing in the preceding 24 hours, 02-21 and 03-06, and those are 13 days apart across the medium change in a failing culture. So no pair in this recording isolates the overnight question. The reason is not bad luck about which days were pre-conditioned. It is that the recording was never designed as a repeated-measures experiment.

**What it changed in this protocol.** The minimum trial count, the matched-history requirement, the response-window definition, the replacement of the exponential fit with a threshold crossing, the parallel baseline-free measure, explicit censoring, and the five-part structure of the Gate 1 acceptance test. All of those were underspecified before the pilot and are specified now.

Code and outputs: `scripts/gate1_run.py`, `output/gate1/`, plus `scripts/day_census.py` for the stimulation-history census. Standing analysis rules: `docs/ai/design/analysis_standards.md`.

## Session budget

The first study is twenty wells over two weeks: dissociated rodent cortical culture, single-site stimulation for gates 1 to 3, multi-site on a subset for gate 4.

One single-site session is 20 minutes. Multi-site at four to six sites is one to two hours per well. Gate 1 needs two sessions on consecutive days. Gate 3 needs a dose arm and vehicle wells, randomized across the plate and across time of day.

That last point is not a detail. Two attempts to extract this measurement from public data have now failed, each for a different reason, and both reasons are design failures rather than analysis failures. In the first, five drug doses were recorded from one preparation across a single afternoon, so dose and elapsed time moved together and nothing could separate them. In the second, described above, the two comparable sessions differed in conditioning history, so tissue change and pre-conditioning moved together, and a census of the remaining nineteen days found no pair anywhere in the recording that separates them. Randomized dosing, vehicle wells, washout, logged conditioning history, matched session pairs and fixed electrode routing are in this design from the first plate for that reason.

The first study may not need the full spatial resolution of a high-density array. Gates 1 to 3 run on a sparse routing or on a conventional multiwell MEA. The dense routing becomes necessary at gate 4, where the measurement depends on spatial resolution.

## What I do not know

Grouped by who can answer it. Any one of these answered is worth the conversation.

**Can the hardware do this?** The electrode-to-channel ratio, the stimulation buffer count and the sampling rates are settled and are in the Platform section. These are not.

- How long does the stimulation artifact last on recording channels near a stimulation site, and does the acquisition software blank it or is that left to analysis? This sets the lower edge of the response window and no public documentation states it.
- Does each of the 32 stimulation buffers take an independent amplitude? Per-site calibration to half-maximal response depends on it.
- Can a routing configuration be saved and reloaded exactly across sessions, weeks apart? Gate 1 depends on this directly.
- Does MaxTwo provide 32 stimulation buffers per well, or 32 shared across all wells? This decides whether multi-site and multiwell can run in the same study.
- What is the charge injection limit per phase on an 11.5 × 11.5 µm PEDOT-coated platinum electrode? My arithmetic says bare platinum at 100 µC/cm² allows about 130 pC per phase, which is 1.3 µA at a 100 µs phase width, and PEDOT should buy roughly an order of magnitude more. That is uncomfortably close to the amplitudes needed to evoke a response, so it needs confirming or correcting rather than discovering on a plate.
- Does a 2-second 20 Hz burst at half-maximal amplitude degrade PEDOT-coated electrodes over repeated weekly sessions?
- What is the *active* neuron density after maturation, as opposed to the recommended seeding density of 6,000 to 15,000 cells/mm²? At 3,265 electrodes/mm² the seeding density exceeds the electrode density, so whether multiple electrodes see one neuron depends on survival and on field spread rather than on counts.
- What is realistic throughput per day, accounting for setup, calibration and media changes?

**Would this survive review?**

- Which controls are missing? The interleaved no-burst cycles handle the probe train's own effect, and the Gate 1 test above handles pipeline validity. What else?
- Is comparing the pushed rate against the rate inferred from spontaneous fluctuation a fair test, or is there a known reason those should not agree in tissue?
- Are 0.7 and 0.5 sensible thresholds for gates 1 and 2, or arbitrary?
- Baseline firing rate moved about 1.5x overnight in the pilot. Is that normal for a culture in decline, or a sign the recording conditions drifted?
- How long does a preparation need between sessions for conditioning history to stop mattering? The pilot suggests 7.4 hours is not enough, which affects session scheduling directly.
- For gate 4, is there an existing measure of trajectory distinguishability I should be using rather than inventing?
- What is the first thing you would attack about this design?

**What would it cost to run?**

- Can this be run as a service, and on what instrument?
- What would twenty wells over two weeks cost, with and without the multi-site extension?
- What format do raw recordings come back in?

## Who I am, and what I am asking for

Ryan J. McCall, PhD — Continuity Labs. Software engineer by background, previously Apple, Amazon and Google. I am building toward a benchmark where the endpoint is death rather than chronological age, with this assay as the fast half of it.

I have no lab. What I have is the design, the analysis, and a standing commitment to publish the protocol, the gates and the code openly whether they pass or fail.

Three things would each move this forward on their own:

1. Twenty minutes on a call to tell me where this breaks on real hardware.
2. A read of the gates — specifically whether the thresholds are defensible and what is missing.
3. A quote, or a pointer to whoever runs this kind of thing as a service.

I am not asking anyone to run an experiment for free. The useful thing right now is judgment, and I would rather be told this is wrong in October than find out in June.

## Sources

**Prior art for the measurement**

- [Tabak, Mascagni & Bertram, Effects of synaptic depression and recovery on synchronous network activity (2007)](https://pubmed.ncbi.nlm.nih.gov/17414972/)

**Prior art for the standardization pattern**

- [Blinova et al., International multisite study of human iPSC-derived cardiomyocytes for drug proarrhythmic potential assessment, Cell Reports (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6226030)
- [Millard et al., CiPA MEA pilot study, Toxicological Sciences](https://files.axionbiosystems.com/sites/default/files/millard_cipa_pilot_study_toxsci2108.pdf)
- [HESI NeuTox committee, MEA subteam update](https://hesiglobal.org/wp-content/uploads/2020/09/HESI-NeuTox-2020-MidYear-Update.pdf)

**Platform**

- [MaxOne HD-MEA specifications](https://www.mxwbio.com/products/maxone)
- [MaxWell CMOS switch-matrix architecture](https://www.mxwbio.com/our-technology)
- [MaxOne neuronal cell plating protocol](https://www.mxwbio.com/resources/maxone-maxone-neuronal-cell-plating-protocol)
- [Getting started with closed-loop experiments on HD-MEAs](https://webflow.mxwbio.com/updates/getting-started-with-closed-loop-experiments-on-hd-meas)
- [Kagan et al., In vitro neurons learn and exhibit sentience when embodied in a simulated game-world, Neuron (2022)](https://www.cell.com/neuron/fulltext/S0896-6273(22)00806-6)

**Hardware and stimulation**

- [Stimulation and artifact-suppression techniques for in-vitro high-density MEA systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC6711758/)
- [Single-cell electrical stimulation using CMOS-based high-density MEAs](https://pubmed.ncbi.nlm.nih.gov/30918481/)
- [3D high-density MEAs for acute slices, spheroids and organoids, PLOS ONE (2025)](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0328903)

**Substrate**

- [Directly reprogrammed neurons retain aging-associated signatures (Salk, 2015)](https://www.salk.edu/news-release/researchers-learn-how-to-grow-old-brain-cells/)
- [Long-term human organotypic brain slice culture protocol](https://www.biorxiv.org/content/10.1101/2023.10.10.561508.full.pdf)

**Pilot data**

- FinalSpark Neuroplatform whole-life export fs369, [finalspark.com](https://finalspark.com/)


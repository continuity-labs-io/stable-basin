# Tissue Recovery Profiling — Protocol Brief for Review

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

Two things in the protocol are unusual rather than borrowed. The probe delays sit between 0.25 and 32 seconds, which is the gap between paired-pulse work in milliseconds and plasticity work in minutes to hours — that band is thinly studied. And recovery time is treated as the primary readout rather than as a mechanistic probe of synaptic function, which is a reframing, not a discovery.

## Substrate

No preparation gives donor age, survival over weeks, and intact circuit architecture at the same time. The plan uses three, each for a different job.

| Preparation | What it is for | Donor age | Lives for | Architecture |
| --- | --- | --- | --- | --- |
| Dissociated rodent cortical culture, E18, plated on the array | Gate 1 — is the assay repeatable at all | none | months | none |
| Acute hippocampal or cortical slices, C57BL/6 at 3, 12 and 24 months | the aging comparison | real | 6–12 hours | yes |
| Directly converted induced neurons, young vs old human donors | the human arm | real | weeks | none |

The first is chosen because it has the lowest variance available, which is what you want when the thing under test is the instrument rather than the biology. The second is cross-sectional by necessity — one session per slice, so this compares animals rather than tracking a preparation. The third is the only route to human cells that carry the donor's actual age: iPSC reprogramming erases age signatures, direct conversion from fibroblasts retains them.

Organoids were considered and set aside: a three-dimensional body only contacts the electrodes it happens to sit on, batch variability is large relative to the effect, and there is no donor age, so there is no aging phenotype to measure.

Organotypic hippocampal slice culture sits between the first two — weeks of viability with laminar circuitry intact, neonatal donor — and may be the right substrate for the blockade work described below.

## The protocol

One 20-minute session per preparation, non-destructive, repeatable on the same tissue.

1. Record three minutes of spontaneous activity at rest.
2. Deliver a conditioning burst: 20 Hz for 2 seconds, 40 biphasic cathodic-first pulses, 100 µs per phase.
3. Deliver single probe pulses at log-spaced delays from 0.25 s out to 32 s, and measure how much of the response has returned at each delay.
4. Record three minutes of spontaneous activity after.

Twelve cycles per session: eight with the conditioning burst, four interleaved control cycles that carry the same probe-delay sequence with no burst. The controls exist because the probe train has its own suppressive effect, and without them that effect is indistinguishable from the thing being measured.

Stimulus amplitude is calibrated per site to half-maximal response, and recalibrated every session rather than carried forward.

**Multi-site extension.** For the route-diversity measure, the burst is delivered at four to six distinct sites per session rather than one, and the return trajectories are compared. This multiplies session length rather than plate count and needs no change to the hardware.

## What comes out

Four numbers per session, from fitting the probe responses against delay.

| Output | What it is | Why it is separate |
| --- | --- | --- |
| Resting response | how strongly the tissue answers a probe pulse before the burst | the baseline everything else is read against |
| Suppression depth | how far the burst knocks the response down | a drug can soften the blow without touching the rate |
| Recovery time | how long the response takes to climb back | the quantity of interest |
| Residual | how much has not returned by the last delay | incomplete recovery is a different failure from slow recovery |

The fit is a single exponential return: response at delay d equals one minus A times exp of minus d over tau, where A is suppression depth and tau is recovery time. The residual is what the fit leaves unexplained at the longest delay.

Depth and rate are kept apart deliberately. A system that is knocked down further but recovers at the same rate is a different finding from one that is knocked down the same and takes longer to come back, and collapsing them into one score loses that.

## Pass / fail gates

Stated before any data exists. Each one can kill the program and is meant to.

**Gate 1 — is it repeatable?** Two sessions on consecutive days, untreated wells. Pass if recovery times correlate at 0.7 or better across sessions. Fail and nothing downstream is worth building.

**Gate 2 — is it a real rate?** The rate measured by pushing should agree with the rate read from the tissue's own spontaneous fluctuations. Pass if they correlate at 0.5 or better across wells. This is the gate that matters conceptually: it tests whether the active and passive routes to the same quantity actually agree, which the whole two-track program assumes and nobody has checked.

**Gate 3 — does it beat what already exists?** Pass if recovery time separates a drug dose from vehicle where firing rate and burst rate do not. Fail and this is a slower version of standard MEA software. This is also the commercial gate: a recovery readout that catches a liability signal the standard metrics miss has buyers in seizure and neurotoxicity screening today, independent of whether the aging claim holds.

**Gate 4 — are there distinct routes back?** With multi-site stimulation, pass if recovery trajectories from different stimulation sites stay reliably distinguishable in a preparation believed healthy. Collapse onto a common trajectory is the predicted signature of lost route diversity. This gate is exploratory: there is no established threshold, and the first study sets one.

Twenty wells and two weeks answer the first three. Gate 4 rides along inside the same study at the cost of session length.

## Session budget

The first study is twenty wells over two weeks: dissociated rodent cortical culture, single-site stimulation for gates 1 to 3, multi-site on a subset for gate 4.

One single-site session is 20 minutes. Multi-site at four to six sites is one to two hours per well. Gate 1 needs two sessions on consecutive days. Gate 3 needs a dose arm and vehicle wells, randomized across the plate and across time of day.

That last point is not a detail. An earlier attempt to extract this measurement from public data failed because five drug doses were recorded from one preparation across a single afternoon, so dose and elapsed time moved together and nothing could separate them. Randomized dosing, vehicle wells and washout are in this design from the first plate for that reason.

The first study may not need a high-density array at all. Gate 1 runs on a conventional multiwell MEA. High density becomes necessary at gate 4, where the measurement depends on spatial resolution.

## What I do not know

Grouped by who can answer it. Any one of these answered is worth the conversation.

**Can the hardware do this?**

- At a 250 ms shortest probe delay, is stimulation artifact a non-issue, or does it still need blanking or subtraction?
- Is multi-site stimulation with per-site amplitude calibration scriptable, or is it a manual reconfiguration each time?
- What is realistic throughput per day, accounting for setup, calibration and media changes?
- Does a 2-second 20 Hz burst at half-maximal amplitude risk electrode degradation or tissue damage over repeated weekly sessions?
- Which of these parameters are the ones people usually get wrong the first time?

**Would this survive review?**

- Which controls are missing? The interleaved no-burst cycles handle the probe train's own effect — what else?
- Is comparing the pushed rate against the rate inferred from spontaneous fluctuation a fair test, or is there a known reason those should not agree in tissue?
- Are 0.7 and 0.5 sensible thresholds for gates 1 and 2, or arbitrary?
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

**Hardware and stimulation**

- [Stimulation and artifact-suppression techniques for in-vitro high-density MEA systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC6711758/)
- [Single-cell electrical stimulation using CMOS-based high-density MEAs](https://pubmed.ncbi.nlm.nih.gov/30918481/)
- [3D high-density MEAs for acute slices, spheroids and organoids, PLOS ONE (2025)](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0328903)

**Substrate**

- [Directly reprogrammed neurons retain aging-associated signatures (Salk, 2015)](https://www.salk.edu/news-release/researchers-learn-how-to-grow-old-brain-cells/)
- [Long-term human organotypic brain slice culture protocol](https://www.biorxiv.org/content/10.1101/2023.10.10.561508.full.pdf)

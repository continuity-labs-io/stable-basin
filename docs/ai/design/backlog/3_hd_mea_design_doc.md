# Stage 0 — Retrospective HD-MEA Analysis Design

What can be extracted from high-density microelectrode recordings already in hand, before any money is spent on new experiments.

## Purpose, and one honest limitation

The goal is to find out, using recordings that already exist, whether the recovery and route-diversity measures can be computed at all — before committing money to new experiments.

**The limitation that shapes everything below: the existing recordings have no controlled stimulation.** The assay as designed delivers a conditioning burst and probes the return. Public and previously collected HD-MEA data is almost entirely spontaneous activity. So the measure as specified cannot be run retrospectively.

The substitute is to treat **spontaneous network bursts as the perturbation**. A network burst is a large, stereotyped excursion followed by a return to quiescence, and that return is a recovery. Every recording with network bursting therefore contains hundreds of uncontrolled perturbation–recovery events, for free.

This is exactly the passive route described in the talk — inferring the recovery rate from the system's own fluctuations rather than measuring it by pushing. Using it here is not a compromise of the thesis; it is the half of the thesis that can be tested without a lab.

Three consequences follow, and they are design constraints rather than caveats.

- The perturbation is not chosen, so burst amplitude and the preceding quiet interval vary from event to event. Both will affect the recovery, so both must be conditioned on rather than averaged over.
- That variation is itself useful. Plotting recovery against the preceding quiescent interval is a **restitution curve obtained for free**, which is the same measurement cardiac electrophysiology makes with paced S1–S2 protocols.
- Any result here is about whether the analysis machinery works and what the variance looks like. It cannot establish that recovery tracks aging, because these recordings have no aged tissue and no mortality outcome.

**What would make this stage a failure:** the measures cannot be computed reliably on real recordings, or they can be computed but are not reproducible within a single recording. Either outcome kills gate 4 before a dollar is spent, which is the point.

## The data abstraction

Every dataset enters through one loader interface and produces one in-memory representation. Nothing downstream ever knows which file format or which lab the data came from. This is the single most valuable piece of work in the project and the most delegable.

**The loader contract.** A loader takes a path and returns a `Recording` with exactly these fields, or raises:

| Field | Type | Notes |
| --- | --- | --- |
| `recording_id` | string | stable, unique, derived from the source file |
| `spikes` | array of (time_s, electrode_index) | sorted by time; multi-unit per electrode is acceptable |
| `electrode_xy` | array of (x_um, y_um) | physical positions; required, not optional |
| `duration_s` | float | wall-clock length of the recording |
| `sample_rate_hz` | float | of the underlying raw signal |
| `n_electrodes` | int | total on the array, including silent ones |
| `stim_events` | array of (time_s, electrode_index, amplitude) or empty | empty is the expected case |
| `meta` | dict | everything in the registry below |

Electrode positions are mandatory. Every spatial measure in tier 4 depends on them, and a dataset that cannot supply them is out of scope rather than partially usable.

**The metadata registry.** One row per recording, in a single table, filled before any analysis runs. A missing value is recorded as missing rather than guessed.

- Source dataset, publication or DOI, and licence
- Preparation: species, culture type, donor age if known, days in vitro
- Chip or array identifier, well identifier, plate identifier
- Recording date and **clock time of day**
- Treatment, dose, vehicle flag, and time since treatment
- Operator or lab, if known

Time of day and chip identity are in the registry because the diazepam reanalysis failed on exactly those two axes. Anything that could differ systematically between the groups being compared belongs here before anyone looks at a result.

**Output contract.** Every analysis writes a tidy table of one row per event or per recording, plus a JSON sidecar recording the code version, the parameter values and the random seed. No figure is ever produced from anything but one of those tables.

## Validation comes first

None of the measures below are written until the controls exist. This project has already been burned twice by the same failure — a null result that turned out to be a broken pipeline rather than absent biology — and both times it cost weeks. **A null is uninterpretable unless the pipeline has been shown to find a signal that is known to be there.**

**Synthetic positive control.** Generate spike trains with a known recovery time constant and a known spatial propagation pattern. The pipeline must recover both within a stated tolerance. This runs in continuous integration on every commit. If it ever fails, no real-data result is trusted until it passes again.

**Synthetic negative control.** Generate spike trains with matched firing rates and burst structure but no spatial organisation and no recovery structure. The pipeline must return chance. A pipeline that finds structure in this is finding it in the method, not the data.

**Shuffle controls on real data.** Three, each destroying one thing and preserving the rest.

1. **Electrode identity shuffle** — permute which electrode each spike train belongs to. Destroys spatial structure, preserves every rate and every temporal statistic. This is the null for every tier 4 measure.
2. **Burst-time shuffle** — permute which burst each recovery window follows. Destroys the link between a burst and its own recovery.
3. **Circular shift** — roll each electrode's spike train by a random offset. Destroys cross-electrode timing, preserves each electrode's own statistics.

The null distribution is built by running the **entire pipeline** on shuffled data, not by shuffling at the last step. The earlier failure came precisely from a null computed at the wrong stage.

**Split-half reliability.** Within a single recording, compute every measure on odd-numbered bursts and on even-numbered bursts separately, and correlate. A measure that is not reproducible within one recording cannot possibly separate groups, and should be dropped before it reaches a figure. This is the cheapest and most informative test in the document.

**Blinding.** Group labels stay out of the analysis environment until the pipeline is frozen and the parameter choices are written down. Treatment and age columns are joined from the registry only at the final comparison step.

**The confound register.** Before any group comparison, each candidate confound is tested against the outcome on its own: recording duration, number of active electrodes, clock time of day, chip and plate identity, days in vitro, and operator. Any confound that predicts the measure gets reported alongside the result, whether or not it changes the conclusion.

## The measure ladder

Five tiers. Each has a gate, and a failed gate stops the climb rather than being worked around.

**Tier 1 — is there usable data?** Per recording: number of electrodes with any spiking, median firing rate across active electrodes, total recording duration, and whether network bursts are detectable at all. *Gate: at least 20 recordings with network bursting and at least 50 bursts each.* Below that there is nothing to analyse and the honest output of this project is a one-line note saying so.

**Tier 2 — recovery after a spontaneous burst.** For each detected burst, take the population firing rate in the window following it and fit the return to baseline. Three numbers per burst: how far activity drops below baseline, how long it takes to come back, and how much has not returned by the end of the window. These are the same four outputs the assay specifies, minus the resting response, which has no equivalent here. *Gate: split-half reliability of the recovery time constant at 0.6 or better within a recording.*

**Tier 3 — natural restitution.** Plot each burst's recovery and amplitude against the length of the preceding quiescent interval. A system with margin should look flat; a system near its limit should show recovery lengthening steeply as the preceding interval shortens. *Gate: the relationship is monotonic and reproducible across recordings from the same preparation.* This tier is the one most likely to produce something publishable on its own, because it is a known cardiac measurement made in neural tissue without a pacing rig.

**Tier 4 — spatial stereotypy.** Three separate questions, in increasing difficulty.

1. **Initiation repertoire.** Where does each burst start? Count the distinct initiation sites and compute the effective number of them — a system using many sites has a larger repertoire than one that always starts in the same place.
2. **Propagation conservation.** Does the spatial order in which electrodes recruit stay the same burst to burst? Rank-correlate recruitment order across burst pairs.
3. **Recovery-order conservation.** After the burst, in what order do electrodes return to firing, and is that order conserved? This is the retrospective stand-in for the route-diversity measure.

*Gate for all three: the measure exceeds the electrode-identity shuffle null, and split-half reliability is 0.5 or better.* If the shuffle null is not cleared, the measure is detecting array geometry rather than tissue organisation, and tier 4 is dead.

**Tier 5 — does anything separate groups?** Only where the registry supplies a real contrast: dose versus vehicle, or days in vitro, or donor age if any dataset has it. Every confound from the register is reported alongside. *Gate: the separation survives the confound adjustment.* Expect this tier to be underpowered — the point is to measure the effect size and the variance so the live experiment can be sized properly, not to prove anything.

## Trade-offs to settle before writing code

If the Census above passes, the next step is making six decisions. Each is cheap to make now and expensive to revisit after the pipeline exists.

**Spike sorting, or multi-unit per electrode?** Sorting HD-MEA data into single units is slow, introduces its own errors, and the errors are not independent across conditions. Every measure in this document is a network-level quantity. *Recommendation: multi-unit per electrode, no sorting.* Revisit only if tier 4 fails and there is reason to think unit identity is the missing piece.

**Which burst detector?** There are many published methods and the results will depend on the choice. *Recommendation: pick one published method, pre-register it, then demonstrate the headline result survives two alternatives.* Reporting only the method that worked is the failure mode this project is built to avoid.

**Fit the recovery, or measure it non-parametrically?** An exponential fit gives one interpretable number but assumes a shape the data may not have. A threshold crossing — time to return to some fraction of baseline — assumes nothing but is noisier. *Recommendation: compute both from the start.* They should agree; where they disagree, the disagreement is informative.

**Validate against synthetic truth, or against a published result?** Synthetic truth proves correctness. Reproducing a published figure from the same raw data proves the loader and preprocessing are right, which synthetic data cannot. *Recommendation: both, and the published reproduction first, because it is the one that catches format and unit errors.*

**How much is pre-registered?** Full pre-registration of every parameter is slow and will be wrong in places. No pre-registration makes every result unfalsifiable. *Recommendation: freeze and publish the gates and the primary measure definitions; leave secondary exploration explicitly labelled as exploratory.*

**Build for these datasets, or build for the live experiment?** Writing the loader to handle only what is in hand is faster; writing for the future recordings risks over-engineering something that may never run. *Recommendation: the `Recording` contract above is the forward-compatible piece — it already carries a `stim_events` field that is empty today. Everything else is built for the data in hand.*

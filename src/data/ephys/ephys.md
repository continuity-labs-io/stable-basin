# Dataset Inventory — Tissue Recovery Profiling

Candidate electrophysiology datasets for **Stage 0**: the retrospective analysis that
tests whether the recovery and route-diversity measures can be computed at all,
before any money is spent on new recordings.

Each entry is scored on what it can actually contribute to the measure ladder in the
Stage 0 design doc. Earlier versions of this file justified datasets on hardware and
modeling grounds; those criteria no longer apply.

---

## Evaluation criteria

A dataset is only useful here to the extent it can serve one of the five tiers.

| Tier | What it asks | What a dataset must have |
| --- | --- | --- |
| 1 | Is there usable data? | ≥20 recordings with detectable network bursting, ≥50 bursts each |
| 2 | Recovery after a spontaneous burst | Spike times, population-level bursting, adequate duration |
| 3 | Natural restitution | Many bursts with varying inter-burst intervals |
| 4 | Spatial stereotypy | **Electrode XY positions** and enough electrodes to resolve propagation |
| 5 | Group separation | A real contrast in metadata (dose vs vehicle, age, DIV) without confounds |

Electrode positions are mandatory for tier 4. A dataset that cannot supply them is
out of scope for that tier rather than partially usable.

---

## Ranking

| Rank | Dataset | Best for | Dead for |
| --- | --- | --- | --- |
| 1 | FinalSpark whole-life telemetry | Tiers 2–3, logged perturbations, lifespan axis | Tier 4 if electrode count is low — **unresolved** |
| 2 | Pharmacological shock (Sharf 2022) | Tiers 1–4, pipeline development | Tier 5 — fatally confounded |
| 3 | 3Brain NEUROPulse 4,096-ch | Tier 4 development, spatial pipeline | Tier 1 if only one recording |
| — | MaxWell HD-MEA (generic) | Not a dataset — a loader target | — |
| — | Continuous UHD-LFP | Not a dataset — the synthetic control generator | — |
| — | Spike Prophecy / Steinmetz | Nothing here. Different project. | All tiers |

---

## 1. FinalSpark Whole-Life Telemetry

- **Origin:** Open, remotely accessible Neuroplatform for wetware computing research
- **Source URLs:**
  - Project: [https://finalspark.com/neuroplatform/](https://finalspark.com/neuroplatform/)
  - API data: `fs369_package.hdf5`
- **Sampling & format:** 30 kHz raw activity samples, HDF5 with a `.parquet`
  segment index
- **Modalities:** Electrophysiology paired with incubator environment monitoring —
  temperature, humidity, CO₂, O₂, pressure, and **door_opening** — spanning the
  experiment lifecycle from start to endpoint

### Applicability to Tissue Recovery Profiling

**The only dataset in this inventory with logged perturbations.** Incubator door
openings are thermal, vibrational and gas-composition disturbances recorded with a
timestamp. That is not a controlled push, but it is a *dated* one, which partly
dissolves the central limitation of retrospective analysis: recovery can be measured
from a known disturbance rather than only from spontaneous bursts.

It is also the only dataset with a lifespan axis — recordings running to endpoint —
which is the structure the whole organism-track argument depends on.

- **Tier 1:** Likely the only candidate that clears it, if whole-life telemetry spans
  many sessions. **Unverified.**
- **Tiers 2–3:** Strong. Long continuous recordings with many bursts and variable
  inter-burst intervals.
- **Tier 4:** Depends entirely on electrode count per organoid. If the platform uses a
  low-count MEA rather than a high-density CMOS array, propagation order and
  initiation repertoire cannot be resolved and tier 4 is out.
- **Tier 5:** Door-opening events are the contrast. Environmental covariates are
  already recorded, which is rare and useful.

### Open questions

- [ ] How many electrodes per organoid, and are XY positions available?
- [ ] How many distinct recordings/sessions, and how many show network bursting?
- [ ] Are door-opening timestamps aligned to the ephys clock?
- [ ] What marks "endpoint" in the data, and is it a consistent criterion?

### Status

- [ ] Downloaded
- [ ] Loader written
- [ ] Tier 1 counts run

---

## 2. Pharmacological Shock

- **Origin:** *Functional neuronal circuitry and oscillatory dynamics in human brain
  organoids* (Nature Communications, 2022)
- **Source URLs:**
  - Paper: [https://www.nature.com/articles/s41467-022-32115-4](https://www.nature.com/articles/s41467-022-32115-4)
  - Data (Dryad): [https://datadryad.org/dataset/doi:10.25349/D9031Z](https://datadryad.org/dataset/doi:10.25349/D9031Z)
- **Sampling & format:** Continuous extracellular activity on high-density CMOS
  microelectrode arrays
- **Modalities:** Human brain organoid slices under pharmacological intervention,
  including diazepam dose series

### Applicability to Tissue Recovery Profiling

**The development substrate.** Real HD-MEA geometry, bursting tissue, published and
reproducible, and a loader already exists. This is where tiers 1 through 4 get built,
debugged and validated.

- **Tiers 1–4:** Good. Electrode positions are real, density is sufficient for
  propagation measures, and bursting is present.
- **Tier 5: fatally confounded.** Five doses from one organoid across a single
  afternoon, no vehicle well. Dose and elapsed time move together and cannot be
  separated. Do not attempt to rescue this — recovery time falling with dose is
  exactly the result we want and exactly the result this design cannot support.

This dataset is the reason the metadata registry records clock time of day.

### Status

- [ ] Downloaded
- [ ] Loader written
- [ ] Tier 1 counts run

---

## 3. 3Brain HD-MEA 4,096-Channel (NEUROPulse)

- **Origin:** HD-MEA NEUROPulse
- **Source URLs:**
  - Data (Zenodo): [https://zenodo.org/records/13908319](https://zenodo.org/records/13908319)
- **Sampling & format:** 20 kHz continuous telemetry, raw BrainWave format
  (`hdmea_neuropulse.brw`)
- **Modalities:** 4,096-channel BioCAM/Accura spatial grid

### Applicability to Tissue Recovery Profiling

**The best spatial resolution in the inventory, and probably too few recordings to
conclude anything.** A 64×64 grid means electrode positions come free from the
geometry, which makes this the ideal substrate for *developing* the tier 4 measures —
bugs in a spatial pipeline are visible at this density and invisible at low density.

- **Tier 1:** Likely fails. If this is a single stress-test file, it cannot reach
  twenty recordings.
- **Tier 4:** Excellent for development. Run the measure here, then reproduce it on a
  dataset with replication.
- **Tier 5:** No treatment metadata described.

### Open questions

- [ ] How many recordings are actually in the Zenodo record?
- [ ] Python reader for `.brw` — which one, and does it round-trip correctly?

### Status

- [ ] Downloaded
- [ ] Loader written
- [ ] Tier 1 counts run

---

## 4. MaxWell HD-MEA — *not a dataset*

- **Format:** Continuous-time voltage arrays in HDF5 (`.raw.h5`)

This is a file format, not a dataset. Its value is that the `.raw.h5` loader is the
single most reusable asset in the project and the format most future recordings will
arrive in.

**Reclassified as a loader target.** Build it first, to the `Recording` contract in
the Stage 0 design doc, and it serves every MaxWell recording that ever comes in —
including from the live experiment.

---

## 5. Continuous UHD-LFP — *not a dataset*

- **Origin:** Simulated macroscopic LFP across a 2D grid, generated as spatial tensors

Simulated data is not evidence. But the Stage 0 design doc requires a synthetic
positive and negative control **before any analysis runs**, and this is existing
machinery for generating spatial structure with a known ground truth.

**Reclassified as the synthetic control generator.** Needs two modes:

- **Positive:** known recovery time constant and known spatial propagation pattern.
  The pipeline must recover both within tolerance.
- **Negative:** matched firing rates and burst structure, no spatial organisation,
  no recovery structure. The pipeline must return chance.

Both run in CI on every commit. Nothing on real data is trusted while either fails.

---

## 6. Spike Prophecy (Steinmetz) — *dismissed*

- **Origin:** [https://arxiv.org/html/2605.12992](https://arxiv.org/html/2605.12992),
  built on the Steinmetz dataset
- **Modalities:** In vivo multi-region Neuropixels from behaving mice

**Not applicable.** No in vitro network bursting, no tissue-level perturbation, and
electrodes sit along a shank through 3D brain rather than on a plane — so initiation
repertoire, propagation order and recovery order have no meaning. The entire measure
framework fails to transfer.

Retained in the broader state-space modeling work. Out of scope here.

---

## First job — before any loader is written

None of these datasets obviously clears the tier 1 gate. Sharf is one organoid.
NEUROPulse is probably one file. FinalSpark is unknown and is the one that might.

So the first task is a counting exercise across the three real datasets:

- [ ] Number of recordings per dataset
- [ ] Number with detectable network bursting
- [ ] Bursts per recording
- [ ] Electrode count, and whether XY positions exist
- [ ] Total usable duration

One day of work. It either confirms the plan or says the tier 1 gate needs rewriting —
and finding that out in week one rather than week six is the reason the gate was
written down first.

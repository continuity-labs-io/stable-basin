# Minimum Dataset Specification

Version 1.0
Status: normative
Implemented by: `src/benchmarks/recovery/contract.py`

Defines the smallest dataset from which a within-session reliability ceiling can be computed for a stimulation-evoked measure.

The key words MUST, MUST NOT, SHOULD and MAY are to be interpreted as in RFC 2119.

---

## 1. Terms

**Preparation** — one physical specimen: a culture, well, construct or slice.

**Session** — one continuous acquisition from one preparation, with no intervening medium change, incubator opening or handling event.

**Recording unit** — one spatially distinct channel of measurement: an electrode, ROI, well or pixel group. The unit is the axis across which the reliability correlation is computed.

**Trial** — one stimulus delivery and the recorded response that follows it.

**Condition** — a stimulus parameter set. Two trials share a condition only if every stimulus parameter is equal.

**Measure** — a scalar computed per unit per trial from the post-stimulus response.

---

## 2. The computation this supports

The ceiling is a split-half correlation with Spearman-Brown correction. Trials within one session are partitioned into two halves; the measure is computed independently on each half, giving two vectors indexed by recording unit; the two vectors are correlated across units; the result is corrected by

    rho_full = 2 * rho_half / (1 + rho_half)

Every requirement in section 3 follows from this procedure.

---

## 3. Requirements

**MD-1 — Single session.**
The dataset MUST contain exactly one session from exactly one preparation. Longitudinal data is out of scope for this computation.

**MD-2 — Repeated trials.**
The dataset MUST contain at least 10 trials of at least one condition. It SHOULD contain 20 or more. Below 10, the two halves are too short for the correlation to be stable.

**MD-3 — Multiple recording units.**
The dataset MUST contain at least 8 recording units that are active in every trial of the evaluated condition. It SHOULD contain 15 or more. A single unit cannot produce a correlation.

**MD-4 — Stimulus timestamps.**
The dataset MUST provide the onset time of every stimulus, on the same clock as the response data. Timestamps MUST be strictly increasing within a condition.

**MD-5 — Trial-level resolution.**
The dataset MUST provide per-trial data: spike times, event times or raw signal. Summary statistics aggregated over trials do not satisfy this requirement and cannot be partitioned into halves.

**MD-6 — Post-stimulus window.**
Each trial MUST carry at least 2.0 seconds of recording after stimulus onset. This is the requirement that most often fails, because many stimulation protocols use short inter-stimulus intervals. A protocol truncating at 200 ms cannot resolve a seconds-scale return.

---

## 4. Not required

Multiple sessions. Conditioning bursts. The probe ladder. Any particular tissue type, species or cell source. Any particular instrument. Any particular modality: calcium imaging satisfies MD-1 through MD-6 as readily as an electrode array. Any aging-related metadata.

---

## 5. Optional, in order of value

**OD-1** — Multiple stimulus amplitudes. Permits direct evaluation of the dose-response curve and the titration rule, not only the ceiling.

**OD-2** — More than one session on the same preparation. Promotes the question from "is the measure capable" to "does it repeat."

**OD-3** — Both stimulus polarities. Separates artifact from response.

**OD-4** — Preparation metadata: age, cell source, days in vitro, medium changes, handling events. Medium changes are known to be material.

**OD-5** — Raw signal rather than detected events. Events cannot be un-thresholded; raw preserves every amplitude measure.

---

## 6. Format

Any. HDF5, NWB, .mat, Axion AxIS, or CSV of event times with a separate stimulus timestamp file. Conformance is defined by content, not container.

---

## 7. Size

A conforming dataset sent as events is on the order of megabytes: 10 trials across 20 units with 2 seconds of post-stimulus recording.

Raw voltage is not. At 20 kHz across 1,020 channels it is approximately 40 MB per second of recording, and a full raw session SHOULD be negotiated separately from the minimum dataset.

---

## 8. Conformance

A dataset conforms when MD-1 through MD-6 all hold. `contract.validate()` raises on the first violation and names the requirement ID, the value found, and the value required.

Conformance is necessary, not sufficient. A conforming dataset supports the computation; it does not guarantee the measure clears any threshold.

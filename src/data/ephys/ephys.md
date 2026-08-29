# Dataset Inventory & MVM Justification

## 1. FinalSpark Whole-Life Telemetry

- **Origin:** Open and remotely accessible Neuroplatform for research in wetware
  computing
- **Source URLs:**
  - Project:
    [https://finalspark.com/neuroplatform/](https://finalspark.com/neuroplatform/)
  - API Data: `fs369_package.hdf5`[cite: 1]
- **Sampling & Format:** 30kHz-resolution raw activity samples[cite: 1]. Data is
  provided in HDF5 format with a `.parquet` segment index[cite: 1].
- **Modalities:** High-frequency electrophysiology paired with incubator
  environment monitoring (temperature, humidity, CO2, O2, pressure,
  door_opening)[cite: 1]. This spans the experiment lifecycle from start to
  endpoint[cite: 1].
- **The MVM Proof (Multi-Modal Trajectory):** Validates that the Invariant Core
  can fuse microsecond electrical telemetry with macroscopic environmental
  shifts (like incubator door openings). It demonstrates how environmental
  perturbations alter the thermodynamic vector and predict the ultimate lifespan
  of the organoid.

## 2. Pharmacological Shock

- **Origin:** _Functional neuronal circuitry and oscillatory dynamics in human
  brain organoids_ (Nature Communications, 2022)
- **Source URLs:**
  - Paper:
    [https://www.nature.com/articles/s41467-022-32115-4](https://www.nature.com/articles/s41467-022-32115-4)
  - Data (Dryad):
    [https://datadryad.org/dataset/doi:10.25349/D9031Z](https://datadryad.org/dataset/doi:10.25349/D9031Z)
- **Sampling & Format:** Continuous extracellular neural activity recorded via
  high-density CMOS microelectrode spatial arrays.
- **Modalities:** Human brain organoid slices subjected to direct
  pharmacological interventions, specifically capturing the neural network's
  response to drugs like Diazepam.
- **The MVM Proof (Phase Transition):** Serves as the ultimate ground-truth test
  for the continuous-time solver. Instead of just counting dropped spikes, this
  dataset proves the architecture can detect the exact millisecond the
  pharmacological agent collapses the network's Kinetic Stability Metric (KSM)
  from 1.0 down to 0.0.

## 3. 3Brain HD-MEA 4,096-Channel Stress Test

- **Origin:** _HD-MEA NEUROPulse_
- **Source URLs:**
  - Data (Zenodo):
    [https://zenodo.org/records/13908319](https://zenodo.org/records/13908319)
- **Sampling & Format:** 20kHz continuous telemetry stored in raw BrainWave
  format (`hdmea_neuropulse.brw`).
- **Modalities:** High-density spatial grid recorded from 4,096-channel
  BioCAM/Accura hardware arrays.
- **The MVM Proof (Hardware Scale):** The ultimate engineering benchmark. By
  piping this massive spatial grid into the state-space engine, you generate the
  visual proof that the architecture maintains a linear, O(1) VRAM footprint
  without triggering Out-Of-Memory crashes, defeating standard Transformer
  models side-by-side.

## 4. MaxWell HD-MEA (Generic)

- **Origin:** MaxWell Biosystems high-density microelectrode arrays.
- **Sampling & Format:** Continuous-time voltage arrays stored in HDF5
  (`.raw.h5`) format.
- **Modalities:** Multi-channel electrophysiological telemetry.
- **The MVM Proof (Generalization):** Proves that the neural integration
  architecture is hardware-agnostic, capable of dynamically scaling to ingest
  raw multidimensional continuous-time signals natively from MaxWell hardware
  arrays without manual feature engineering.

## 5. Spike Prophecy (Steinmetz)

- **Origin:** _Spike Prophecy_ (https://arxiv.org/html/2605.12992) based on the
  Steinmetz dataset.
- **Sampling & Format:** Multi-unit spike arrays, historically binned into
  discrete time steps (e.g., 50ms) but ingested as continuous streams.
- **Modalities:** In-vivo multi-region Neuropixels recordings from behaving
  mice.
- **The MVM Proof (Substrate Independence):** Demonstrates that the state-space
  engine can decode complex biological behavior from discrete spike trains
  across completely different brain regions (in vivo) natively, without
  requiring task-specific architectural changes.

## 6. Continuous UHD-LFP

- **Origin:** Simulated macroscopic continuous Local Field Potential (LFP)
  mapped across a 2D grid.
- **Sampling & Format:** Continuous electromagnetic waves generated as spatial
  tensors.
- **Modalities:** Macroscopic brain waves and continuous-time phase transitions.
- **The MVM Proof (Ephaptic Lock-in):** As established in TopoEncoder tests,
  this proves that a "thought" is a measurable thermodynamic phase transition.
  The continuous LFP data allows the Mamba-2 engine to mathematically track and
  align the geometry of this continuous wave directly to a semantic concept
  (using InfoNCE mathematics).

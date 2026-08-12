# MELD Data Pipeline: The Sim2Real Bridge

Standard computational biology trains discrete ML models (like Transformers) on
dead cells to predict end-states. It is the equivalent of trying to understand a
plane crash by looking at the debris.

Project MELD is building a **Continuous-Time Biological Flight Data Recorder**.
We are engineering a State-Space AI to predict the exact chronological minute a
living human brain network crosses the thermodynamic point of no return into
senescence.

Because the full optical hardware pipeline is still being assembled, this
directory houses the **Simulation-to-Reality (Sim2Real) Bridge**. We have
scaffolded the exact multi-scale tensor our hardware will produce, and we need
data engineers to swap our synthetic math with empirical proxy datasets.

## The 114-Dimensional State Vector (The Trifecta)

Our architecture films living tissue across three entirely different physical
and temporal scales, yielding a 114-Dimensional state vector per cell, per
timestamp:

- **Sigma — The Hardware (100-D):** The physical 3D shape and density of the
  cell, compressed via Spatial VAE. Updates continuously every **minute**.
- **Psi — The Software (12-D):** The 12 Waddington RNA anchor genes (e.g.,
  _TP53, TREM2_). Measured via sparse flashes every **5 minutes** (leaving
  massive `NaN` gaps in between).
- **Omega — The Trigger (2-D):** The bioelectric voltage of the membrane.
  Recorded via micro-bursts at **>500 Hz (milliseconds)**.

## The ML Objective: Hunting the Crash

In a healthy cell, these three layers are tied together. The slow RNA (Psi)
builds the physical Shape (Sigma), which acts as a shock absorber to control the
fast Voltage (Omega).

As the cell crashes into senescence, this entrainment breaks. We are not
predicting "death." We are building models to detect the exact minute these
signatures occur:

1. **Predictability Drop:** The physical shape loses its grip, and the structure
   can no longer predict the voltage.
2. **Electrical Jitter:** The physical shock-absorbers evaporate, and the
   electrical baseline begins to violently flicker.
3. **Bounce-Back Time:** The RNA stress alarms turn ON, but the cell loses its
   elasticity and the alarms get stuck in the ON position forever.

## The Engineering Bounties (Pick Your Module)

Inside this directory, you will find three dataloader scaffolds. They currently
run on synthetic Python physics. We need Data Engineers, Bioinformaticians, and
Signal Processing experts to upgrade the `[TASK]` blocks inside the code with
real-world proxy data.

- 👁️ **`sigma_phase_structure_dataloader.py`**
  - **The Bounty:** We need Computer Vision engineers to pull real Quantitative
    Phase Imaging (QPI) from CZ Biohub OME-Zarr AWS buckets, run 3D
    segmentation, and use a **Latent ODE or Gaussian Process** to map 1-minute
    morphological drift to a millisecond AI clock.
- 🧬 **`psi_rna_dataloader.py`**
  - **The Bounty:** We need Single-Cell Genomics engineers to pull Tony
    Wyss-Coray / CZ Biohub `.h5ad` AnnData files. You must use **Palantir or
    Waddington-OT** to mathematically warp dead-cell snapshots onto our
    continuous live-cell timeline without forward-filling the `NaN` gaps.
- ⚡ **`omega_bioelectric_dataloader.py`**
  - **The Bounty:** We need High-Frequency Signal Processing engineers to pull
    real kilohertz optical patch-clamp data from **Neurodata Without Borders
    (NWB)**. You must verify our Ratiometric Motion-Cancellation math and inject
    true biological action potential kinematics.

## Getting Started

Run the master Sim2Real builder to generate a synthetic baseline tensor:

```bash
python data_pipeline/meld_sim2real_builder.py
```

This will output `data/MELD_Xi114_Sim2Real_Stub.csv`. Look at the output. You
will immediately see the multi-scale time problem (500Hz data living next to
massive NaN gaps). Standard Transformers will OOM-crash on this (see
baselines/transformer.ipynb). Welcome to continuous-time biology.

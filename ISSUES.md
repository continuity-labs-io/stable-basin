# Stable Basin Benchmark Issues & Technical Debt

This document tracks known issues, technical debt, and open TODOs within the
codebase.

## Publications

### Paper 1: The Core Foundation (Our Current Focus)
**"Latent Stasis: Mask-Aware Subspace Routing for Asynchronous State Space Models"**
* **The Vibe:** Hardcore Machine Learning Architecture.
* **The Problem:** Sensor fusion in continuous-time AI fails because zero-padding asynchronous inputs actively decays the latent state.
* **The Solution:** The Orthogonal Subspace Router dynamically modulates the $\Delta t$ and $B$ parameters to mathematically freeze hidden channels when sensors drop offline.
* **Why it Matters:** This establishes you as the engineer who solved the missing-data problem for modern state-space models. It requires no wet-lab data, making it fast to publish in top ML venues.

### Paper 2: The Biological Physics Engine
**"Hierarchical State-Space Models for Macroscopic Biological Entrainment"**
* **The Vibe:** Computational Neuroscience / Systems Biology.
* **The Problem:** Single-layer recurrent models act as passive filters; they process noise but cannot spontaneously generate higher-order biological structures (like traveling waves or synchronized rhythms).
* **The Solution:** Your `hierarchical_ssm.py` module. You prove that by coupling a fast/local layer with a delayed, top-down macro layer, you trigger a thermodynamic phase transition into a stable Attractor Limit Cycle.
* **Why it Matters:** This is the deep physics paper. It appeals to theoretical neuroscience, laying the mathematical groundwork for mapping a biological "thought" into a digital latent space.

## Overhaul `src/modules/hierarchical_ssm.py` for Prime Time

The `HierarchicalSSM` is currently a frozen mathematical toy simulator demonstrating standing wave phase transitions. It needs to be upgraded for actual training and multi-sensor fusion:

1.  **Make Parameters Learnable**: Remove `requires_grad=False` from the core matrices (`A1`, `B1`, `A2`, `B2`, `W_td`).
2.  **Discretize for Parallel Scans**: The current explicit Euler integration (`for t in range(steps):`) is unacceptably slow for GPUs. Implement a discretization step (e.g., Zero-Order Hold or Bilinear Transform) so the continuous ODE can be unrolled via an associative parallel scan (like the Mamba architecture).
3.  **Implement Multi-Rate Multi-Sensor Polling**: Upgrade the hardcoded single scalar input `u` to accept a multimodal block tensor. Modulate `dt` or assign different blocks of the `A` matrix to different polling rates (e.g., Layer 1 catching 20kHz electrical spikes, Layer 2 catching 1Hz RNA reads) to prove out native multi-frequency sensor fusion.


### Paper 3: The "Glass Box" Safety Protocol
**"Thermodynamic Diagnostic: Exact Relevance Propagation for Continuous Biological Trajectories"**
* **The Vibe:** AI Safety / Explainable Biology / DARPA-Grade AI.
* **The Problem:** Even if an AI perfectly predicts a catastrophic biological event (e.g., a Waddington crash), it is useless to clinicians if it acts as a black box.
* **The Solution:** You bring your `mamba_lrp.py` and `diagnostic_engine.py` modules into the spotlight. You introduce MambaLRPEpsilon, proving how to perfectly conserve attribution backward through the continuous $\exp(A \Delta t)$ matrix.
* **Why it Matters:** Explainability is the ultimate bottleneck for FDA-approved biological foundation models. When the model predicts a crash, this engine traces the exact causal chain back to the root event (e.g., "A TP53 RNA flash at T-30 mins caused the collapse").

## Integrate Isolated Modules into Main Training Loop

Several core modules were developed in isolation and currently contain technical debt regarding their integration into the main training harness (`src.harness.clinical_diagnostic_runner`). 

### 1. `SpatialCompressor` Integration
- **Issue**: `src/models/encoders/spatial_compressor.py` is currently isolated.
- **Task**: Integrate this encoder into the main pipeline to process volumetric (3D+time) optical data (e.g., from `AOLLSMDataset`) down to 1D latent vectors for the Mamba core.

### 2. `TopoEncoder` Integration
- **Issue**: `src/models/encoders/topo_encoder.py` is currently isolated.
- **Task**: Hook this up to process continuous electrophysiological standing waves (e.g., from `SpikeProphecyDataset` or `BioelectricLoader`) to provide spatial priors to the main Thermodynamic State Space Model.

### 3. `MeldLoss` Integration
- **Issue**: `src/models/losses/meld_loss.py` is currently isolated.
- **Task**: Apply this composite loss function in the main training loop to enforce thermodynamic constraints and time-reversal penalties on the model's predictions.

## Hook Up `MultimodalBioDataset` to `HierarchicalSSM`

- **Issue**: Now that we have a functional multimodal dataloader (`MultimodalBioDataset`) yielding phase, voltage, and RNA tensors, we need to test the `HierarchicalSSM` on it.
- **Task**: Connect the `MultimodalBioDataset` outputs to the `HierarchicalSSM`'s input block tensor, effectively implementing the multi-rate multi-sensor polling architecture described above.


## [COMPLETED] Add Neurospike data integration.

FinalSpark Neuroplatform: Standard Intan MEA, Low: 32 to 64 channels per well,
20 kHz to 30 kHz, Continuous 24/7 streaming for months. Use for Time: Perfect
for testing Mamba-2's ability to handle infinite sequence lengths and compute
thermodynamic drift over weeks, but it lacks the spatial density to show off
Mamba's capacity.

## V2 Architecture: The 129-D Quintet Tensor

We seek data engineers to build actual dataloaders for single-cell epigenetic
clocks (Gamma) and in-line electrochemical sensors (Mu). Mamba-2's
data-dependent step size ($\Delta t$) will gracefully absorb the massive `NaN`
gaps of hourly epigenetic reads.

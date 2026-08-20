# Stable Basin Benchmark Issues & Technical Debt

This document tracks known issues, technical debt, and open TODOs within the
codebase.

## Add Neurospike data integration.

FinalSpark Neuroplatform: Standard Intan MEA, Low: 32 to 64 channels per well,
20 kHz to 30 kHz, Continuous 24/7 streaming for months. Use for Time: Perfect
for testing Mamba-2's ability to handle infinite sequence lengths and compute
thermodynamic drift over weeks, but it lacks the spatial density to show off
Mamba's capacity.

## V2 Architecture: The 129-D Quintet Tensor

We need data engineers to build actual dataloaders for single-cell epigenetic
clocks (Gamma) and in-line electrochemical sensors (Mu). Mamba-2's
data-dependent step size ($\Delta t$) will gracefully absorb the massive `NaN`
gaps of hourly epigenetic reads.

## The ATP Metabolic Checkbook Loss Constraint

We need to update `src/models/meld_loss.py`. The physics constraint is: If the
model predicts high-frequency action potentials or massive RNA transcription, it
must mathematically subtract from the global ATP reserve dimension. If the model
hallucinates a high-energy repair cascade while the ATP checkbook is empty, the
loss function must geometrically explode.

## Overhaul `src/modules/hierarchical_ssm.py` for Prime Time

The `HierarchicalSSM` is currently a frozen mathematical toy simulator demonstrating standing wave phase transitions. It needs to be upgraded for actual training and multi-sensor fusion:

1.  **Make Parameters Learnable**: Remove `requires_grad=False` from the core matrices (`A1`, `B1`, `A2`, `B2`, `W_td`).
2.  **Discretize for Parallel Scans**: The current explicit Euler integration (`for t in range(steps):`) is unacceptably slow for GPUs. Implement a discretization step (e.g., Zero-Order Hold or Bilinear Transform) so the continuous ODE can be unrolled via an associative parallel scan (like the Mamba architecture).
3.  **Implement Multi-Rate Multi-Sensor Polling**: Upgrade the hardcoded single scalar input `u` to accept a multimodal block tensor. Modulate `dt` or assign different blocks of the `A` matrix to different polling rates (e.g., Layer 1 catching 20kHz electrical spikes, Layer 2 catching 1Hz RNA reads) to prove out native multi-frequency sensor fusion.

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

### Paper 3: The "Glass Box" Safety Protocol
**"Thermodynamic Diagnostic: Exact Relevance Propagation for Continuous Biological Trajectories"**
* **The Vibe:** AI Safety / Explainable Biology / DARPA-Grade AI.
* **The Problem:** Even if an AI perfectly predicts a catastrophic biological event (e.g., a Waddington crash), it is useless to clinicians if it acts as a black box.
* **The Solution:** You bring your `mamba_lrp.py` and `diagnostic_engine.py` modules into the spotlight. You introduce MambaLRPEpsilon, proving how to perfectly conserve attribution backward through the continuous $\exp(A \Delta t)$ matrix.
* **Why it Matters:** Explainability is the ultimate bottleneck for FDA-approved biological foundation models. When the model predicts a crash, this engine traces the exact causal chain back to the root event (e.g., "A TP53 RNA flash at T-30 mins caused the collapse").

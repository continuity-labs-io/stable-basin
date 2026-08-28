# Stable Basin Benchmark Issues & Technical Debt

This document tracks known issues, technical debt, and open TODOs within the
codebase.

## Publications

### H-SSM - The Biological Physics Engine
**"Hierarchical State-Space Models for Macroscopic Biological Entrainment"**
* **The Vibe:** Computational Neuroscience / Systems Biology.
* **The Problem:** Single-layer recurrent models act as passive filters; they process noise but cannot spontaneously generate higher-order biological structures (like traveling waves or synchronized rhythms).
* **The Solution:** Your `hierarchical_ssm.py` module. You prove that by coupling a fast/local layer with a delayed, top-down macro layer, you trigger a thermodynamic phase transition into a stable Attractor Limit Cycle. Research breadcrumb: Oscillatory dynamics as the
coordination layer of the
organism: waves, Markov blankets,
and the virtual space of cognition. (Daniel, 2026). Check out cross-frequency coupling as a possible method of hierarchical control in the brain.
* **Why it Matters:** This is the deep physics paper. It appeals to theoretical neuroscience, laying the mathematical groundwork for mapping a biological "thought" into a digital latent space.

## `icebox/src/modules/hierarchical_ssm.py`

The `HierarchicalSSM` is currently a frozen mathematical toy simulator demonstrating standing wave phase transitions. It needs to be upgraded for actual training and multi-sensor fusion:

1.  **Make Parameters Learnable**: Remove `requires_grad=False` from the core matrices (`A1`, `B1`, `A2`, `B2`, `W_td`).
2.  **Discretize for Parallel Scans**: The current explicit Euler integration (`for t in range(steps):`) is unacceptably slow for GPUs. Implement a discretization step (e.g., Zero-Order Hold or Bilinear Transform) so the continuous ODE can be unrolled via an associative parallel scan (like the Mamba architecture).
3.  **Implement Multi-Rate Multi-Sensor Polling**: Upgrade the hardcoded single scalar input `u` to accept a multimodal block tensor. Modulate `dt` or assign different blocks of the `A` matrix to different polling rates (e.g., Layer 1 catching 20kHz electrical spikes, Layer 2 catching 1Hz RNA reads) to prove out native multi-frequency sensor fusion.

## Hook Up `MultimodalBioDataset` to `HierarchicalSSM`

- **Issue**: Now that we have a functional multimodal dataloader (`MultimodalBioDataset`) yielding phase, voltage, and RNA tensors, we need to test the `HierarchicalSSM` on it.
- **Task**: Connect the `MultimodalBioDataset` outputs to the `HierarchicalSSM`'s input block tensor, effectively implementing the multi-rate multi-sensor polling architecture described above.

## Integrate Isolated Modules into Main Training Loop

Several core modules were developed in isolation and currently contain technical debt regarding their integration into the main training harness (`src.harness.clinical_diagnostic_runner`). 

### 2. `TopoEncoder` Integration
- **Issue**: `src/models/encoders/topo_encoder.py` is currently isolated.
- **Task**: Hook this up to process continuous electrophysiological standing waves (e.g., from `SpikeProphecyDataset` or `BioelectricLoader`) to provide spatial priors to the main Thermodynamic State Space Model.

### 3. `MeldLoss` Integration
- **Issue**: `src/models/losses/meld_loss.py` is currently isolated.
- **Task**: Apply this composite loss function in the main training loop to enforce thermodynamic constraints and time-reversal penalties on the model's predictions.


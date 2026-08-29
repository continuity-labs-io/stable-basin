# Stable Basin Issues

## Publications to Develop

### H-SSM - The Biological Physics Engine

**"Hierarchical State-Space Models for Macroscopic Biological Entrainment"**

- **The Vibe:** Computational Neuroscience / Systems Biology.
- **The Problem:** Single-layer recurrent models act as passive filters; they
  process noise but cannot spontaneously generate higher-order biological
  structures (like traveling waves or synchronized rhythms).
- **The Solution:** Your `hierarchical_ssm.py` module. You prove that by
  coupling a fast/local layer with a delayed, top-down macro layer, you trigger
  a thermodynamic phase transition into a stable Attractor Limit Cycle. Research
  breadcrumb: Oscillatory dynamics as the coordination layer of the organism:
  waves, Markov blankets, and the virtual space of cognition. (Daniel, 2026).
  Check out cross-frequency coupling as a possible method of hierarchical
  control in the brain.
- **Why it Matters:** This is the deep physics paper. It appeals to theoretical
  neuroscience, laying the mathematical groundwork for mapping a biological
  "thought" into a digital latent space.

## `icebox/src/modules/hierarchical_ssm.py`

The `HierarchicalSSM` is currently a frozen mathematical toy simulator
demonstrating standing wave phase transitions. It needs to be upgraded for
actual training and multi-sensor fusion:

1.  **Make Parameters Learnable**: Remove `requires_grad=False` from the core
    matrices (`A1`, `B1`, `A2`, `B2`, `W_td`).
2.  **Discretize for Parallel Scans**: The current explicit Euler integration
    (`for t in range(steps):`) is unacceptably slow for GPUs. Implement a
    discretization step (e.g., Zero-Order Hold or Bilinear Transform) so the
    continuous ODE can be unrolled via an associative parallel scan (like the
    Mamba architecture).
3.  **Implement Multi-Rate Multi-Sensor Polling**: Upgrade the hardcoded single
    scalar input `u` to accept a multimodal block tensor. Modulate `dt` or
    assign different blocks of the `A` matrix to different polling rates (e.g.,
    Layer 1 catching 20kHz electrical spikes, Layer 2 catching 1Hz RNA reads) to
    prove out native multi-frequency sensor fusion.

### Hook Up `MultimodalBioDataset` to `HierarchicalSSM`

- **Issue**: Now that we have a functional multimodal dataloader
  (`MultimodalBioDataset`) yielding phase, voltage, and RNA tensors, we need to
  test the `HierarchicalSSM` on it.
- **Task**: Connect the `MultimodalBioDataset` outputs to the
  `HierarchicalSSM`'s input block tensor, effectively implementing the
  multi-rate multi-sensor polling architecture described above.

## TODOs

### `MeldLoss` Ablation Tests

- **Issue**: The new MeldLoss functionality is untested and un-ablated on a
  large scale.
- **Task**: Run the loss ablation tests via `make loss-ablation` to get a sense
  of what loss function to use for biological homeostasis modeling.

### Waddington Collapse Benchmark Execution

- **Issue**: The `waddington_collapse.py` benchmark script currently evaluates an untrained `PredictiveCodingGraph` on the first 1,000 frames (`dataset[0]`) of the `PharmacologicalShockDataset`. Since the HD-MEA recording samples at a high frequency, 1,000 frames represents a fraction of a second before the drug takes effect. An untrained network also lacks a formed attractor basin, meaning its curvature metrics are effectively random.
- **Task**: We need to write a training loop to burn-in the `PredictiveCodingGraph` on healthy baseline sequences and then run the benchmark explicitly on the sequence where the pharmacological shock begins in the dataset.

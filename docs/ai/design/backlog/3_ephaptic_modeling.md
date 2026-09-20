# Design Document: Hierarchical Ephaptic State-Space Models (HESSM)
**Visual Metaphor:** The Lightning Rod and The Arc

## Executive Summary

Modern Deep Learning relies on a thermodynamically illiterate assumption: that
computation scales linearly. Standard architectures assume that stacking 100
Transformer or MLP layers will magically yield emergence, despite Layer 100
operating on the exact same physical rules (matrix multiplication of discrete
tokens) as Layer 1. 

**Biology changes the laws of physics as it
scales.**

*   **The Apple (Micro-Scale):** A single neuron or minicolumn minimizing its
    local prediction error is governed by slow, point-to-point synaptic
    chemistry.
*   **The Orange (Macro-Scale):** When 100,000 minicolumns fire in sync, their
    electrical exhausts fuse into a bulk extracellular electric field. This
    generates a thermodynamic phase transition ("More is Different"). The
    macroscopic entity detaches from the physical wires and becomes a
    continuous, high-speed bioelectric wave.
*   **The Apple-to-Orange Enslavement:** This macroscopic wave turns around and
    completely engulfs the microscopic components. It acts as a sovereign,
    higher-order physical entity that subtly alters the resting membrane
    potentials of the micro-components, universally orchestrating exactly *when*
    they are allowed to fire.

To build a physics engine that models true biological emergence, we must abandon
the "flat pipe." Conjure Cortez, burn the ships. This document outlines the architectural upgrade to the
**Stable Basin** continuous-time engine. By explicitly modeling **Theta-Gamma
Phase-Amplitude Coupling** and **Ephaptic Enslavement** using a Hierarchical
State-Space Model (H-SSM), we ensure the macroscopic layer operates on
fundamentally different physical principles than the microscopic layer, and
physically constrains its computation.

## The Rosetta Stone: Thermodynamics $\leftrightarrow$ Wetware Neuroscience

The behavior of a biological system navigating an attractor basin is defined by
the Fristonian Non-Equilibrium Steady State (NESS) equation: 

$$\dot{x} = (Q - \Gamma) \nabla E(x) + \omega$$

Biological concepts are rarely isolated to a single variable; rather, they
represent the dynamic interaction between these thermodynamic terms.

### 2.1 The Topographical Prior: $\nabla E(x)$ (The Landscape)
*   **Neuroscience:** The physical synaptic architecture of the brain, gene
    transcription, protein folding. Slow, molecular, highly specific.
*   **Physics Engine:** The deep Waddington basin carved out by evolution and
    memory. It dictates where energy *wants* to flow.

### 2.2 The Micro-Scale: Dissipative Friction / Gamma ($\Gamma$)
*   **Neuroscience:** High-frequency Gamma bursts (30-100 Hz). The "Lightning
    Rods" (neocortical minicolumns) firing localized, high-energy error
    corrections.
*   **Physics Engine:** Dissipative friction. When local networks receive
    sensory noise ($\omega$), they generate a prediction error. To minimize free
    energy, the system rapidly slides down the gradient ($\nabla E(x)$).
    $\Gamma$ is the fast, localized thermodynamic work dissipating the error
    into heat.

### 2.3 The Macro-Scale: Solenoidal Flow / Theta & Ephaptic Fields ($Q$)
*   **Neuroscience:** Low-frequency Theta waves (4-8 Hz) and continuous bulk
    extracellular electric fields (Ephaptic Coupling). The "Arc of Electricity"
    that sweeps horizontally across the cortex at near the speed of light.
*   **Physics Engine:** The divergence-free, solenoidal limit cycle. If the
    brain only had $\Gamma$, it would instantly slide to the bottom of the basin
    and freeze (die). $Q$ provides the macroscopic momentum that pushes the
    system orthogonally to the gradient, maintaining active, continuous-time
    biological orbits.

### 2.4 Dynamic Precision: $\Pi$ (Ionizing the Fluid)
*   **Neuroscience:** Global Neuromodulators (e.g., Dopamine, Acetylcholine,
    Noradrenaline) flooding the extracellular space, effectively "ionizing the
    fluid" to make the tissue highly conductive.
*   **Physics Engine:** The precision matrix ($\Pi$) output by our Energy-Based
    Model. It dynamically alters the shape of the Waddington basin. Dopamine
    steepens it ("Burn ATP to update now"); Noradrenaline flattens it
    ("Explore").

## 3. The Cybernetic Objective: Theta-Gamma Enslavement

**Theta-Gamma coupling is simply $Q$ orchestrating $\Gamma$.**

If local networks fired fast Gamma bursts ($\Gamma$) randomly and continuously,
the brain would exhaust its metabolic ATP and trigger a seizure. To prevent
this, the global macroscopic flow ($Q$) acts as a temporal throttle. The phase
of the slow $Q$ wave creates precise, 20-millisecond thermodynamic windows where
the local $\Gamma$ friction is authorized to act on the $\nabla E(x)$ gradient. 

The low-frequency wave *packages* the high-frequency bursts. The macroscopic
momentum ($Q$) organizes the microscopic work ($\Gamma$).

### Ephaptic Coupling: Synchronization Without Wires
When minicolumns (Lightning Rods) process input, they pump a vertical dipole of
voltage into the extracellular space. 
1.  **Ionizing the Air:** The tissue detects saliency and projects a Precision
    Matrix ($\Pi$), shifting local voltages to make the pathway conductive.
2.  **The Arc Jumps:** The combined electrical exhaust fuses into the Ephaptic
    Field, a slow, sweeping continuous wave ($Q$).
3.  **Top-Down Enslavement:** As this continuous "arc of electricity" washes
    over isolated, unwired lightning rods, it gently lifts their resting
    voltages right to the absolute edge of their firing threshold. Disparate
    columns are forced to fire their $\Gamma$ bursts in absolute, perfect
    lockstep because they are caught in the exact same electromagnetic undertow.

## 4. Architectural Implementation in `Stable Basin`

We will instantiate this phase transition in our existing
`PredictiveCodingGraph`.

### 4.1 The Micro-Layer (The Lightning Rods)
*   **Codebase Entity:** `MarkovBlanketObserver` (Micro)
*   **Role:** Simulates isolated, localized cortical minicolumns. Processes
    high-dimensional, fast sensory tokens ($\omega$).
*   **Physics:** Dominated by `DissipativeFriction` ($\Gamma$). Runs highly
    precise, localized updates. Noisy and fast.

### 4.2 The Macro-Layer (The Ephaptic Arc)
*   **Codebase Entity:** `MarkovBlanketObserver` (Macro)
*   **Role:** Simulates the continuous bioelectric standing wave.
*   **Physics:** Dominated by pure `SolenoidalFlow` ($Q$). It does not process
    discrete sensory tokens; it reads the spatial variance of the Micro-Layer
    and generates a continuous standing wave.

### 4.3 The Cybernetic Loop (Top-Down Enslavement)
The layers are lashed together through thermodynamic gating in
`HierarchicalThermoFlowFactor`, not feedforward wires.
1.  **Precision Projection (Implemented):** The Macro-Layer projects a Precision
    Matrix (`Pi_macro`) back down. This mathematically "ionizes the air,"
    dynamically steepening the local energy basin and scaling the
    micro-prediction errors.
2.  **Temporal Gating (Engine Upgrade Required):** We must update the `sample()`
    function in `HierarchicalThermoFlowFactor`. Currently, `dt` is a static
    constant. The integration limit ($\Delta t$) of the Micro-Layer's SDE solver
    must be dynamically gated by the phase of the Macro-Layer's state ($x_m$).
    The Macro field will act as a master clock, turning the Micro $\Gamma$
    updates on and off based on the crests and troughs of the $Q$ wave.

## 5. Clinical Interpretability (The Rosetta Translation)

With this architecture, clinical pathology is reduced to thermodynamic failure
states:

*   **Alzheimer's / Cognitive Decline:** Breakdown of Theta-Gamma coupling. The
    tissue loses its solenoidal momentum ($Q$). Without the macroscopic orbit to
    coordinate timing, local $\Gamma$ friction misfires, causing the structural
    prior ($\nabla E(x)$) to thermodynamically flatten.
*   **Seizures (Epilepsy):** Runaway friction ($\Gamma$ overpowering $Q$). The
    macroscopic throttle fails. Millions of cells enter a hyper-synchronized,
    uncontrolled $\Gamma$ burst, burning through the ATP supply and throwing the
    state vector into high-entropy chaos.
*   **Deep Sleep (SWA):** Pure Solenoidal Flow ($Q$). Sensory noise ($\omega$)
    disconnects. Friction ($\Gamma$) drops. The brain is dominated by massive,
    rolling delta waves ($Q$) to flush metabolic waste and perform maintenance
    on the topographical landscape ($\nabla E(x)$).
*   **The "Bystander Effect" (Contagious Collapse):** When a dying cell's
    resting membrane potential collapses, it instantly alters the local Ephaptic
    gradient ($Q$). This biophysical undertow electrically depolarizes
    neighbors, dragging their state-vectors across the bifurcation boundary
    faster than chemical cytokines (the SASP) can diffuse.

## 6. Next Steps & Benchmark Implementation

We will revive the dormant `ContinuousLFPDataset`
(`src/data/ephys/uhd_lfp_dataset.py`) to build the **Ephaptic EEG/LFP
Benchmark** and prove the superiority of H-ESSM over feedforward MLPs:

1.  **Engine Upgrade:** Implement `dt_micro = dt_base * macro_gate(x_macro)`
    inside `HierarchicalThermoFlowFactor.sample()`.
2.  **Metric Validation:** Use `SpectralMetrics.calculate_cfc_pac`
    (Phase-Amplitude Coupling via Mean Vector Length) to mathematically prove
    the Macro-layer autonomously learned to pace the amplitude variance of the
    Micro-layer without hardcoded wires.
3.  **Pathology Simulation:** Simulate Alzheimer's by synthetically ablating
    $Q_{macro}$. Prove the Micro-layer devolves into heat death (PAC drops to
    0.0).
4.  **Therapeutic Rescue:** Use a simulated hardware controller to inject
    exogenous $Q$ ($q_{ext}$) into the Macro-Layer and successfully restore the
    theta-gamma limit cycle.

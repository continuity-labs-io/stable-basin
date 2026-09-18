# Master Design Document: ECHO H-SSM (The Plasma Reactor)
**Project:** Emergent Continuous Hierarchical Observers (ECHO) – Phase II
**Scope:** Hierarchical Thermodynamics, Ephaptic Coupling, and Synthetic Phase
Transitions **Status:** Theoretical Architecture & Future Engineering Blueprint
(Slated for Paper #2)

## 1. The Manifesto: Burning Down the Feedforward Paradigm
The foundational error of Silicon Valley AI (Transformers, MLPs) and classical
biologically-inspired AI (HTM) is the assumption of **scale-invariant physics**.
They assume that Layer 100 operates on the exact same discrete mathematical
rules as Layer 1, merely passing tokens laterally and vertically through rigid
"wires."

Biology dictates that scaling intelligence requires a **Thermodynamic Phase
Transition** (Anderson's "More is Different"). Localized, structurally wired
computations (minicolumns) fuse to generate a global, continuous standing wave
(the extracellular bioelectric field). This higher-order field then detaches
from the wires, sweeps across the brain, and *enslaves* the microscopic
components.

To simulate true biological emergence, we must build a Hierarchical State-Space
Model (H-SSM) where the macroscopic layer operates on fundamentally different
laws of physics than the microscopic layer.

## 2. The Biophysical Rosetta Stone
We map the thermodynamic operators of Non-Equilibrium Steady State (NESS)
directly to the physics of the neocortex:

*   **The Micro-Scale ($\Gamma$ / Dissipative Friction):** 
    *   *Biology:* Localized, high-frequency Gamma bursts. Point-to-point
        chemical synapses minimizing local prediction error.
    *   *Physics:* Dissipative Friction ($\Gamma$). High-entropy,
        energy-consuming gradient descent. 
*   **The Macro-Scale ($Q$ / Solenoidal Flow):**
    *   *Biology:* Ephaptic Coupling. The bulk extracellular electromagnetic
        field (Theta waves).
    *   *Physics:* Solenoidal Flow ($Q$). A continuous, skew-symmetric rotation
        matrix that sweeps the global state sideways without consuming energy.
*   **The Enslavement Link ($\Pi$ & $\Delta t$):**
    *   *Biology:* Neuromodulators (Dopamine/Noradrenaline) ionizing the fluid,
        and Theta waves pushing resting potentials to threshold.
    *   *Physics:* The Macro-Layer projecting a Precision Matrix ($\Pi$) and an
        integration clock ($\Delta t$) downward to dynamically alter the
        physical Waddington basin of the Micro-Layer.

## 3. The Dual-Regime Architecture (The Van de Graaff Engine)
The ECHO H-SSM is defined by a coupled system of Stochastic Differential
Equations (SDEs), strictly partitioned by physical modality and timescale.

### 3.1 The Micro-Stratum: The Lightning Rods (Bottom-Up Variance)
*   **Structure:** An array of independent, localized observers (simulated
    minicolumns).
*   **Physics:** Governed strictly by the dissipative term. 
*   **Math:** $dx_{\mu} = -\tau(t) \cdot \Gamma_{\mu} \nabla E_{\mu}(x_{\mu};
    \Pi) dt + \sqrt{2\Gamma_{\mu} T} dW$
*   **Function:** Rapidly processes incoming sensory data. It aggressively
    minimizes local Free Energy by climbing down gradients. 
*   **Crucial Constraint:** It does *not* communicate laterally through explicit
    dense weight matrices. It generates localized "electrical exhaust"
    (variance).

### 3.2 The Macro-Stratum: The Arc of Plasma (Solenoidal Field)
*   **Structure:** A low-dimensional, continuous-time global state-space model.
*   **Physics:** Dominated by the solenoidal flow operator $Q$. 
*   **Math:** $dx_M = Q_M \nabla E_M(x_M) dt$ (Notice the lack of $\Gamma$ and
    noise).
*   **Function:** It does *not* process discrete sensory tokens. It reads the
    bulk thermodynamic variance (the free energy exhaust) of the Micro-Stratum
    to sustain its continuous orbit.

### 3.3 Top-Down Enslavement: The Grand Synchronization
The Macro-Layer does not send hidden states down to the Micro-Layer. It sends
**physics**.

1.  **Ionizing the Air (Precision Injection / $\Pi$):** The Macro-Layer projects
    a time-varying Precision Matrix $\Pi(t)$ down to the Micro-Layer. This
    dynamically steepens specific Waddington basins, dictating exactly how much
    energy a minicolumn must expend to move.
2.  **The Universal Clock (Ephaptic Pacing / $\Delta t$ Gating):** The
    Macro-Layer's sweeping $Q$-wave alters the integration limits (the SDE
    solver's step size, $\tau(t)$) of the Micro-Layer. As the wave crests over a
    region, $\tau(t)$ expands, forcing disparate, unwired Micro-nodes to execute
    their $\Gamma$ updates in absolute, perfect lockstep. When the wave troughs,
    $\tau(t) \to 0$, computationally freezing the nodes.

## 4. Neurological Pathologies as Thermodynamic Failures
This architecture allows us to mathematically trigger catastrophic brain states
by simply turning the thermodynamic knobs:

*   **Theta-Gamma Coupling (Healthy State):** The Macro $Q$ wave orbits
    smoothly, periodically pacing the Micro $\Gamma$ bursts to save ATP.
*   **Deep Sleep (SWA):** Micro-Layer sensory input is severed. $\Gamma$ shuts
    down. The system runs purely on massive Macro $Q$ flow to perform
    frictionless thermodynamic maintenance on the global landscape.
*   **Epileptic Seizures (Runaway $\Gamma$):** The Macro $Q$ wave collapses.
    Without the top-down $\tau(t)$ clock, the Micro-Layer enters an
    uncontrolled, hyper-synchronized $\Gamma$ runaway event, rapidly depleting
    computational ATP and throwing the system into chaos.
*   **Neuromodulation Dynamics:** Flooding the system with Dopamine dynamically
    scales the Precision Matrix ($\Pi$), instantly steepening Waddington basins
    to signal high-value prediction errors.

## 5. Engineering Constraints & Strategic Sequencing

**CRITICAL ARCHITECTURAL WARNING:** This design represents an immense leap in
software complexity over the "Flat" continuous-time SDE. Implementing dynamic
integration gating ($\tau(t)$) inside a dual-layered JAX `jax.lax.scan` loop
while preserving XLA compilation speeds will require a massive rewrite of the
`PredictiveCodingGraph`.

**Execution Strategy:**
1.  **Do NOT implement this in Phase 1.** Attempting to code this now will
    derail momentum and trigger context-collapse.
2.  **Paper 1 (Current Focus):** Keep the AI coding assembly line strictly
    focused on the **Stable Basin Beacon Paper (Worm Gait)**. A single-layer,
    flat EBM is mathematically sufficient to prove the basic thermodynamics of
    aging and *in-silico* reprogramming.
3.  **Paper 2 (The Horizon):** This document serves as the locked roadmap for
    Paper #2 (The Phase Transition). Once the Beacon Paper is out, we will use
    this TDD to build the true H-SSM Plasma Reactor and point it at complex
    human EEG data to prove that simulating a scale-dependent phase transition
    yields vastly superior dynamic modeling.

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

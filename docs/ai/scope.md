# 🧭 STABLE BASIN: Scope & Icebox Protocol

## 1. The Core Identity
Stable Basin is NOT a general-purpose machine learning framework, nor is it an
open-ended research sandbox. It is a highly opinionated **Computational Fluid
Dynamics (CFD) Engine for Longevity**. Its sole purpose is to model biological
systems as continuous-time observers governed by non-equilibrium thermodynamics,
calculate their Waddington basin degradation, and compute the energetic control
forces required for *in-silico* reprogramming.

## 2. The Current "Warm Box" (Maximum Allowed Complexity)
The repository is currently locked onto **Paper #1: The Thermodynamic Reversal
of Worm Gait**. To ensure scientific verifiability and avoid unmanageable
software complexity, the architecture is strictly capped at the following
boundaries:
*   **Data Domain:** Low-dimensional, continuous-time kinematic trajectories
    (e.g., 6D Eigenworms).
*   **Model Topology:** A single-layer PredictiveCodingGraph utilizing a
    `PrecisionWeightedEBM` (Flat SDEs).
*   **Success Metric:** Computing robust statistical metrics (Mean, Std,
    KS-Stat, Wasserstein) on the Hessian Trace to mathematically prove
    Waddington Basin Flattening and Therapeutic Rescue.

## 3. 🪓 The Icebox Protocol (The Axe)
Any concept, dataset, or architectural feature that exceeds the "Warm Box"
boundaries MUST be relegated to the `icebox/` directory immediately. 

**Currently IN THE ICEBOX (Do NOT touch, import, or integrate into the main
execution path):**
1. **High-Dimensional / Noisy Clinical Data:** Human Scalp EEG (MNE, LEMON,
   Sleep-EDF datasets), nonlinear EEG classifiers. *Reason: Biological noise and
   PCA dimensionality reduction mask the pure thermodynamic signals and make the
   SDE gradients unstable. Dilutes the core math proof.*
2. **Deep Structural Hierarchies (H-SSM):** Multi-node Markov partitions,
   complex `owmeta` RDF connectome parsing, or dynamic integration gating
   ($\tau(t)$). *Reason: Exceeds the current complexity budget. Reserved
   strictly for Paper #2 (The Plasma Reactor).*
3. **Hardware Deployment:** Edge-compute compilation, Bio-Blade I/O scripts.
   *Reason: Software physics engine must be mathematically verified before
   firmware compilation.*

**Agent Mandate:** When generating code or auditing the repo, if a feature
violates the Scope Protocol, you must refuse to integrate it into `src/`.
Propose moving the logic to `icebox/` instead.

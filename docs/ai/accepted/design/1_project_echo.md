### The Blueprint: Directory Structure
We will scaffold a dedicated `echo` subsystem to cleanly separate the continuous-time physics engine from the legacy PyTorch harnesses.

```text
src/
 ├── echo/
 │    ├── physics/               <-- PHASE I: The Bedrock
 │    │    ├── solenoidal.py     # Q matrices (skew-symmetric limit cycles)
 │    │    ├── dissipative.py    # Γ matrices (positive semi-definite friction)
 │    │    └── thermostat.py     # Fluctuation-Dissipation bounds
 │    ├── primitives/            <-- PHASE II: Software Scaffold
 │    │    ├── ebm.py            # Parametrized E_θ(x) & Precision networks
 │    │    └── thermalizer.py    # Torx Graph compiler & ThermoFlowFactor
 │    └── architecture/          <-- PHASE III: Hierarchical Observers
 │         ├── markov_hull.py    # Boundary enforcement p(μ | η, b)
 │         └── hierarchy.py      # Top-down / Bottom-up message passing
 ├── metrics/
 │    └── thermal_interpretability.py <-- PHASE V: Opening the Black Box
 └── data/
      └── toy/
           └── muller_brown_dataset.py <-- PHASE V: Toy Benchmark
```

---

### PHASE I: Theoretical Physics (The Bedrock)
We do not want the network to "learn" physics; we want to enforce it topologically at the tensor level using Equinox modules.

*   **The Engine ($Q$): `SolenoidalFlow`**
    *   *Infrastructure:* A parameterized class where the underlying weight matrix $W$ is strictly enforced as skew-symmetric ($Q = W - W^T$). This mathematically guarantees that $x^T Q x = 0$. The engine pushes the state sideways along the contour without doing thermodynamic work (orbiting the sweet spot).
*   **The Brakes ($\Gamma$): `DissipativeFriction`**
    *   *Infrastructure:* Parameterized via a Cholesky factor ($L L^T + \epsilon I$). This strictly enforces a positive-definite matrix, ensuring the "brakes" always consume energy and pull the system down the gradient toward homeostasis.
*   **The Thermostat: `FluctuationDissipation`**
    *   *Infrastructure:* The physical integrator. It binds the internal friction ($\Gamma$) to the environmental Wiener noise ($\omega$) based on ambient temperature ($D = \Gamma k_B T$). This guarantees the simulation never numerically freezes or overheats.

---

### PHASE II: Computational Primitives (The Software Scaffold)
We promote your Torx sandbox to a first-class continuous-time EBM compiler.

*   **`PrecisionWeightedEBM`**
    *   *Infrastructure:* An Equinox neural network that maps a state vector $x$ to a scalar energy value $E_\theta(x)$ **AND** outputs a dynamic Precision Tensor $\Pi_\theta(x)$ that defines its confidence in the steepness of its own energy landscape.
*   **`ThermoFlowFactor (torx.AbstractReferenceFactor)`**
    *   *Infrastructure:* The stochastic workhorse. In its `.sample()` method, it natively invokes `jax.grad` to compute $\nabla_x E_\theta(x)$ at runtime, evaluating the non-linear SDE: $\dot{x} = -(Q - \Gamma) \nabla_x E_\theta(x) + \omega$.
*   **`TorxThermalizer` (The Compiler)**
    *   *Infrastructure:* A stateful orchestrator. It takes a Torx Directed Factor Graph (DFG), dips the nodes in the Thermostat, wraps them in a `torx.ChainFactor`, and compiles the loop to XLA using `jax.lax.scan` for blistering fast continuous-time unrolling.

---

### PHASE III: Architectural Structure (The Network)
This is where we wire the physics into self-evidencing predictive coding entities.

*   **`MarkovHull`**
    *   *Infrastructure:* A rigid PyTree abstraction that partitions a flat biological tensor into Internal ($\mu$), Sensory ($s$), Active ($a$), and External ($\eta$). It physically enforces conditional independence by severing gradient flows between internal and external states.
*   **`HierarchicalObserver`**
    *   *Infrastructure:* Composes the Hull, $Q$, $\Gamma$, and the EBM.
*   **`PredictiveCodingGraph`**
    *   *Infrastructure:* Orchestrates the nested Russian dolls.
        *   **Bottom-Up:** Pipes unresolvable prediction errors (surprisal) upward.
        *   **Top-Down:** The macro-observer projects contextual beliefs and, crucially, its Precision Weighting ($\Pi$) downward. This top-down signal dynamically steepens the lower-level energy basin to thermodynamically "enslave" and entrain the fast micro-states.

---

### PHASE V: Empirical Validation (The Proving Ground)
Standard ML metrics (MSE, CrossEntropy) cannot measure biological survival. We need robust hooks to measure the *thermodynamic health* of the neural network.

#### 1. The Hook: `thermal_interpretability.py`
We must extract the topographical geometry of the Waddington landscape at runtime.
*   **`HessianCurvatureTracker`:** Uses `jax.hessian` on the learned $E_\theta(x)$ to compute the eigenvalues and Trace of the precision matrix. A dropping trace mathematically quantifies biological aging (the flattening of the tissue's macroscopic prior). We hook this into your `TelemetryLogger` to stream to Rerun.

#### 2. The Toy Benchmark: `muller_brown_dataset.py`
Before touching messy biological telemetry, we prove the physics engine works in a vacuum.
*   **The Setup:** The Müller-Brown potential is the chemical physics gold standard: a 2D surface with distinct attractor basins separated by saddle points.
*   **The Task:** We expose this as a continuous-time dataloader. The goal: Prove the EBM can autonomously discover the boundaries and use its Solenoidal Flow ($Q$) to orbit the sweet spot without sliding into the void, maintaining free-energy minimization.

#### 3. Real Data Benchmark Proposal: "The Waddington Collapse"
Predicting the *exact* crash frame (as done in your current `clinical_diagnostic_runner.py`) is a binary ML task. We are building a thermodynamic engine, so we need a thermodynamic benchmark.
*   **The Setup:** We repurpose your existing `PharmacologicalShockDataset` (20kHz HD-MEA arrays). We deploy a 2-level `HierarchicalObserver`. Level 1 tracks the ultra-fast HD-MEA voltage spikes. Level 2 tracks the slow, aggregate tissue state.
*   **The Event:** We feed it the data where the 50µM Diazepam toxic shock hits the tissue, causing a biological crash.
*   **The Metric (Energy Basin Escape Time - EBET):** The benchmark passes if—and only if—the `ThermalInterpretability` hook detects the Macro-Observer's Top-Down Precision Matrix ($\Pi$) flattening **before** (or exactly concurrently with) the actual catastrophic electrical spikes in the data.
*   **The Proof:** This empirically proves that biological aging/death is a top-down thermodynamic control failure (loss of precision weighting) that precedes the actual structural crash, and that ECHO can isolate it.

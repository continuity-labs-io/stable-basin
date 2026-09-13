**Context Files to Load / Create:**

- `src/data/behavior/celegans_gait_dataset.py` (Create)
- `src/echo/models/primitives/ebm.py` (Create)

**Task: Phase 3 Organism - Worm Gait Data Loaders and EBM Baselines** Please
build the data infrastructure and energy baseline models for the _C. elegans_
worm gait integration test. We must prove the training loop natively converges
on a clean, low-dimensional (6D) biological limit cycle without triggering
expensive JAX XLA recompilations.

**Core Objectives:**

**1. Static-Length PyTorch DataLoader (`celegans_gait_dataset.py`):**

- **The Issue:** The Open Worm Database provides biological posture trajectories
  of highly variable lengths. JAX `@jit` requires static shapes; passing
  variable-length sequences will cause devastating XLA recompilations on every
  single batch.
- **The Solution:** Create a standard PyTorch `Dataset` class
  (`CElegansGaitDataset`).
- In the `__getitem__` method, it must extract random, contiguous, fixed-length
  crops (strictly hardcode or default `seq_len=500`) from the underlying
  variable-length 6D eigenworm time series.
- **Synthetic Fallback:** For CI testing and deterministic development without
  requiring the full Open Worm DB download, implement a
  `generate_synthetic_limit_cycle(seq_len=500)` static method within the
  dataset. This should generate a seeded, deterministic 6D oscillation (e.g.,
  phase-shifted sine/cosine waves mimicking the biological limit cycle of
  forward locomotion).

**2. The Mathematical Baselines (`src/echo/models/primitives/ebm.py`):** We need
two distinct Energy-Based Models (EBMs) defined as pure Equinox Modules
(`eqx.Module`) to facilitate the ablation study comparing rigid mathematical
basins against biological neural network landscapes.

- **Model A: `GaussianEBM` (The Laplace Baseline):**
  - Implements a rigid, single-basin parabolic landscape:
    $E(x) = \frac{1}{2} (x - \mu)^T \Pi (x - \mu)$.
  - The trainable parameters are the mean vector $\mu$ and the precision matrix
    $\Pi$. Ensure $\Pi$ is strictly positive-definite by parameterizing it via a
    Cholesky decomposition ($L L^T$).
  - **The Proof Constraint:** Because the landscape is purely quadratic, the
    Hessian ($\nabla^2 E$) is _exactly_ the constant matrix $\Pi$. Its Hessian
    trace will mathematically be a flat line across all input states. This
    serves as a strict mathematical proof that our `HessianCurvatureTracker`
    (built in the runner) is operating correctly.

- **Model B: `PrecisionWeightedEBM` (Multimodal MLP):**
  - Implements a learned, free-form energy landscape capable of asymmetric,
    multimodal Waddington wells.
  - Use a small, deep Multi-Layer Perceptron (e.g., Input -> SiLU -> Linear ->
    SiLU -> 1 Output) to map the state vector $x$ to a scalar energy value
    $E_\theta(x)$.
  - This model's Hessian trace will dynamically change based on the state $x$,
    allowing us to test if it naturally captures biological decline (shallower
    basins in older worms).

**Constraints:**

- Strictly maintain the framework firewall: PyTorch handles the data
  (`celegans_gait_dataset.py`); Equinox/JAX handles the models (`ebm.py`). Do
  not cross-import these frameworks.
- Use the standard Python `logging` module. Keep all log messages peaceful,
  precise, and practical (e.g.,
  `logger.info("Generating seeded synthetic 6D limit cycle for CI testing.")`).
  Avoid dramatic, capitalized, or emoji-laden print statements.

**Context Files to Load / Create:**
* `src/echo/harness/echo_trainer.py`
* `src/echo/harness/pytorch_jax_bridge.py`
* `src/echo/harness/echo_runner.py` (Create)
* `configs/echo_training.yaml` (Create)

**Task: Phase 2 Engine - Execution Orchestration and Curvature Tracking**
Please implement `src/echo/harness/echo_runner.py` and its configuration file `configs/echo_training.yaml`. This layer binds the PyTorch DataLoaders (via the DLPack bridge) to the pure Equinox/Optax training loop, managing the training lifecycle and providing crucial diagnostic visibility into the thermodynamic physics engine.

**Core Objectives:**

**1. The `EchoRunner` Orchestrator:**
* Build the `EchoRunner` class to coordinate the training loop over multiple epochs.
* It must cleanly integrate:
    * The PyTorch DataLoader (fetching batched tensors).
    * The DLPack bridge (zero-copy transfer to JAX arrays using `pytorch_jax_bridge.py`).
    * The `EchoTrainer` (executing the `@eqx.filter_jit` update steps).
* Implement standard lifecycle hooks: setup, training epoch loop, and a validation hook.

**2. Weights & Biases (W&B) and Ray Tune Integration:**
* Integrate `wandb` for robust experiment tracking. Log the training MSE, validation MSE, and optimizer learning rates.
* Structure the runner to gracefully support hyperparameter sweeps. If a Ray Tune context is detected, it should report metrics via `ray.train.report`.

**3. The Hessian Curvature Tracker (Basin Steepening):**
* The core hypothesis of this physics engine is that biological health corresponds to a steep Waddington basin (high precision weighting), and aging flattens it. We need a diagnostic to measure this mathematically.
* Attach a `HessianCurvatureTracker` (either as a standalone utility or integrated into the validation hook).
* During the validation phase, evaluate the Hessian matrix ($H = \nabla^2 E_\theta(x)$) of the learned energy landscape with respect to the state vector.
* Compute the trace (sum of eigenvalues) of the Hessian. This value represents the total "steepness" or precision of the basin.
* Log this `hessian_trace` metric to W&B alongside the validation loss. A rising Hessian trace indicates the attractor basin is steepening as the limit cycle is learned.

**4. Configuration Management (`echo_training.yaml`):**
* Create `configs/echo_training.yaml` to hold all hyperparameters to keep the runner script clean and stateless.
* Include sections for:
    * `optimization`: learning rate, weight decay, gradient clip norm, max epochs.
    * `model`: hidden dimensions, `use_blanket_topology` flag (defaulting to False).
    * `logging`: W&B project name.

**Constraints:**
* **Memory Safety:** The Hessian computation can explode memory if not batched correctly or if evaluated on too large a sample. Use JAX's `jax.hessian` and `jax.vmap` strictly over a small, fixed-size subset (e.g., first 32 sequences) of the validation data.
* **Sharp Boundaries:** Ensure the boundary between PyTorch (DataLoader loop) and JAX (compute step) remains sharp. Do not leak JAX arrays back into PyTorch tensors unless explicitly necessary for logging scalar metrics (e.g., converting to standard python floats via `.item()`).
* Use the standard Python `logging` module. Keep all log messages peaceful, precise, and practical (e.g., `logger.info("Epoch 5 complete. Validation loss: 0.04. Hessian trace: 12.3.")`). Avoid dramatic, capitalized, or emoji-laden print statements.

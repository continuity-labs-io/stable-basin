**Context Files to Load / Create:**
* `src/echo/models/observer.py` (To reference the `PredictiveCodingGraph` and
  state partition definitions)
* `src/echo/harness/echo_trainer.py` (Create)

**Task: Phase 2 Engine - Functional Optax Training Loop & BPTT** 
Please implement the core engine training loop in `src/echo/harness/echo_trainer.py`.
This must be a purely functional JAX/Equinox pipeline that uses Backpropagation
Through Time (BPTT) to align a latent Waddington basin to observed biological
dynamics via Free Energy minimization.

**Core Objectives:**

**1. Partially Observable Teacher-Forced Loss (MSE):**
* In biological datasets (EEG, worm gait), we only observe sensory data; the
  internal and macro states remain latent.
* Implement a sequence prediction loss function utilizing Teacher-Forcing: 
  * Inject the true data $s_{true}(t)$ into the *sensory slice* of the model's
    state vector at each timestep.
  * Execute the physics step to predict the full next state $x(t+1)$.
  * Compute the Mean Squared Error (MSE) strictly between the *sensory slice* of
    the predicted $x(t+1)$ and the actual next data frame $s_{true}(t+1)$. Do
    not compute loss on the latent internal states.

**2. Pure Functional Equinox/Optax Pipeline:**
* Implement the core optimization logic using `optax.adamw`.
* **Gradient Stability:** Training continuous-time physics (SDEs) via BPTT is
  highly prone to gradient explosions. You must chain
  `optax.clip_by_global_norm` in the optimizer.
* Use `eqx.filter` and `eqx.partition` to cleanly separate trainable weights
  (the learned energy landscape $E_\theta$, the dissipative friction $\Gamma$,
  and the solenoidal flow $Q$) from static topologies (like the Markov partition
  boundaries) before computing gradients.
* Compute gradients using `eqx.filter_value_and_grad`.
* Wrap the forward/backward pass and optimizer update in a highly optimized
  `@eqx.filter_jit` step.

**3. Trainer Abstraction:**
* Create an `EchoTrainer` class (or purely functional equivalent Equinox
  structure) that manages the optimizer state, handles parameter partitioning,
  and provides a clean `step(model, batch)` interface.

**Constraints:**
* Adhere to strict functional programming paradigms required by JAX. Do not
  introduce side effects or mutate state internally within the JIT-compiled
  functions. 
* This file must be pure JAX/Equinox/Optax. Zero PyTorch data loaders or
  hardware movement logic allowed here.
* Use the standard Python `logging` module. Keep all log messages peaceful,
  precise, and practical (e.g., `logger.debug("Executing JIT-compiled BPTT
  update step.")`). Avoid dramatic, capitalized, or emoji-laden print
  statements.

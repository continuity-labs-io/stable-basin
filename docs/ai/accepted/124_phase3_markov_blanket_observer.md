ROLE: You are an elite Scientific Machine Learning Engineer specializing in JAX, Equinox, and Torx.

TASK: We are building `src/echo/architecture/observer.py`. Implement a robust, production-ready Equinox module called `MarkovBlanketObserver` that fuses physical boundaries, energy-based learning, and stochastic unrolling into a single, localized self-evidencing entity.

MATHEMATICAL CONSTRAINTS:
1. The Observer must instantiate and own the entire physical stack for a single entity: `MarkovHull`, `PrecisionWeightedEBM`, `SolenoidalFlow`, `DissipativeFriction`, `Thermostat`, and `TorxThermalizer`.
2. The core mechanism is applying the Hull's topological mask to the physics to enforce conditional independence. You must create a custom internal Torx factor called `MaskedThermoFlowFactor` (inheriting from `torx.factor.AbstractReferenceFactor`) inside this file to override the standard flow.
3. Specifically, inside this custom factor's `.sample()` method: 
   - Retrieve the mask: `M = self.hull.get_topology_mask()`
   - Masked Engine: `Q_masked = Q * M`
   - Masked Brakes: `Gamma_masked = Gamma * M` (element-wise multiplication).
   - Because masking Gamma breaks the original Cholesky factorization and can destroy positive-definiteness, you must compute the diffusion step safely. Use an eigenvalue decomposition to find the matrix square root `S` of the masked brakes:
     `evals, evecs = jnp.linalg.eigh(Gamma_masked + epsilon * jnp.eye(d_state))`
     `evals = jnp.maximum(evals, 0.0)`
     `S = evecs @ jnp.diag(jnp.sqrt(evals))`
   - The diffusion term becomes: `jnp.sqrt(2 * temperature * dt) * (S @ dW)`.
   - Execute the Euler-Maruyama step with these masked components.

TECHNICAL REQUIREMENTS:
- Do NOT use PyTorch. Inherit from `equinox.Module`.
- The `__init__` method must accept:
  - `d_internal`, `d_sensory`, `d_active`, `d_external` (ints).
  - `ebm_hidden_size`, `ebm_depth` (ints).
  - `n_steps` (int) - The number of simulation steps to unroll.
  - `temperature` (float).
  - `key` (jax.random.PRNGKey) - Used to initialize all sub-components via `jax.random.split`.
  - `epsilon` (float, default 1e-4).
- Inside `__init__`:
  - Instantiate `self.hull = MarkovHull(d_internal, d_sensory, d_active, d_external)`. Calculate `d_state`.
  - Instantiate `self.ebm = PrecisionWeightedEBM(...)`.
  - Instantiate `self.solenoidal = SolenoidalFlow(...)` and `self.dissipative = DissipativeFriction(...)`.
  - Instantiate `self.thermostat = Thermostat(temperature=temperature)`.
  - Instantiate your custom `MaskedThermoFlowFactor` (passing the instantiated modules to it).
  - Finally, instantiate `self.thermalizer = TorxThermalizer(flow_factor, n_steps, d_state)`.
- The `__call__(self, key: jax.random.PRNGKey, x_init: jax.Array, dt: float)` method must simply delegate to `self.thermalizer(key, x_init, dt)` and return the unrolled trajectory.
- Provide a helper method `extract_internal_state(self, x: jax.Array) -> dict` that delegates to `self.hull.partition(x)`.

TESTING:
Create `tests/echo/architecture/test_observer.py`. Write a rigorous `pytest` suite that:
1. Instantiates a `MarkovBlanketObserver` with `d_internal=2, d_sensory=1, d_active=1, d_external=4` (total `d_state=8`), `n_steps=5`.
2. Execution Test: Passes a random initial state `x_init` and `dt=0.01`. Asserts the resulting trajectory has the expected shape and contains no NaNs.
3. The Severance Test (The Acid Test): 
   - Define a dummy loss function that computes the sum of the *internal* state at the end of the 5-step simulation.
   - Use `equinox.filter_value_and_grad` to compute the gradient of this loss with respect to the initial state `x_init`.
   - **Crucial Assertion:** Assert that the gradient with respect to the `external` indices of `x_init` is exactly 0.0 (`jnp.allclose(grad_external, 0.0)`). This mathematically proves that no gradient/information leaked directly from the external world into the internal state, verifying the Markov Blanket is impenetrable!
4. Asserts the `__call__` method can be passed through `jax.jit`.

Write production-grade code with clean docstrings. Focus strictly on this module.

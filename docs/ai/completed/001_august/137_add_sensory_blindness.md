ROLE: You are an elite Scientific Machine Learning Engineer specializing in JAX, Equinox, and clean systems architecture.

TASK: Implement [UPDATE 001] (Sensory Degradation / Silent Drift) into the core ECHO architecture. We must decouple physical reachability from observability by introducing an optional Sensory Degradation Matrix (`D_s`). This must be an optional feature flag; if `D_s` is `None`, the system behaves identically to its baseline. 

This requires surgical updates to three files:
1. `src/echo/architecture/markov_hull.py`
2. `src/echo/architecture/observer.py`
3. `src/echo/architecture/hierarchy.py`

---

### PART 1: The Observation Operator (Markov Hull)
Modify `src/echo/architecture/markov_hull.py`.
- Add a new attribute to `MarkovHull`: `D_s: jax.Array | None = eqx.field(default=None)`.
- Update `__init__` to accept an optional argument `D_s: jax.Array | None = None` and assign it to `self.D_s`.
- Add a new method: `def apply_sensory_degradation(self, x: jax.Array) -> jax.Array:`
  - If `self.D_s` is `None`, return `x` unaltered (Zero-cost pass-through).
  - If `self.D_s` is provided, it will be a matrix of shape `(d_sensory, d_sensory)`. 
  - The method must:
    1. Extract the `sensory` slice from `x` (using `self.partition(x)`).
    2. Compute the degraded sensory state: `s_obs = self.D_s @ partitions["sensory"]`.
    3. Reconstruct and return the full state vector `x_obs`, substituting the original sensory slice with `s_obs`.

---

### PART 2: Decoupling the Observer (Single Level)
Modify `src/echo/architecture/observer.py`.
- Update `MarkovBlanketObserver.__init__` to accept `D_s: jax.Array | None = None`. Pass it into the `MarkovHull` instantiation.
- Locate the `MaskedThermoFlowFactor.sample()` method. We must force the EBM to evaluate the *degraded* state, not the true physical state.
- Update the `energy_fn(state)` closure to:
  ```python
  def energy_fn(state):
      # The EBM only "feels" the degraded observable state
      state_obs = self.hull.apply_sensory_degradation(state)
      e, _ = self.ebm(state_obs)
      return e
  ```

### PART 3: Decoupling the Hierarchy (Joint Free Energy)
Modify src/echo/architecture/hierarchy.py.

- Locate the joint_energy_fn(x_u, x_m) closure inside HierarchicalThermoFlowFactor.sample().

- The Joint Free Energy must be evaluated strictly on the observable states, not the true physical states.

- Modify the closure:

1. Apply degradation to both inputs:
```python
x_u_obs = self.micro_hull.apply_sensory_degradation(x_u)
x_m_obs = self.macro_hull.apply_sensory_degradation(x_m)
```

2. Compute EBMs using the degraded states:
```python
E_micro, _ = self.micro_ebm(x_u_obs)
E_macro, Pi_macro = self.macro_ebm(x_m_obs)
```

3. Compute the top-down belief and precision penalty using the degraded states:
```python
belief = self.W_down(x_m_obs)
diff = x_u_obs - belief
diff_proj = self.W_down.weight.T @ diff
penalty = 0.5 * diff_proj.T @ Pi_macro @ diff_proj
```

4. `return E_micro + E_macro + penalty`

### PART 4: Unit Testing (The Proof of Silent Drift)

- Create `tests/echo/architecture/test_silent_drift.py`. Write a rigorous pytest suite that mathematically proves the feature works:

1.Hull Test: Instantiate a MarkovHull. Pass a dummy x and set D_s to a matrix of all zeros. Assert that the returned x_obs from apply_sensory_degradation has exactly 0.0 for its sensory dimensions, while the internal, active, and external dimensions remain perfectly unaltered. Test that D_s=None returns an identical x.

2. Single Observer Gradient Test (The Acid Test):

- Instantiate a MarkovBlanketObserver passing D_s = jnp.zeros((d_sensory, d_sensory)).

- Extract the energy_fn logic (or write a wrapper that computes jax.grad of the observer's energy).

- Crucial Assertion: Assert that the gradient of the energy with respect to the sensory dimensions of x is exactly 0.0. This proves the EBM is completely blind to sensory perturbations.

3. Hierarchical Surprisal Test:

- Create a scenario testing joint_energy_fn with micro_hull.D_s = jnp.zeros(...).

- Use jax.grad on the joint_energy_fn with respect to x_u.

- Crucial Assertion: Assert that the resulting gradient (grad_micro) for the sensory dimensions is exactly 0.0. This mathematically proves "Silent Drift": the physical state can deviate, but because the sensors are degraded, the system generates ZERO prediction error (surprisal) to pass up the hierarchy.

Write production-grade code. Maintain all existing type hints and pure JAX/Equinox paradigms. Do not create any demo scripts; focus entirely on the core math and the unit tests.

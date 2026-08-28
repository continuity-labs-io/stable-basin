## ROLE
You are an elite Scientific Machine Learning Engineer specializing in JAX and differential geometry.

## TASK
We are building `src/echo/metrics/thermal_interpretability.py`. Implement a robust, production-ready class called `HessianCurvatureTracker` designed to extract the topographical geometry of the biological attractor basin (the Waddington landscape) at runtime.

## MATHEMATICAL CONSTRAINTS
1. The structural integrity of the tissue's macroscopic prior is defined by the curvature of the energy landscape E_θ(x).
2. The curvature is given by the Hessian matrix of E_θ with respect to the state x: H = ∇² E_θ(x).
3. The total "steepness" or "precision" of the basin is quantified by the Trace of the Hessian (the sum of its eigenvalues). A high trace indicates a healthy, youthful attractor. A low trace indicates thermodynamic flattening (biological aging).
4. Because our EBM outputs both Energy and Precision `(energy, precision) = ebm(x)`, the tracker must also compute the Trace of the network's explicit Precision matrix output (`Pi`) to compare the learned confidence against the actual physical curvature.

## TECHNICAL REQUIREMENTS
- Do NOT use PyTorch. The code must be pure JAX/Equinox.
- Implement the `HessianCurvatureTracker` class. It does not need to inherit from `equinox.Module` if it has no learnable parameters, but it must be JAX-compatible (e.g., standard Python class).
- The `__init__` method must accept:
  - `ebm`: An instance of `PrecisionWeightedEBM` (from `src.echo.primitives.ebm`).
- Implement a method `calculate_curvature(self, x: jax.Array) -> dict`:
  - `x` is the 1D state vector of shape `(d_state,)`.
  - Define a closure `energy_fn(state)` that evaluates `self.ebm(state)` and returns ONLY the scalar energy (ignoring the precision output).
  - Use `jax.hessian(energy_fn)` to compute the Hessian matrix `H` at state `x`. Shape must be `(d_state, d_state)`.
  - Compute the eigenvalues using `jnp.linalg.eigvalsh(H)` (since Hessians of smooth scalar fields are symmetric).
  - Compute `hessian_trace = jnp.sum(eigenvalues)`.
  - Evaluate the EBM normally `_, Pi = self.ebm(x)` and compute `explicit_precision_trace = jnp.trace(Pi)`.
  - Return a dictionary: `{"hessian_trace": hessian_trace, "explicit_precision_trace": explicit_precision_trace, "eigenvalues": eigenvalues}`.
- Provide a `vmap` wrapped version of this method, `batch_calculate_curvature(self, x_seq: jax.Array) -> dict`, that takes an unrolled trajectory of shape `(time_steps, d_state)` and returns a dictionary with 1D arrays of traces/metrics over time. Ensure the `ebm` is safely closed over or that `in_axes` is properly configured so weights aren't mapped.
- Use `jax.jit` internally on the computation logic to ensure high performance when analyzing trajectories.

## TESTING
Create `tests/echo/metrics/test_thermal_interpretability.py`. Write a rigorous `pytest` suite that:
1. Instantiates a `PrecisionWeightedEBM` (with smooth activations, e.g., GELU) and the `HessianCurvatureTracker`.
2. Passes a random single state vector `x` and asserts the dictionary returns scalars/arrays of the correct shapes with no NaNs.
3. Shape & VMAP Test: Passes a batch trajectory `x_seq` of shape `(100, d_state)`. Asserts the returned dictionary contains 1D arrays of shape `(100,)` for the scalar metrics.
4. The Convexity Test (The Acid Test): 
   - We must mathematically prove the Hessian is working.
   - Define a dummy Equinox module `PerfectBowlEBM` that mimics the EBM signature but hardcodes the energy function as a perfect quadratic bowl: `E(x) = 0.5 * k * jnp.sum(x**2)` (where `k` is a steepness constant, e.g., `k=3.0`), and returns an Identity matrix for Precision.
   - Pass this dummy EBM into the tracker.
   - Evaluate `calculate_curvature` at any point `x` (e.g. `d_state=4`).
   - **Crucial Assertion:** The Hessian of `0.5 * k * sum(x^2)` is exactly `k * I`. Assert that `hessian_trace` exactly equals `k * d_state` (`jnp.allclose`). This proves your JAX autodiff curvature math is fundamentally flawless.

Write production-grade code with clean docstrings. Focus strictly on this module.

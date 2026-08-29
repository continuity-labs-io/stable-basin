## ROLE

You are an elite Scientific Machine Learning Engineer specializing in JAX and
Equinox.

## TASK

We are building `src/echo/physics/thermostat.py`. Implement a robust,
production-ready Equinox module called `Thermostat` that enforces the
Fluctuation-Dissipation Theorem. This module integrates the deterministic
physics (Q and Γ) with stochastic environmental noise over a continuous time
step (dt) using the Euler-Maruyama method.

## MATHEMATICAL CONSTRAINTS

1. The Fluctuation-Dissipation Theorem states that environmental noise is
   proportional to internal friction and temperature. For this implementation,
   we define a scalar `temperature` (T). The noise covariance matrix over a time
   step dt is 2 * T * Γ * dt.
2. The deterministic drift of the system is given by: drift = -(Q - Γ) @ grad_E,
   where `grad_E` is the gradient of the energy landscape at state x.
3. The stochastic diffusion (noise) is given by: diffusion = sqrt(2 * T * dt) *
   (L @ dW).
   - `L` is the lower-triangular Cholesky factor of Γ (such that Γ = L @ L^T).
   - `dW` is a vector of standard normal Gaussian noise (Wiener process
     increment).
4. The final Euler-Maruyama state update is: x_next = x + (drift * dt) +
   diffusion.

## TECHNICAL REQUIREMENTS

- Do NOT use PyTorch. Inherit from `equinox.Module`.
- The `__init__` method should accept `temperature: float = 1.0` and store it as
  a static field using `equinox.field(static=True)`.
- The `__call__` method must accept the following arguments:
  - `x`: The current biological state vector of shape `(d_state,)`.
  - `grad_E`: The pre-computed gradient vector of the energy landscape at `x`,
    shape `(d_state,)`.
  - `Q`: The computed skew-symmetric matrix from the `SolenoidalFlow` module,
    shape `(d_state, d_state)`.
  - `L`: The lower-triangular matrix from the `DissipativeFriction` module
    (representing the Cholesky factor of Γ), shape `(d_state, d_state)`.
  - `dt`: The scalar continuous time step (e.g., 0.01).
  - `key`: A `jax.random.PRNGKey` to generate the Wiener process noise `dW`.
- Use `jax.random.normal(key, shape=x.shape)` to generate `dW`.
- Compute Γ dynamically inside the method as `Gamma = L @ L.T`.
- Ensure the method supports a single 1D vector `x`. Do not write batching logic
  inside the module; we will use `jax.vmap` externally.
- Add robust type hinting using `jaxtyping` and `jax.Array`. Default to
  `jnp.float32`.

## TESTING

Create `tests/echo/physics/test_thermostat.py`. Write a rigorous `pytest` suite
that:

1. Instantiates the `Thermostat` with a temperature of 1.0.
2. Creates dummy inputs for `x`, `grad_E`, `Q` (ensure it is skew-symmetric, Q =
   W - W.T), and `L` (ensure it is lower triangular, L = jnp.tril(W)), all of
   dimension `d_state=4`. Set `dt=0.1`.
3. Asserts that the output `x_next` has the exact same shape as the input `x`
   and contains no NaNs.
4. Zero-Temperature Test: Instantiate the `Thermostat` with `temperature = 0.0`.
   Run the update and assert that `x_next` exactly equals the deterministic
   Euler step `x + (-(Q - (L @ L.T)) @ grad_E) * dt`.
5. Asserts that the module can be successfully JIT-compiled (`jax.jit`).
6. Asserts that the module can be batched (`jax.vmap`) across a batch of `x`,
   `grad_E`, and `keys`, while keeping `Q`, `L`, and `dt` shared across the
   batch (using `in_axes`).

Write production-grade code with clean docstrings. Focus strictly on this
module.

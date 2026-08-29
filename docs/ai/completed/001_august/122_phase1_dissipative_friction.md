## ROLE
You are an elite Scientific Machine Learning Engineer specializing in JAX and
Equinox.

## TASK
We are building `src/echo/physics/dissipative.py`. Implement a robust,
production-ready Equinox module called `DissipativeFriction` that represents the
energy-consuming homeostatic correction (Γ) of a biological cell (the "brakes"
that pull the system down the gradient).

## MATHEMATICAL CONSTRAINTS
1. The matrix Γ (Gamma) must be strictly symmetric positive-definite to
   guarantee that friction always dissipates energy.
2. To ensure this, parameterize the matrix using a trainable lower-triangular
   matrix L of shape (d_state, d_state). 
3. Create a property or method `Gamma` that dynamically computes the
   positive-definite matrix via a Cholesky-style factorization: Γ = L @ L^T. 
4. To ensure strict numerical stability and prevent the matrix from becoming
   singular during training, add a small, fixed diagonal jitter (e.g., epsilon *
   I) to the computation: Γ = (L @ L^T) + (epsilon * jnp.eye(d_state)).
5. The `__call__(self, x)` method should compute the matrix-vector product Γx.
   Because Γ is positive-definite, the quadratic form x^T Γ x > 0 must hold true
   for all non-zero x.

## TECHNICAL REQUIREMENTS
- Do NOT use PyTorch. Inherit from `equinox.Module`.
- Use `jax.random.normal` to initialize an unconstrained matrix W. The
  `__init__` method must accept `d_state: int`, `key: jax.random.PRNGKey`, and
  an optional `epsilon: float = 1e-4`.
- Store the unconstrained matrix as a learnable parameter `self.W`. 

## - IMPORTANT
In the `Gamma` property/method, you MUST extract the lower-triangular part
dynamically using `L = jnp.tril(self.W)` before computing `L @ L^T`. If you only
apply `tril` during `__init__`, the optimizer will bleed non-zero gradients into
the upper triangle of the parameter during training, destroying the
parameterization.
- Scale the initialization of W by `1.0 / math.sqrt(d_state)` to ensure variance
  stability.
- Ensure the `__call__(self, x: jax.Array)` method supports a single 1D vector
  `x` of shape `(d_state,)`. Do not write batching logic inside the module; we
  will use `jax.vmap` externally.
- Add robust type hinting using `jaxtyping` (e.g., `Float[Array, "d_state
  d_state"]` and `Float[Array, "d_state"]`).
- Default to `jnp.float32` for all operations.

## TESTING
Create `tests/echo/physics/test_dissipative.py`. Write a rigorous `pytest` suite
that:
1. Initializes a 64-dimensional `DissipativeFriction` module with a fixed PRNG
   key.
2. Asserts that the computed matrix Γ is symmetric (`jnp.allclose(Gamma,
   Gamma.T, atol=1e-5)`).
3. Computes the eigenvalues of Γ (`jnp.linalg.eigh`) and asserts they are all
   strictly positive (greater than or equal to epsilon), mathematically proving
   positive-definiteness.
4. Passes a random non-zero state vector x through it and asserts that the
   quadratic form `jnp.dot(x, Gamma_x)` is strictly greater than 0.0.
5. Asserts that the module can be successfully JIT-compiled (`jax.jit`) and
   batched over a dimension (`jax.vmap`).

Write production-grade code with clean docstrings. Focus strictly on this
module.

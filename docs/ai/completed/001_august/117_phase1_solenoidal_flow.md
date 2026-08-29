ROLE: You are an elite Scientific Machine Learning Engineer specializing in JAX
and Equinox.

TASK: We are building `src/echo/physics/solenoidal.py`. Implement a robust,
production-ready Equinox module called `SolenoidalFlow` that represents the
energy-conserving rotational dynamics (Q) of a biological attractor basin.

MATHEMATICAL CONSTRAINTS:

1. The matrix Q must be strictly skew-symmetric (Q = -Q^T).
2. To ensure this, initialize an unconstrained trainable weight matrix W of
   shape (d_state, d_state).
3. Create a property or method `Q` that dynamically computes the skew-symmetric
   matrix: Q = W - W^T.
4. The `__call__(self, x)` method should compute the matrix-vector product Qx.
   Because Q is skew-symmetric, the mathematical invariant x^T Q x = 0 must hold
   true (meaning the flow does no thermodynamic work).

TECHNICAL REQUIREMENTS:

- Do NOT use PyTorch. Inherit from `equinox.Module`.
- Use `jax.random.normal` for initialization. The `__init__` method must accept
  `d_state: int` and `key: jax.random.PRNGKey`.
- Scale the initialization of W by `1.0 / math.sqrt(d_state)` to ensure variance
  stability.
- Ensure the `__call__(self, x: jax.Array)` method supports a single 1D vector
  `x` of shape `(d_state,)`. Do not write batching logic inside the module; we
  will use `jax.vmap` externally.
- Add robust type hinting using `jaxtyping` (e.g., `Float[Array, "d_state"]`).
- Default to `jnp.float32` for all operations.

TESTING: Create `tests/echo/physics/test_solenoidal.py`. Write a rigorous
`pytest` suite that:

1. Initializes a 64-dimensional `SolenoidalFlow` module.
2. Asserts that the computed matrix Q is perfectly skew-symmetric
   (`jnp.allclose(Q, -Q.T, atol=1e-6)`).
3. Passes a random state vector x through it and asserts that the dot product
   `jnp.dot(x, Q_x)` is mathematically 0.0 (within float32 `atol=1e-5`).
4. Asserts that the module can be successfully JIT-compiled (`jax.jit`) and
   batched over a dimension (`jax.vmap`).

Write production-grade code with clean docstrings. Do not hallucinate other
parts of the physics engine, focus strictly on this module.

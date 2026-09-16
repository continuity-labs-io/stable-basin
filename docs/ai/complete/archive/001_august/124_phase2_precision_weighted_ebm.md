## ROLE

You are an elite Scientific Machine Learning Engineer specializing in JAX and
Equinox.

## TASK

We are building `src/echo/primitives/ebm.py`. Implement a robust,
production-ready Equinox module called `PrecisionWeightedEBM` that acts as the
core Energy-Based Model for our biological observer.

## MATHEMATICAL CONSTRAINTS

1. The network maps a biological state vector `x` to two outputs: a scalar
   Energy E_θ(x), and a Precision matrix Π_θ(x).
2. Energy E_θ(x) represents the thermodynamic potential (altitude) and must be a
   smooth scalar value (shape `()`).
3. Precision Π_θ(x) must be a strictly symmetric positive-definite (SPD) matrix
   of shape `(d_state, d_state)`. It represents the certainty/steepness of the
   local energy landscape.
4. To guarantee the Precision matrix is SPD, the network's precision head should
   output a flat vector of size `d_state * d_state`, reshape it to `(d_state,
   d_state)`, extract the lower triangle L = jnp.tril(W_raw), and compute Π_θ =
   (L @ L.T) + (epsilon * jnp.eye(d_state)).
5. The network must be twice-differentiable everywhere. You MUST use smooth
   activation functions (e.g., `jax.nn.gelu` or `jax.nn.softplus`).
   `jax.nn.relu` is strictly forbidden as its second derivative is
   zero/undefined, which will crash our Hessian-based topography metrics later
   in the pipeline.

## TECHNICAL REQUIREMENTS

- Do NOT use PyTorch. Inherit from `equinox.Module`.
- Architecture:
  - A shared backbone MLP (using `equinox.nn.MLP` or custom linear layers).
    Note: If using `equinox.nn.MLP`, you must pass the correct activation
    function to its constructor.
  - Two distinct output heads (Linear layers):
    - `energy_head`: Outputs 1 feature (squeezed to a scalar).
    - `precision_head`: Outputs `d_state * d_state` features.
- The `__init__` method must accept `d_state: int`, `hidden_size: int`, `depth:
  int`, `key: jax.random.PRNGKey`, and `epsilon: float = 1e-4`. Route the PRNG
  keys correctly (using `jax.random.split`) to initialize the trunk and the two
  linear heads.
- Store `epsilon` using `equinox.field(static=True)`.
- The `__call__(self, x: jax.Array)` method must take a 1D vector `x` of shape
  `(d_state,)` and return a tuple: `(energy, precision)`.
  - `energy` must be a scalar `jax.Array` of shape `()`.
  - `precision` must be a `jax.Array` of shape `(d_state, d_state)`.
- Do not write batching logic inside the module; we will use `jax.vmap`
  externally.
- Add robust type hinting using `jaxtyping` (e.g., `Float[Array, "d_state"]`).
  Default to `jnp.float32`.

## TESTING

Create `tests/echo/primitives/test_ebm.py`. Write a rigorous `pytest` suite
that:

1. Instantiates a `PrecisionWeightedEBM` with `d_state=4`, `hidden_size=16`,
   `depth=2`.
2. Shape & Constraint Test: Passes a random vector `x`. Asserts `energy` is a
   scalar (`energy.shape == ()`), and `precision` is `(4, 4)`. Asserts
   `precision` is perfectly symmetric and its eigenvalues are strictly positive
   (>= epsilon).
3. First-Derivative Test (The Force): Uses `jax.grad` to compute the derivative
   of the _energy_ output with respect to the input `x` (`jax.grad(lambda x:
   model(x)[0])`). Asserts the resulting gradient has shape `(4,)` and contains
   no NaNs.
4. Second-Derivative Test (The Curvature): Uses `jax.hessian` to compute the
   second derivative of the _energy_ output with respect to `x`. Asserts the
   Hessian has shape `(4, 4)` and contains no NaNs.
5. Asserts that the module can be successfully batched (`jax.vmap`) and
   JIT-compiled (`jax.jit`).

Write production-grade code with clean docstrings. Focus strictly on this
module, do not hallucinate outside components.

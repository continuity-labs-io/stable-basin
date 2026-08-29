## ROLE

You are an elite Scientific Machine Learning Engineer specializing in JAX and
Equinox.

## TASK

We are building `src/echo/architecture/markov_hull.py`. Implement a robust,
production-ready Equinox module called `MarkovHull` that mathematically enforces
a Markov Blanket partition on a biological state tensor.

## MATHEMATICAL CONSTRAINTS

1. The Hull receives a flat, 1D JAX array `x` representing the total universe.
2. It partitions `x` into four distinct sub-vectors sequentially:
   - `internal` (μ): The core biological state.
   - `sensory` (s): The incoming environmental signals.
   - `active` (a): The outgoing biological actions.
   - `external` (η): The environment.
3. The fundamental law of the Hull: The `internal` state MUST NOT interact
   directly with the `external` state.
4. To enforce this, the Hull must generate a `(d_state, d_state)` topological
   adjacency mask. This binary matrix should be 1.0 everywhere, EXCEPT for the
   block connecting `internal` to `external`, and the block connecting
   `external` to `internal`, which must be exactly 0.0.

## TECHNICAL REQUIREMENTS

- Do NOT use PyTorch. Inherit from `equinox.Module`.
- The `__init__` method must accept four integers: `d_internal`, `d_sensory`,
  `d_active`, and `d_external`. Store these using `equinox.field(static=True)`.
- Compute and store `d_state` as the sum of all components as a static field.
- Implement a method `partition(self, x: jax.Array) -> dict` that takes a flat
  1D array of size `d_state` and uses standard JAX slicing to return a
  dictionary:
  `{"internal": array, "sensory": array, "active": array, "external": array}`.
- Implement a method `reconstruct(self, partitions: dict) -> jax.Array` that
  concatenates the dictionary values back into a flat 1D array in the exact
  order: `[internal, sensory, active, external]`.
- Implement a method `get_topology_mask(self) -> jax.Array`. This method
  constructs and returns a `(d_state, d_state)` binary mask (using
  `jnp.float32`).
  - The block corresponding to row indices `internal` and column indices
    `external` MUST be 0.0.
  - The block corresponding to row indices `external` and column indices
    `internal` MUST be 0.0.
  - All other elements (including the blanket interactions and
    self-interactions) MUST be 1.0.
- Add robust type hinting using `jaxtyping` and `jax.Array`.

## TESTING

Create `tests/echo/architecture/test_markov_hull.py`. Write a rigorous `pytest`
suite that:

1. Instantiates a `MarkovHull` with `d_internal=2`, `d_sensory=2`, `d_active=2`,
   `d_external=4` (total 10 dimensions).
2. Reversibility Test: Generates a random 10D vector `x`. Asserts that
   `reconstruct(partition(x))` perfectly matches the original array `x`
   (`jnp.allclose`).
3. Partition Shape Test: Asserts that `partition(x)` correctly splits the array
   into shapes `(2,)`, `(2,)`, `(2,)`, `(4,)`.
4. Topological Mask Test (The Causal Sever):
   - Calls `get_topology_mask()` and asserts the shape is `(10, 10)`.
   - Asserts that the top-right block (internal rows `0:2`, external cols
     `6:10`) is exactly all zeros.
   - Asserts that the bottom-left block (external rows `6:10`, internal cols
     `0:2`) is exactly all zeros.
   - Asserts that all other blocks (e.g., sensory/active `2:6, 2:6`, and their
     connections to internal/external) are all ones.
5. Asserts that the module and its methods can be passed through `jax.jit`
   seamlessly.

Write production-grade code with clean docstrings. Focus strictly on this
module, do not hallucinate outside components.

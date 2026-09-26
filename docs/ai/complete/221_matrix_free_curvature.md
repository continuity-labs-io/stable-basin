Next, let's implement Fix 2: Matrix-Free Curvature Estimation using Hutchinson's trace estimator to avoid O(d^3) operations and O(d^2) memory footprints when evaluating the Hessian trace.

1. Open `src/echo/metrics/energy_landscape.py`.
2. Add a new function: `@eqx.filter_jit def hutchinson_hessian_trace(energy_fn: Callable, x: jax.Array, key: jax.Array, n_probes: int = 15) -> jax.Array`.
3. Inside `hutchinson_hessian_trace`:
   - Define `grad_fn = jax.grad(energy_fn)`.
   - Generate Rademacher vectors: `keys = jax.random.split(key, n_probes)`. Generate `zs` of shape `[n_probes, x.shape[0]]` using `jax.vmap(lambda k: jax.random.rademacher(k, x.shape, dtype=x.dtype))(keys)`. (If `jax.random.rademacher` is unavailable, use `jax.random.choice(k, jnp.array([-1.0, 1.0], dtype=x.dtype), shape=x.shape)`).
   - Define an inner function `def hvp(z_i): _, hvp_out = jax.jvp(grad_fn, (x,), (z_i,)); return jnp.dot(z_i, hvp_out)`.
   - Use `trace_estimates = jax.vmap(hvp)(zs)` and return `jnp.mean(trace_estimates)`.
4. Add the vmapped batch version: `@eqx.filter_jit def batch_hutchinson_trace(energy_fn: Callable, x_seq: jax.Array, key: jax.Array, n_probes: int = 15) -> jax.Array`.
   - Split `key` into `x_seq.shape[0]` subkeys.
   - Return `jax.vmap(lambda x, k: hutchinson_hessian_trace(energy_fn, x, k, n_probes))(x_seq, keys)`.
5. Modify the driver function `curvature_over_states(...)`:
   - Add new kwargs to the signature (after `rank_tol`): `estimator: Literal["exact_hessian", "hutchinson"] = "exact_hessian"`, `hutchinson_key: jax.Array | None = None`, and `hutchinson_probes: int = 15`.
   - Inside the chunk loop where `if full_spectrum` is evaluated, update the `else` branch:
     - If `estimator == "exact_hessian"`, use the existing `batch_hessian_trace(energy_fn, chunk)`.
     - If `estimator == "hutchinson"`, you need a PRNG key. If `hutchinson_key` is None, raise a ValueError. Otherwise, split `hutchinson_key` to get a `chunk_key` and a new `hutchinson_key`. Call `batch_hutchinson_trace(energy_fn, chunk, chunk_key, hutchinson_probes)`.
     - Raise a ValueError if the `estimator` string is not recognized.

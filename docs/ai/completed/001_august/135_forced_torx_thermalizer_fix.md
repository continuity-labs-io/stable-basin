ROLE: You are an elite Scientific Machine Learning Engineer specializing in JAX, Equinox, and clean systems architecture.

TASK: Resolve an abstraction leak in our repository. Currently, `src/echo/benchmarks/waddington_collapse.py` manually rips a Torx factor out of a compiled DFG and runs a custom `jax.lax.scan` loop to inject exogenous sensory data over time (`unroll_with_data`). We are adopting "Option C" from our analysis to push this logic down into the primitives layer, maintaining architectural hygiene.

You will modify four files:
1. `src/echo/primitives/thermalizer.py`
2. `src/echo/architecture/observer.py`
3. `src/echo/architecture/hierarchy.py`
4. `src/echo/benchmarks/waddington_collapse.py`

---

### PART 1: `src/echo/primitives/thermalizer.py`
Create a new Equinox module `ForcedTorxThermalizer` in this file. While `TorxThermalizer` is for autonomous (closed) systems, this class is for forced (open) systems driven by an external time-series sequence.

TECHNICAL REQUIREMENTS:
- Inherit from `equinox.Module`.
- `__init__(self, flow_factor: torx.factor.AbstractReferenceFactor, d_state: int, injection_start_idx: int)`:
  - Store `flow_factor`.
  - Store `d_state` and `injection_start_idx` as static fields (`equinox.field(static=True)`). The `injection_start_idx` tells the thermalizer *where* in the 1D state vector to inject the incoming exogenous data.
- `@equinox.filter_jit`
  `__call__(self, key: jax.random.PRNGKey, x_init: jax.Array, dt: float, seq: jax.Array) -> jax.Array`:
  - `seq` is a 2D array of shape `(n_steps, d_sensor)`.
  - Implement a `step_fn(state, carry)` that unrolls over the sequence using `jax.lax.scan`.
  - `carry` should be a tuple `(data_frame, step_key)`.
  - Inside `step_fn`: 
    1. Inject the `data_frame` into the current `state` using `jax.lax.dynamic_update_slice(state, data_frame, (self.injection_start_idx,))`.
    2. Construct `inputs = {"x": state, "dt": jnp.array(dt, dtype=jnp.float32)}`.
    3. Call `self.flow_factor.sample(step_key, inputs=inputs, params={})` to get `next_state`.
    4. Return `(next_state, next_state)`.
  - Split the `key` to match `seq.shape[0]`.
  - Execute `jax.lax.scan` over `step_fn`, passing `x_init` as the initial state, and return the resulting trajectory.

---

### PART 2: `src/echo/architecture/observer.py`
Update `MarkovBlanketObserver` to expose this new forced unrolling capability.

TECHNICAL REQUIREMENTS:
- In `__init__`, alongside `self.thermalizer`, instantiate:
  `self.forced_thermalizer = ForcedTorxThermalizer(flow_factor=masked_factor, d_state=d_state, injection_start_idx=self.hull.d_internal)`
- Add a new method:
  `def forced_unroll(self, key: jax.random.PRNGKey, x_init: jax.Array, dt: float, seq: jax.Array) -> jax.Array:`
  - Call and return `self.forced_thermalizer(key, x_init, dt, seq)`.

---

### PART 3: `src/echo/architecture/hierarchy.py`
Update `PredictiveCodingGraph` to expose the exact same forced unrolling capability.

TECHNICAL REQUIREMENTS:
- In `__init__`, alongside `self.thermalizer`, instantiate:
  `self.forced_thermalizer = ForcedTorxThermalizer(flow_factor=factor, d_state=d_state, injection_start_idx=micro_observer.hull.d_internal)`
  *(Note: In the joint state vector, the micro state comes first, and its sensory slice begins exactly after the internal slice, so `d_internal` is the mathematically correct injection index).*
- Add a new method:
  `def forced_unroll(self, key: jax.random.PRNGKey, x_micro_init: jax.Array, x_macro_init: jax.Array, dt: float, seq: jax.Array) -> jax.Array:`
  - Concatenate the initial states: `x_init = jnp.concatenate([x_micro_init, x_macro_init])`.
  - Call and return `self.forced_thermalizer(key, x_init, dt, seq)`.

---

### PART 4: `src/echo/benchmarks/waddington_collapse.py`
Clean up the benchmark script to remove the abstraction leak.

TECHNICAL REQUIREMENTS:
- Delete the entire nested `@eqx.filter_jit def unroll_with_data(...)` function and its internal `step_fn`.
- Replace the invocation with a clean call to the new graph method:
  `trajectory = graph.forced_unroll(k4, x_micro_init, x_macro_init, dt, data_seq)`
- Ensure the rest of the benchmark logic (EBET calculation, plotting) remains completely intact.

---

TESTING & HYGIENE:
- Ensure you import `ForcedTorxThermalizer` appropriately in `observer.py` and `hierarchy.py`.
- Check that all type hints are strictly maintained.
- Ensure no PyTorch code leaks into `thermalizer.py`, `observer.py`, or `hierarchy.py`.
- After making the changes, verify that the separation of concerns is restored: The benchmark script should *only* pass data to the architecture, and the architecture should *only* delegate to the primitives.

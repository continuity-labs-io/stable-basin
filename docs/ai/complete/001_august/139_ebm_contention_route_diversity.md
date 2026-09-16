ROLE: You are an elite Scientific Machine Learning Engineer specializing in JAX, Equinox, and Torx.

TASK: Implement the core infrastructure for [UPDATE 002] (Map EBM Contention & Route Diversity). We must modify the physics engine to accept simultaneous external thermodynamic forces (`omega_ext`) so we can model concurrent demands overwhelming the biological system.

This requires updates to 4 files:
1. `src/echo/physics/thermostat.py`
2. `src/echo/primitives/thermalizer.py`
3. `src/echo/architecture/observer.py`
4. `src/echo/architecture/hierarchy.py`

---

### PART 1: The Thermostat
Modify `src/echo/physics/thermostat.py`.
- Update the `__call__` method to accept an optional argument `omega_ext: jax.Array | None = None`.
- If `omega_ext` is provided, add it to the deterministic drift before integrating: `drift_total = drift + omega_ext`.
- Update the Euler-Maruyama step to use the total drift: `x_next = x + (drift_total * dt_jnp) + diffusion`.

### PART 2: Thermalizer & Torx Factors
Modify `src/echo/primitives/thermalizer.py`.
- In `ThermoFlowFactor.__init__`, add `"omega_ext": jax.ShapeDtypeStruct((d_state,), jnp.float32)` to `self.input_ports`.
- In `ThermoFlowFactor.sample`, extract `omega_ext = inputs.get("omega_ext", jnp.zeros(self.d_state))` and pass it to the thermostat.
- In `TorxThermalizer.__init__`, update the DFG `input_ports` to include `"omega_ext_constant": jax.ShapeDtypeStruct((d_state,), jnp.float32)`. Update the `parents` and `porting_fn` of the `torx.Site` to map `"omega_ext_constant"` to `"omega_ext"`.
- In `TorxThermalizer.__call__`, update the `inputs` dict to provide `"omega_ext_constant": jnp.zeros_like(x_init)`.
- In `ForcedTorxThermalizer.__call__`:
  - Change the signature to make `seq` optional: `seq: jax.Array | None = None`, and add `omega_seq: jax.Array | None = None`.
  - Because JAX tracing handles Python control flow (`if seq is not None`) at compile time, write two separate `jax.lax.scan` paths. 
  - If `seq` is provided, `dynamic_update_slice` the state as before. 
  - If `omega_seq` is provided, pass `omega_frame` into the `inputs` dict for the factor. If not, pass zeros.
  - Ensure the method supports providing ONLY `omega_seq` (where `seq` is `None`), allowing the system to evolve freely under a force without its coordinates being hard-overwritten.

### PART 3: Hierarchical Factors
Modify `src/echo/architecture/hierarchy.py`.
- In `HierarchicalThermoFlowFactor.__init__`, add `"omega_ext": jax.ShapeDtypeStruct((self.d_micro + self.d_macro,), jnp.float32)` to `self.input_ports`.
- In `.sample()`, extract `omega_ext = inputs.get("omega_ext", jnp.zeros(self.d_micro + self.d_macro))`. Split it into `omega_micro = omega_ext[:self.d_micro]` and `omega_macro = omega_ext[self.d_micro:]`.
- Pass `omega_micro` to the `micro_thermostat` and `omega_macro` to the `macro_thermostat`.

### PART 4: Observer Exposure
Modify `src/echo/architecture/observer.py` and `src/echo/architecture/hierarchy.py`.
- Ensure `MarkovBlanketObserver.forced_unroll` and `PredictiveCodingGraph.forced_unroll` signatures are updated to accept `seq: jax.Array | None = None` and `omega_seq: jax.Array | None = None`.
- Pass both down to `self.forced_thermalizer(key, x_init, dt, seq=seq, omega_seq=omega_seq)`.

TESTING:
Create `tests/echo/physics/test_contention_forces.py`. 
- Instantiate a `Thermostat`. Call it with `omega_ext` set to a vector of `10.0`s. Assert that the output state is shifted by approximately `10.0 * dt` compared to calling it with `omega_ext=None` (proving the force was applied).
- Assert that `ForcedTorxThermalizer` can run `forced_unroll` with `seq=None` and a non-zero `omega_seq` without crashing.

Write production-grade JAX. Maintain all type hints and pure functional paradigms.

ROLE: You are an elite Scientific Machine Learning Engineer specializing in JAX, Equinox, and clean systems architecture.

TASK: Implement the core infrastructure for [UPDATE 003] (Diagnostic Logic for Bio-Blade Actuation). We must modify the physics engine to expose an "Actuation Port" (`q_ext`). This allows exogenous, continuous-time therapeutic signals (from the Bio-Blade hardware) to be superimposed onto the biological system to test if its physical reachability is still intact.

This requires updates to 4 files:
1. `src/echo/physics/thermostat.py`
2. `src/echo/primitives/thermalizer.py`
3. `src/echo/architecture/observer.py`
4. `src/echo/architecture/hierarchy.py`

---

### PART 1: The Thermostat
Modify `src/echo/physics/thermostat.py`.
- Update the `__call__` method to accept an optional argument `q_ext: jax.Array | None = None`.
- Ensure `omega_ext` is also still supported if it was added previously.
- The deterministic drift is now: 
  `drift_total = drift + (omega_ext if omega_ext is not None else 0.0) + (q_ext if q_ext is not None else 0.0)`
- Update the Euler-Maruyama step to use `drift_total`.

### PART 2: Thermalizer & Torx Factors
Modify `src/echo/primitives/thermalizer.py`.
- In `ThermoFlowFactor.__init__`, add `"q_ext": jax.ShapeDtypeStruct((d_state,), jnp.float32)` to `self.input_ports`.
- In `ThermoFlowFactor.sample`, extract `q_ext = inputs.get("q_ext", jnp.zeros(self.d_state))` and pass it to the thermostat.
- In `TorxThermalizer.__init__`, update the DFG `input_ports` and the Site `parents`/`porting_fn` to map `"q_ext_constant"` to `"q_ext"`. In `__call__`, pass `q_ext_constant=zeros`.
- In `ForcedTorxThermalizer.__call__`:
  - Add `q_seq: jax.Array | None = None` to the signature.
  - In the `step_fn`, if `q_seq` is provided, pass `q_frame` into the `inputs` dict. If not, pass zeros. 
  - Ensure the `jax.lax.scan` elegantly handles any combination of `seq`, `omega_seq`, and `q_seq` being None or provided.

### PART 3: Hierarchical Factors
Modify `src/echo/architecture/hierarchy.py`.
- In `HierarchicalThermoFlowFactor.__init__`, add `"q_ext": jax.ShapeDtypeStruct((self.d_micro + self.d_macro,), jnp.float32)` to `self.input_ports`.
- In `.sample()`, extract `q_ext`, split it into `q_micro` and `q_macro` (just like `omega_ext`), and pass them to their respective thermostats.

### PART 4: Observer Exposure
Modify `src/echo/architecture/observer.py` and `src/echo/architecture/hierarchy.py`.
- Update `forced_unroll` signatures in both `MarkovBlanketObserver` and `PredictiveCodingGraph` to accept `q_seq: jax.Array | None = None`.
- Pass `q_seq` down into `self.forced_thermalizer`.

TESTING:
Create `tests/echo/physics/test_bioblade_actuation.py`. 
- Instantiate a `Thermostat` with `temperature = 0.0`. 
- Provide a `q_ext` vector of `[5.0, -5.0]`. Assert the final state mathematically equals `x + (drift * dt) + (q_ext * dt)`.
- Instantiate an Observer, call `forced_unroll` with ONLY `q_seq` provided (and `seq=None`, `omega_seq=None`), and assert it executes without JAX concretization or shape errors.

Write production-grade JAX. Maintain all type hints and pure functional paradigms.

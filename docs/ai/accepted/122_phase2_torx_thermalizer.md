ROLE: You are an elite Scientific Machine Learning Engineer specializing in JAX, Equinox, and the Torx probabilistic programming framework.

TASK: We are building `src/echo/primitives/thermalizer.py` (append to the existing file). Implement a robust, production-ready class called `TorxThermalizer` that acts as the continuous-time compiler for our biological simulation.

MATHEMATICAL CONSTRAINTS:
1. The `TorxThermalizer` must take our single-step `ThermoFlowFactor` and unroll it sequentially over `n_steps` to simulate continuous time.
2. Because this simulates an autonomous biological entity, the state `x` at time `t` must feed directly into the state `x` at time `t+1`.
3. The time step `dt` must remain constant across all steps.

TECHNICAL REQUIREMENTS:
- Do NOT use PyTorch. Inherit from `equinox.Module`.
- The `__init__` method must accept:
  - `flow_factor`: An instantiated `ThermoFlowFactor`.
  - `n_steps`: int (The number of integration steps to unroll).
  - `d_state`: int (The dimension of the biological state vector).
  - Store `n_steps` using `equinox.field(static=True)`.
- Inside `__init__`, construct a `torx.ChainFactor`:
  - `base=flow_factor`
  - `n_steps=n_steps`
  - `feedback_porting_fn="x"` (Crucial: This tells Torx to feed the output `x_next` back into the input port `"x"` for the next step).
  - `weight_tied=True` (Ensures the EBM and physics parameters remain identical across time steps).
- After creating the `ChainFactor`, wrap it in a `torx.DFG`:
  - Define a Site: `torx.Site(name="chain", factor=chain_factor, parents=("x_init", "dt_constant"), porting_fn=("x", "dt"), param_key=None, info_key=None, site_info=None)`.
  - Define `input_ports`: `{"x_init": jax.ShapeDtypeStruct((d_state,), jnp.float32), "dt_constant": jax.ShapeDtypeStruct((), jnp.float32)}`.
  - Set `output_name="chain"`.
  - Store this compiled graph as `self.graph`.
- The `__call__(self, key: jax.random.PRNGKey, x_init: jax.Array, dt: float)` method must execute the simulation:
  - Construct the `inputs` dictionary: `{"x_init": x_init, "dt_constant": dt}`.
  - Call `self.graph.sample(key, inputs=inputs, params={})`.
  - Return the resulting trajectory or final state (Torx `ChainFactor` naturally handles this).
- Decorate `__call__` with `@equinox.filter_jit` to ensure the entire unrolled simulation is compiled into a single XLA artifact.
- Add robust type hinting using `jaxtyping` and `jax.Array`. Default to `jnp.float32`.

TESTING:
Append to `tests/echo/primitives/test_thermalizer.py`. Write a rigorous `pytest` suite that:
1. Instantiates the full stack: `SolenoidalFlow`, `DissipativeFriction`, `Thermostat`, `PrecisionWeightedEBM`, and `ThermoFlowFactor` (with `d_state=4`).
2. Instantiates `TorxThermalizer` wrapping the factor with `n_steps=10`.
3. Passes a random initial state `x_init` and `dt=0.01` through the thermalizer's `__call__` method.
4. Asserts that the output contains no NaNs and has the expected shape.
5. VRAM/XLA Proof: Proves that calling the thermalizer twice executes the `@equinox.filter_jit` compiled graph without throwing a `ConcretizationTypeError` (proving that `n_steps` was correctly handled as a static field).
6. Backpropagation Through Time (BPTT): Runs a dummy gradient check using `equinox.filter_value_and_grad` on a loss function wrapping the `__call__` method. Assert that gradients flow seamlessly through the unrolled `jax.lax.scan` loop back into the `PrecisionWeightedEBM` weights without crashing.

Write production-grade code with clean docstrings. Focus strictly on this module, do not hallucinate outside components.

## ROLE
You are an elite Scientific Machine Learning Engineer specializing in JAX, Equinox, and the Torx probabilistic programming framework.

## TASK
We are building `src/echo/primitives/thermalizer.py`. Implement a robust, production-ready class called `ThermoFlowFactor` that inherits from `torx.factor.AbstractReferenceFactor`. This factor represents a single, stochastic continuous-time update step of our biological simulation.

## MATHEMATICAL CONSTRAINTS
In its `.sample()` method, this factor must evaluate the following Stochastic Differential Equation (SDE) using the Euler-Maruyama method:
1. Compute the local slope of the energy landscape: `grad_E = jax.grad(E_theta)(x)`
2. Compute the deterministic drift: `drift = -(Q - Γ) @ grad_E`
3. Execute the physical integration (Thermostat): `x_next = x + (drift * dt) + diffusion`
*(Note: We will utilize the `Thermostat` module built previously to handle the exact drift and diffusion computation).*

## TECHNICAL REQUIREMENTS
- Do NOT use PyTorch. The class must inherit from `torx.factor.AbstractReferenceFactor`.
- The `__init__` method must accept and store instances of the components built previously: 
  - `ebm`: An instance of `PrecisionWeightedEBM`.
  - `solenoidal`: An instance of `SolenoidalFlow`.
  - `dissipative`: An instance of `DissipativeFriction`.
  - `thermostat`: An instance of `Thermostat`.
  - `d_state`: int
- You must declare Torx port specifications as static fields using `equinox.field(static=True)`:
  - `input_ports`: A dictionary `{"x": jax.ShapeDtypeStruct((d_state,), jnp.float32), "dt": jax.ShapeDtypeStruct((), jnp.float32)}`.
  - `output_spec`: `jax.ShapeDtypeStruct((d_state,), jnp.float32)`.
- The `sample(self, key, inputs, params, info=None, site_info=None, return_aux=False)` method must:
  - Extract `x` and `dt` from the `inputs` dictionary.
  - Define a closure `energy_fn(state)` that calls `self.ebm(state)` and returns *only* the first element (the scalar energy).
  - Use `jax.grad(energy_fn)` to compute `grad_E` at the current state.
  - Compute the physical matrices: `Q = self.solenoidal.Q` and `L = jnp.tril(self.dissipative.W)`.
  - Call the thermostat to perform the update: `x_next = self.thermostat(x=x, grad_E=grad_E, Q=Q, L=L, dt=dt, key=key)`.
  - Return `(x_next, None)` if `return_aux` is True, otherwise return `x_next`.
- The `init_params(self, key)` method can just return `None` (or an empty dict), as our state is tracked entirely by the Equinox modules stored on `self`.

## TESTING
Create `tests/echo/primitives/test_thermalizer.py`. Write a rigorous `pytest` suite that:
1. Instantiates mock or real instances of `SolenoidalFlow`, `DissipativeFriction`, `Thermostat`, and `PrecisionWeightedEBM` (with `d_state=4`).
2. Instantiates the `ThermoFlowFactor`.
3. Creates a dummy `inputs` dictionary containing `x` (shape `(4,)`) and `dt` (scalar `0.1`).
4. Invokes `.sample(key, inputs, params={})` and asserts the output `x_next` has shape `(4,)` and contains no NaNs.
5. Invokes `.sample()` with `return_aux=True` and asserts it correctly returns a tuple `(x_next, None)`.
6. Wires the `ThermoFlowFactor` into a minimal Torx `DFG` (Directed Factor Graph) and asserts that calling `graph.sample()` executes successfully.
7. Differentiability Test: Defines a dummy loss function wrapping `graph.sample`, and asserts that `equinox.filter_value_and_grad` successfully computes gradients with respect to the `ThermoFlowFactor`'s internal EBM, Engine, and Brake parameters.

Write production-grade code with clean docstrings. Assume the physics and EBM modules are importable from `src.echo.physics` and `src.echo.primitives`.

## ROLE

You are an elite Scientific Machine Learning Engineer specializing in JAX,
Equinox, and Torx.

## TASK

We are building `src/echo/architecture/hierarchy.py`. Implement a robust,
production-ready Equinox module called `PredictiveCodingGraph` that couples two
`MarkovBlanketObserver`s (a Micro level and a Macro level) into a nested
hierarchical predictive coding network.

## MATHEMATICAL CONSTRAINTS

1. We bypass manual message passing by defining a Joint Free Energy (F) for the
   coupled system.
2. The Micro state has its base energy: `E_micro, _ = micro_ebm(x_micro)`.
3. The Macro state has its base energy and precision:
   `E_macro, Pi_macro = macro_ebm(x_macro)`.
4. Top-Down Coupling: The Macro state generates a contextual belief about the
   Micro state via a linear projection layer: `belief = W_down @ x_macro`.
5. The Joint Free Energy adds a precision-weighted penalty pulling the Micro
   state toward the Macro belief:
   `F(x_micro, x_macro) = E_micro + E_macro + 0.5 * (x_micro - belief)^T @ Pi_macro @ (x_micro - belief)`
6. By defining this joint scalar function and using
   `jax.grad(..., argnums=(0, 1))`, XLA automatically generates both the
   Top-Down Precision Force (steepening the micro basin) and the Bottom-Up
   Surprisal Force (pushing prediction errors up to the macro state).

## TECHNICAL REQUIREMENTS

- Do NOT use PyTorch. Inherit from `equinox.Module`.
- First, implement a custom Torx factor called `HierarchicalThermoFlowFactor`
  (inheriting from `torx.factor.AbstractReferenceFactor`).
  - `__init__` must accept the unrolled components (Hulls, EBMs,
    SolenoidalFlows, DissipativeFrictions, Thermostats) for both Micro and
    Macro, plus the `W_down` linear layer, and `d_micro`, `d_macro` (ints).
  - To perfectly reuse our existing `TorxThermalizer`, `input_ports` must take
    `{"x": jax.ShapeDtypeStruct((d_micro + d_macro,), jnp.float32), "dt": jax.ShapeDtypeStruct((), jnp.float32)}`.
  - `output_spec` must return
    `jax.ShapeDtypeStruct((d_micro + d_macro,), jnp.float32)`.
  - Inside `.sample()`: a) Split the input `x` into `x_micro` and `x_macro`. b)
    Define a closure `joint_energy_fn(x_u, x_m)` that evaluates the EBMs,
    computes the scalar `F` defined in Constraint 5, and returns it. c) Compute
    both gradients simultaneously:
    `grad_micro, grad_macro = jax.grad(joint_energy_fn, argnums=(0, 1))(x_micro, x_macro)`.
    d) Apply the respective Hull masks to Q and Gamma for BOTH levels (using
    `get_topology_mask()`). e) Compute the safe diffusion matrix `S` (using
    `jnp.linalg.eigh` on the masked Gamma matrices) for BOTH levels, just like
    in the single observer. f) Execute the `Thermostat` step for both levels
    independently using their respective gradients, masked Q, S (passed as L),
    and dt. g) Concatenate the updated states `[x_micro_next, x_macro_next]` and
    return.
- Second, implement `PredictiveCodingGraph(equinox.Module)`.
  - `__init__` accepts two pre-instantiated `MarkovBlanketObserver`s
    (`micro_observer` and `macro_observer`), `n_steps: int`, and a
    `key: jax.random.PRNGKey`.
  - Extract `d_micro` and `d_macro` from their respective hulls.
  - Instantiate
    `self.W_down = equinox.nn.Linear(d_macro, d_micro, use_bias=False, key=key)`.
  - Extract the components from the two observers, instantiate
    `HierarchicalThermoFlowFactor`, and wrap it in a `TorxThermalizer` (where
    `d_state = d_micro + d_macro`).
  - The `__call__(self, key, x_micro_init, x_macro_init, dt)` method
    concatenates the initial states, passes them to the thermalizer, and returns
    the unrolled joint trajectory.

## TESTING

Create `tests/echo/architecture/test_hierarchy.py`. Write a rigorous `pytest`
suite that:

1. Instantiates a Micro `MarkovBlanketObserver` (d_state=4) and a Macro
   `MarkovBlanketObserver` (d_state=6).
2. Instantiates the `PredictiveCodingGraph` with `n_steps=3`.
3. Execution Test: Passes random initial states and `dt=0.01`. Asserts the
   output trajectory shape is `(3, 10)` and has no NaNs.
4. The BPTT Cross-Talk Proof (The Acid Test):
   - Define a loss function that sums the MACRO portion of the final output
     state from the simulation.
   - Use `equinox.filter_value_and_grad` to compute the gradient of this
     macro-loss with respect to `x_micro_init`.
   - Assert that `jnp.linalg.norm(grad_micro) > 0.0`. This mathematically proves
     Bottom-Up Surprisal (the micro state perturbed the future macro state).
   - Define a second loss function summing the MICRO portion of the final state,
     and compute its gradient w.r.t `x_macro_init`.
   - Assert `jnp.linalg.norm(grad_macro) > 0.0`. This proves Top-Down
     Enslavement (the macro belief physically altered the micro trajectory).
5. Asserts the graph compiles with `jax.jit`.

Write production-grade code with clean docstrings. Focus strictly on this
module.

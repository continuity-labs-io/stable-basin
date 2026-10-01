# Optimize O(D^3) Eigendecomposition Bottleneck

As identified in the audit (`AUDIT_ea136a5.md`), we need to optimize the JAX physics engine by hoisting static matrix computations out of the `jax.lax.scan` loop (our ODE solver). Specifically, XLA's Loop Invariant Code Motion (LICM) struggles to hoist `jnp.linalg.eigh` (an O(D^3) solver) and custom block slice assignments, causing these heavy operations to re-execute on every single timestep of the ODE integration.

## Proposed Changes

### Modify `src/echo/primitives/thermalizer.py`
We need the unroller loops to accept external factor parameters and pass them through to the inner `torx` graph.
- Modify `TorxThermalizer.__call__` and `ForcedTorxThermalizer.__call__` to accept a new `factor_params: dict = None` argument.
- Inside the unrollers, pass `{"chain": factor_params}` (for DFG) or `factor_params` into the step function so that the Torx factor receives them during `sample()`.

### Modify `src/echo/architecture/observer.py`
- Add a `precompute(self)` method to `MaskedThermoFlowFactor` that computes and returns a dict with `Q`, `L`, and `Gamma`.
- Update `MaskedThermoFlowFactor.sample()` to check `params` for `Q`, `L`, and `Gamma`. If present, use them; otherwise, fallback to computing them.
- In `MarkovBlanketObserver.__call__` and `forced_unroll()`, call `self.thermalizer.flow_factor.precompute()` and pass the result as `factor_params` into the thermalizer.

### Modify `src/echo/architecture/hierarchy.py`
- Add a `precompute(self)` method to `HierarchicalThermoFlowFactor` that computes `Q_micro`, `Gamma_micro`, `S_micro` (using the faster `jnp.linalg.cholesky` instead of `jnp.linalg.eigh`), and the corresponding macro matrices.
- Update `HierarchicalThermoFlowFactor.sample()` to pull these from `params`, avoiding the O(D^3) loop bottleneck.
- In `PredictiveCodingGraph.__call__` and `forced_unroll()`, evaluate `precompute()` and pass it into the thermalizers.

## Verification Plan
### Automated Tests
- Run `pytest tests/echo/architecture/` and `pytest tests/echo/primitives/` to ensure mathematical output remains strictly identical.
- Ensure no `jnp.linalg.eigh` calls remain inside the factor `sample()` loops, dramatically decreasing compiled graph sizes and simulation latency.

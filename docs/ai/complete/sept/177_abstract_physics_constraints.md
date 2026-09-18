# Goal Description
Refactor the codebase to isolate explicit topological constraints (like Markov Blanket boundaries) into the physics primitives (`DissipativeFriction` and `SolenoidalFlow`). This replaces the naive Hadamard masking of the `Gamma` matrix with explicit block construction of the Cholesky factor (`L`), ensuring that Positive Semi-Definiteness (PSD) is strictly maintained, while simultaneously simplifying higher-level architectural classes like `hierarchy.py` and `observer.py`.

## Proposed Changes

### `src/echo/physics/dissipative.py`
Modify `DissipativeFriction` to optionally accept a `MarkovHull`.
- Add an `L` property that returns the lower-triangular Cholesky factor.
- If a `MarkovHull` is provided, explicitly zero out the External-Internal block of `L` (`L.at[idx_e:, :idx_s].set(0.0)`) to cleanly enforce the topological boundary without destroying PSD.
- Update the `Gamma` property to use `self.L @ self.L.T`.

### `src/echo/physics/solenoidal.py`
Modify `SolenoidalFlow` to optionally accept a `MarkovHull`.
- If provided, mask the anti-symmetric `Q` matrix with `hull.get_topology_mask()` internally.

### `src/echo/architecture/observer.py`
Clean up the `Observer` class to rely on the physics layer for constraints.
- Pass `self.hull` to `DissipativeFriction` and `SolenoidalFlow` during initialization.
- Remove the verbose `jnp.block` manual masking and simply use `self.solenoidal.Q` and `self.dissipative.L`.

### `src/echo/architecture/hierarchy.py`
Clean up the hierarchical flow factor to rely on the physics layer.
- Pass `micro_hull` and `macro_hull` into the physics primitives during initialization.
- Completely remove the naive Hadamard masking (`Gamma_micro_orig * M_micro`) during the forward pass and replace it with direct calls to the constrained `Gamma` properties.

### `src/echo/primitives/thermalizer.py`
Ensure the thermalizer uses the block-constructed Cholesky factor.
- Update `L = jnp.tril(self.dissipative.W)` to `L = self.dissipative.L` so it correctly receives the constrained block matrix.

## Verification Plan
### Automated Tests
- Run tests in `pytest` to ensure no matrix shape mismatches occur.
- Specifically verify that no `NaN` values are generated during backward passes (which would happen if `Gamma` lost its PSD properties).
